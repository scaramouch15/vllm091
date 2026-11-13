"""Local-only CPU scorer for the paper's OPT rank checkpoint.

This module is not wired into the API or scheduler yet. It has no import-time
model loading and does not download model/tokenizer resources.
"""
import hashlib
import logging
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


def _diagnostic(level, message, *args):
    """Diagnostics must not turn a successful model call into a failure."""
    try:
        logger.log(level, message, *args)
    except Exception:
        # Ordinary logging has no durable loss counter; audit its absence.
        pass

EXPECTED_MODEL_SHA256 = 'e7739d26217ee23eb14b6282ee5088f9c79bdc477f456c027b7ebcffacde61ab'
MODEL_REVISION = '39df2b41ffe88d5ed967c6035d3838b5b5960379'
TOKENIZER_REVISION = '27dcfa74d334bc871f3234de431e71c6eeba5dd6'
TOKENIZER_SHA256 = {
    'config.json': '54162c9cd0377110df1859dfabebb106fc0988c8aadb88207ce1ee0d34985df6',
    'merges.txt': '1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5',
    'vocab.json': '06b4d46c8e752d410213d9548eb27a54db70fda0319b6271fb8d59dead5e1cab',
    'special_tokens_map.json': '0f3cd5f307a2735cfee31ddf33be88655e8be165ee2cdc26ced966f04258adf0',
    'tokenizer_config.json': 'd1e4185b43a0a06421ab7218d5c1a775350bd13024507b34fbed25dcf6c14c6b',
}


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class ScoreResult:
    request_id: str
    score: float
    tokenization_ns: int
    inference_ns: int


class PaperLTRScorer:
    """Synchronous, single-request CPU scorer; call outside scheduler.

    Threading, admission, cancellation, timeout and logging are deliberately not
    supplied by this class. Callers must design those before wiring this in.
    """
    def __init__(self, model_dir, tokenizer_dir):
        model_dir, tokenizer_dir = Path(model_dir), Path(tokenizer_dir)
        if file_sha(model_dir / 'model.safetensors') != EXPECTED_MODEL_SHA256:
            raise ValueError('LTR checkpoint SHA256 mismatch')
        for name, expected in TOKENIZER_SHA256.items():
            if file_sha(tokenizer_dir / name) != expected:
                raise ValueError(f'LTR tokenizer SHA256 mismatch: {name}')
        import torch
        import transformers
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.transformers_version = transformers.__version__
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_dir), local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(model_dir), local_files_only=True, trust_remote_code=False,
            use_safetensors=True).to('cpu').eval()
        if self.model.config.model_type != 'opt' or self.model.config.num_labels != 1:
            raise ValueError('LTR model schema mismatch')

    def score(self, request_id, prompt):
        if not isinstance(request_id, str) or not request_id:
            raise ValueError('request_id must be a nonempty string')
        if not isinstance(prompt, str):
            raise ValueError('prompt must be text')
        start = time.perf_counter_ns()
        encoded = self.tokenizer(prompt, max_length=2048, truncation=True,
                                 padding=True, return_tensors='pt')
        tokenization_ns = time.perf_counter_ns() - start
        start = time.perf_counter_ns()
        with self.torch.inference_mode():
            logits = self.model(**encoded).logits
        inference_ns = time.perf_counter_ns() - start
        if tuple(logits.shape) != (1, 1):
            raise ValueError('unexpected LTR prediction shape')
        score = float(logits[0, 0])
        if not math.isfinite(score):
            raise ValueError('non-finite LTR prediction')
        return ScoreResult(request_id, score, tokenization_ns, inference_ns)

class ScoringUnavailable(RuntimeError):
    """An LTR score could not be obtained safely for this request."""


class ModelExecutionTimeout(RuntimeError):
    """A model raised TimeoutError; this is not the caller's deadline."""


def scoring_failure_reason(exc):
    import asyncio
    if isinstance(exc, asyncio.CancelledError):
        return 'cancelled'
    if isinstance(exc, asyncio.TimeoutError):
        return 'timeout'
    known = {'queue_full', 'duplicate_request_id', 'scorer_closed',
             'invalid_score_identity'}
    if isinstance(exc, ScoringUnavailable) and str(exc) in known:
        return str(exc)
    return 'model_exception'


class BoundedAsyncLTRScorer:
    """Bounded single-worker adapter; not yet attached to the API lifecycle.

    A permit covers queued or executing work. A timed-out/cancelled running
    job retains its permit and ID until its underlying thread finishes.
    """
    def __init__(self, scorer, max_inflight=8):
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        if type(max_inflight) is not int or max_inflight < 1:
            raise ValueError('max_inflight must be a positive integer')
        self._scorer = scorer
        self._max_inflight = max_inflight
        self._permits = asyncio.Semaphore(max_inflight)
        self._executor = ThreadPoolExecutor(max_workers=1,
                                             thread_name_prefix='r1-ltr')
        self._ids = set()
        self._closed = False

    async def score(self, request_id, prompt, timeout_seconds):
        import asyncio
        if self._closed:
            raise ScoringUnavailable('scorer_closed')
        if (not isinstance(request_id, str) or not request_id
                or not isinstance(prompt, str)):
            raise ValueError('invalid request identity or prompt')
        if (type(timeout_seconds) not in (float, int)
                or not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
            raise ValueError('timeout_seconds must be finite and positive')
        if request_id in self._ids:
            raise ScoringUnavailable('duplicate_request_id')
        # Do not create an unbounded semaphore-waiting queue in the event loop.
        if len(self._ids) >= self._max_inflight:
            raise ScoringUnavailable('queue_full')
        # Reserve the ID before the first await so duplicate callers cannot race.
        self._ids.add(request_id)
        started = time.monotonic()
        submitted_ns = time.perf_counter_ns()
        late = threading.Event()
        try:
            await asyncio.wait_for(self._permits.acquire(), timeout_seconds)
        except BaseException:
            self._ids.remove(request_id)
            raise
        if self._closed:
            self._permits.release()
            self._ids.remove(request_id)
            raise ScoringUnavailable('scorer_closed')
        loop = asyncio.get_running_loop()
        def run_job():
            worker_start = time.perf_counter_ns()
            _diagnostic(logging.INFO, "R1_LTR diagnostic_v1 request_id=%s event=worker_started queue_wait_ns=%d",
                        request_id, worker_start - submitted_ns)
            try:
                result = self._scorer.score(request_id, prompt)
                if isinstance(result, ScoreResult):
                    _diagnostic(logging.INFO, "R1_LTR diagnostic_v1 request_id=%s event=worker_finished tokenizer_ns=%s inference_ns=%s late=%s",
                                request_id, result.tokenization_ns, result.inference_ns, late.is_set())
                else:
                    _diagnostic(logging.WARNING, "R1_LTR diagnostic_v1 request_id=%s event=worker_finished invalid_result=true late=%s",
                                   request_id, late.is_set())
                return result
            except Exception as exc:
                _diagnostic(logging.WARNING, "R1_LTR diagnostic_v1 request_id=%s event=worker_failed error_type=%s late=%s",
                               request_id, type(exc).__name__, late.is_set())
                if isinstance(exc, TimeoutError):
                    raise ModelExecutionTimeout('model_raised_timeout') from exc
                raise
        try:
            _diagnostic(logging.INFO, "R1_LTR diagnostic_v1 request_id=%s event=submitted", request_id)
            job = self._executor.submit(run_job)
        except BaseException:
            self._permits.release()
            self._ids.remove(request_id)
            raise

        def release(_):
            def on_loop():
                self._ids.discard(request_id)
                self._permits.release()
            try:
                loop.call_soon_threadsafe(on_loop)
            except RuntimeError:
                # A closed event loop cannot accept further requests.
                pass
        job.add_done_callback(release)
        remaining = timeout_seconds - (time.monotonic() - started)
        wrapped = asyncio.wrap_future(job)
        # After caller timeout, retrieve a later exception to avoid an
        # unobserved-future warning; this does not consume successful results.
        wrapped.add_done_callback(lambda f: None if f.cancelled() else f.exception())
        try:
            result = await asyncio.wait_for(
                asyncio.shield(wrapped), max(0.0, remaining))
        except BaseException as exc:
            # This succeeds only before the worker starts; running work completes
            # in the sole worker and is discarded by the cancelled caller.
            late.set()
            was_running = job.running()
            job.cancel()
            _diagnostic(logging.INFO, "R1_LTR diagnostic_v1 request_id=%s event=caller_exit reason=%s error_type=%s foreground_ns=%d still_running=%s",
                        request_id, 'cancelled' if isinstance(exc, asyncio.CancelledError)
                        else 'timeout' if isinstance(exc, asyncio.TimeoutError) else 'model_exception',
                        type(exc).__name__,
                        time.perf_counter_ns() - submitted_ns, was_running)
            raise
        if (not isinstance(result, ScoreResult) or result.request_id != request_id
                or type(result.score) not in (int, float)
                or not math.isfinite(result.score)):
            raise ScoringUnavailable('invalid_score_identity')
        return result

    async def close(self):
        import asyncio
        if not self._closed:
            self._closed = True
            await asyncio.to_thread(self._executor.shutdown, wait=True,
                                    cancel_futures=True)

    def shutdown(self):
        """Reject new work and cancel queued jobs without blocking the event loop."""
        self._closed = True
        self._executor.shutdown(wait=False, cancel_futures=True)


class LTRAdmissionLogger:
    """Logging sink for admission without enabling B shadow predictions."""
    def __init__(self, logger):
        self.logger = logger

    def emit(self, kind, values):
        import json
        self.logger.info("R1_LTR %s %s", kind, json.dumps(values, default=str))

    def observe_elapsed(self, name, start, outcome):
        self.logger.info("R1_LTR %s elapsed_ns=%d outcome=%s", name,
                         time.perf_counter_ns() - start, outcome)

LTR_INTERNAL_KEY = '_vllm_r1_ltr_score_v1'


def consume_score(params, request_id):
    """Consume a frontend-only typed result, bound to its exact request ID.

    A JSON-provided dictionary cannot impersonate the in-process ScoreResult.
    Invalid and forged values are removed before serialization and return None.
    """
    if not params.extra_args or LTR_INTERNAL_KEY not in params.extra_args:
        return None
    params.extra_args = dict(params.extra_args)
    value = params.extra_args.pop(LTR_INTERNAL_KEY)
    if (type(value) is not ScoreResult or value.request_id != request_id
            or type(value.score) not in (int, float)
            or not math.isfinite(value.score)):
        return None
    return float(value.score)
