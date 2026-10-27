# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in CPU shadow inference. No queue, allocation or sampling decisions.

Numerical dependencies are imported only by Predictor at enabled startup.
The immutable envelope is local to the frontend and removed before Core IPC.
"""
from __future__ import annotations

import hashlib
import logging
import math
import os
import platform
import re
import sys
import time
from array import array
from collections import Counter
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

MODEL_SHA256 = "23aff057712e7bec05db91d470af39a1b34eabddbf5852914710e1b8e4b96e1f"
INTERNAL_KEY = "_vllm_r1_shadow_static_v1"
QUANTILES = [0.5, 0.9, 0.95]
SORT_RULE = "p90_ascending_stable_v1"
SORT_TIMING_VERSION = 1
LENGTH_FIELDS = ["num_prompt_tokens"] + [f"text_{i}" for i in range(32)]
TIME_FIELDS = ["num_prompt_tokens", "running_count", "kv_usage",
               "pred_output_q50", "pred_output_q90", "pred_output_q95"]
logger = logging.getLogger(__name__)


def stable_sort_indices(keys):
    return sorted(range(len(keys)), key=keys.__getitem__)


def require(condition, message):
    if not condition:
        raise ValueError("R1 shadow: " + message)


class DurationHistogram:
    """Fixed 497 uint64 counters, O(1) update, no raw samples or RNG.

    0..15 ns are exact; larger bins have eight subdivisions per power of two.
    The last bin handles durations >= 2**64 ns without growing the array.
    Quantiles are nearest-rank bucket bounds, computed only at shutdown.
    """
    NUM_BINS = 497

    def __init__(self):
        self.bins = array("Q", [0]) * self.NUM_BINS
        self.n = self.total_ns = 0
        self.min_ns = self.max_ns = None

    def observe(self, duration_ns):
        if type(duration_ns) is not int or duration_ns < 0:
            raise ValueError("duration must be nonnegative integer nanoseconds")
        if duration_ns >= 1 << 64:
            index = self.NUM_BINS - 1
        else:
            shift = max(0, duration_ns.bit_length() - 4)
            index = (duration_ns if shift == 0 else
                     16 + (shift - 1) * 8 + (duration_ns >> shift) - 8)
        self.bins[index] += 1
        self.n += 1
        self.total_ns += duration_ns
        self.min_ns = duration_ns if self.min_ns is None else min(self.min_ns, duration_ns)
        self.max_ns = duration_ns if self.max_ns is None else max(self.max_ns, duration_ns)

    def report(self):
        occupied = []
        for index, count in enumerate(self.bins):
            if not count:
                continue
            if index < 16:
                low = high = index
            elif index == self.NUM_BINS - 1:
                low, high = 1 << 64, self.max_ns
            else:
                shift = 1 + (index - 16) // 8
                low = (8 + (index - 16) % 8) << shift
                high = low + (1 << shift) - 1
            occupied.append({"index": index, "count": count,
                             "lower_ns": max(low, self.min_ns),
                             "upper_ns": min(high, self.max_ns)})
        result = {"n": self.n, "total_ns": self.total_ns,
                  "min_ns": self.min_ns, "max_ns": self.max_ns,
                  "bins": occupied}
        for percentile in (50, 95, 99):
            bounds = None
            if self.n:
                rank = (self.n * percentile + 99) // 100
                cumulative = 0
                for bucket in occupied:
                    cumulative += bucket["count"]
                    if cumulative >= rank:
                        bounds = [bucket["lower_ns"], bucket["upper_ns"]]
                        break
            result[f"p{percentile}_bounds_ns"] = bounds
            result[f"p{percentile}_upper_ns"] = bounds[1] if bounds else None
        return result

    def merge(self, other):
        """Cold-path merge for overall stage reports; no repeated hot updates."""
        if not other.n:
            return
        for index, count in enumerate(other.bins):
            self.bins[index] += count
        self.total_ns += other.total_ns
        self.min_ns = other.min_ns if not self.n else min(self.min_ns, other.min_ns)
        self.max_ns = other.max_ns if not self.n else max(self.max_ns, other.max_ns)
        self.n += other.n


@dataclass(frozen=True)
class ShadowConfig:
    model: Path
    run_id: str
    directory: Path
    candidates: int = 32
    scan_limit: int = 128
    every: int = 8
    sort: bool = False

    @classmethod
    def from_env(cls):
        enabled = os.environ.get("VLLM_R1_SHADOW", "0")
        if enabled == "0":
            return None
        require(enabled == "1", "VLLM_R1_SHADOW must be 0 or 1")
        model = os.environ.get("VLLM_R1_SHADOW_MODEL", "")
        run = os.environ.get("VLLM_R1_SHADOW_RUN_ID", "")
        directory = os.environ.get("VLLM_R1_SHADOW_DIR", "")
        require(model and directory, "SHADOW_MODEL and SHADOW_DIR are required")
        require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", run),
                "SHADOW_RUN_ID must be a safe 1-128 character identifier")
        cap = int(os.environ.get("VLLM_R1_SHADOW_CANDIDATES", "32"))
        scan = int(os.environ.get("VLLM_R1_SHADOW_SCAN_LIMIT", "128"))
        every = int(os.environ.get("VLLM_R1_SHADOW_EVERY", "8"))
        sort = os.environ.get("VLLM_R1_SHADOW_SORT", "0")
        require(sort in ("0", "1"), "VLLM_R1_SHADOW_SORT must be 0 or 1")
        require(1 <= cap <= scan <= 4096 and every > 0,
                "require 1 <= candidates <= scan_limit <= 4096, every > 0")
        return cls(Path(model).expanduser().resolve(), run,
                   Path(directory).expanduser().resolve(), cap, scan, every,
                   sort == "1")


@dataclass(frozen=True)
class StaticEnvelope:
    prompt_tokens: int
    max_tokens: int
    lengths: tuple[float, float, float]


def valid_lengths(values):
    return (isinstance(values, (tuple, list)) and len(values) == 3
            and all(type(x) in (float, int) and math.isfinite(x)
                    and 0 <= x <= 2048 for x in values)
            and list(values) == sorted(values))


def consume_static(params, prompt_tokens, shadow, request_id=None):
    """Called on Processor's cloned params. Preserve all other extra_args.

    JSON clients cannot construct StaticEnvelope; CompletionRequest only maps
    kv_transfer_params into extra_args. Even a forged dict is discarded.
    """
    start = time.perf_counter_ns() if shadow is not None else None
    outcome = "failed"
    try:
        if not params.extra_args or INTERNAL_KEY not in params.extra_args:
            outcome = "no_static"
            if shadow is None:
                return None
            shadow.counts["processor_without_static"] += 1
            shadow.emit("shadow_processor_skip", {"engine_request_id": request_id,
                                                  "reason": "no_static"})
            return None
        params.extra_args = dict(params.extra_args)
        value = params.extra_args.pop(INTERNAL_KEY)
        if shadow is None:
            return None
        if (type(value) is not StaticEnvelope or params.n != 1
                or value.prompt_tokens != prompt_tokens
                or value.max_tokens != params.max_tokens or params.max_tokens != 2048
                or not valid_lengths(value.lengths)):
            outcome = "mismatch"
            shadow.counts["processor_static_mismatch"] += 1
            shadow.emit("shadow_processor_skip", {"engine_request_id": request_id,
                                                  "reason": "static_mismatch"})
            return None
        outcome = "received"
        shadow.counts["processor_static_received"] += 1
        return tuple(float(x) for x in value.lengths)
    finally:
        if shadow is not None:
            shadow.observe_elapsed("processor_consume", start, outcome)


def validate_service(config):
    model = config.model_config
    expected = "Meta-Llama-3-8B-Instruct"
    require(Path(model.model.rstrip("/")).name == expected
            and Path(model.tokenizer.rstrip("/")).name == expected,
            "first release requires Meta-Llama-3-8B-Instruct model/tokenizer")
    hf = model.hf_config
    require(hf.model_type == "llama" and hf.hidden_size == 4096
            and hf.num_hidden_layers == 32 and hf.vocab_size == 128256,
            "unexpected Llama configuration")
    require(not model.is_multimodal_model and not model.is_encoder_decoder,
            "text decoder model required")
    require(not (model.encoder_config or {}).get("do_lower_case", False),
            "encoder text rewriting is unsupported")
    require(not config.kv_transfer_config and not config.speculative_config,
            "remote KV/speculative decoding are outside the first release")
    parallel = config.parallel_config
    require(parallel.data_parallel_size == 1
            and parallel.pipeline_parallel_size == 1
            and parallel.tensor_parallel_size == 1,
            "first release requires DP=PP=TP=1")
    scheduler = config.scheduler_config
    require(scheduler.policy == "fcfs"
            and scheduler.scheduler_cls == "vllm.v1.core.sched.scheduler.Scheduler",
            "first release requires the native V1 FCFS scheduler class")


class Predictor:
    """Retain only the role's heads after loading the trusted fixed B artifact."""

    def __init__(self, path, role):
        require(platform.python_version() == "3.10.22", "Python must be 3.10.22")
        require(hashlib.sha256(path.read_bytes()).hexdigest() == MODEL_SHA256,
                "B model SHA256 mismatch (checked before joblib deserialization)")
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                     "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            require(os.environ.get(name) == "1", name + " must be 1 before startup")
        import joblib
        import numpy as np
        import scipy
        import sklearn
        import threadpoolctl
        modules = ((np, "2.2.6"), (scipy, "1.15.3"),
                   (sklearn, "1.7.2"), (joblib, "1.5.3"),
                   (threadpoolctl, "3.6.0"))
        for module, version in modules:
            require(module.__version__ == version,
                    module.__name__ + " version mismatch")
        self.environment = {"python": platform.python_version(),
            "interpreter": sys.executable, "packages": {
                module.__name__: {"version": module.__version__,
                                  "file": module.__file__}
                for module, _ in modules}}
        bundle = joblib.load(path)
        require(bundle["policy"] == "tfidf", "unsupported model policy")
        for name, fields in (("length_head", LENGTH_FIELDS),
                             ("time_head", TIME_FIELDS)):
            head = bundle[name]
            require(head["kind"] == "tree" and head["fields"] == fields
                    and head["quantiles"] == QUANTILES
                    and head["scaler"] is None and len(head["models"]) == 3,
                    "unsupported " + name + " feature/quantile/scaler configuration")
        self.np = np
        self.encoder = bundle["encoder"] if role == "frontend" else None
        self.head = bundle["length_head" if role == "frontend" else "time_head"]
        self.threadpools = threadpoolctl.threadpool_info()
        require(self.threadpools and all(p["num_threads"] == 1
                                        for p in self.threadpools),
                "startup numerical threadpools must be single-threaded")

    def processed(self, x):
        # Exact base_models.predict_raw / predict path: float64 input,
        # sklearn itself performs tree float32 conversion and <= comparisons.
        np = self.np
        raw = np.column_stack([m.predict(x) for m in self.head["models"]])
        require(np.isfinite(raw).all(), "nonfinite raw prediction")
        return np.sort(np.maximum(raw, 0), axis=1)

    def static(self, text, tokens):
        start = time.perf_counter_ns()
        sparse = self.encoder["vectorizer"].transform([text])
        tfidf = time.perf_counter_ns()
        z = self.encoder["svd"].transform(sparse)
        transformed = time.perf_counter_ns()
        x = self.np.asarray([[tokens, *[float(v) for v in z[0]]]],
                            dtype=self.np.float64)
        lengths = self.np.minimum(self.processed(x), 2048)[0].tolist()
        end = time.perf_counter_ns()
        require(valid_lengths(lengths), "invalid length prediction")
        return tuple(lengths), {"tfidf_ns": tfidf - start,
                               "svd_ns": transformed - tfidf,
                               "text_transform_ns": transformed - start,
                               "length_head_ns": end - transformed}

    def assemble(self, candidates, running_count, kv_usage):
        return self.np.asarray([
            [r.num_prompt_tokens, running_count, kv_usage, *r.r1_shadow_lengths]
            for r in candidates], dtype=self.np.float64)

    def dynamic(self, matrix):
        result = self.processed(matrix)
        require(result.shape == (len(matrix), 3), "invalid time prediction shape")
        return result.tolist()


class Shadow:
    def __init__(self, config, predictor, writer):
        self.config, self.predictor, self.writer = config, predictor, writer
        self.counts = Counter()
        for outcome in ("success", "failed", "empty", "interval_skip"):
            self.counts["dynamic_outcome_" + outcome] = 0
        for outcome in ("success", "failed", "skipped"):
            self.counts["static_outcome_" + outcome] = 0
        for name in ("sort_attempts", "sort_success", "sort_errors"):
            self.counts[name] = 0
        self.duration_histograms = {}
        self.batch = 0
        self.closed = False

    def observe_elapsed(self, stage, start, outcome, candidate_count=None,
                        total_counter=None):
        # Deliberately no emit/observe recursion. The measurement ends here,
        # before its own histogram bookkeeping, which is separately counted.
        finished = time.perf_counter_ns()
        elapsed = finished - start
        key = (stage, outcome, candidate_count)
        histogram = self.duration_histograms.get(key)
        if histogram is None:
            histogram = self.duration_histograms[key] = DurationHistogram()
        histogram.observe(elapsed)
        if total_counter is not None:
            self.counts[total_counter] += elapsed
        self.counts["timing_bookkeeping_calls"] += 1
        self.counts["timing_bookkeeping_ns"] += time.perf_counter_ns() - finished

    def duration_report(self):
        overall = {}
        for (stage, _, _), histogram in self.duration_histograms.items():
            if stage not in overall:
                overall[stage] = DurationHistogram()
            overall[stage].merge(histogram)
        return {"schema_version": 1, "unit": "ns",
                "method": "nearest_rank_log_bucket_bounds_4_significant_bits",
                "bins_per_group": DurationHistogram.NUM_BINS,
                "overall": {stage: histogram.report()
                            for stage, histogram in overall.items()},
                "groups": [dict(stage=stage, outcome=outcome,
                                candidate_count=count, **histogram.report())
                           for (stage, outcome, count), histogram
                           in self.duration_histograms.items()]}

    def emit(self, kind, values):
        # No prompt, exception message or mutable Request is handed to writer.
        start = time.perf_counter_ns()
        outcome = "failed"
        try:
            ok = self.writer.emit(kind, dict(values, model_sha256=MODEL_SHA256))
            self.counts["log_enqueued" if ok else "log_dropped"] += 1
            outcome = "enqueued" if ok else "dropped"
        except Exception as exc:
            self.counts["log_errors"] += 1
            if self.counts["log_errors"] == 1:
                logger.error("R1 shadow log enqueue failed (%s); records incomplete; "
                             "further failures are counted", type(exc).__name__)
        self.counts["log_enqueue_calls"] += 1
        self.observe_elapsed("log_enqueue", start, outcome,
                             total_counter="log_enqueue_ns")

    def prepare(self, request, params, normalized, engine_prompt, request_id,
                lora=None, adapter=None):
        """One call per Completion prompt, before AsyncLLM / arrival_time."""
        start = time.perf_counter_ns()
        self.counts["completion_prompts"] += 1
        outcome = "failed"
        try:
            if getattr(params, "extra_args", None):
                params.extra_args = dict(params.extra_args)
                params.extra_args.pop(INTERNAL_KEY, None)
            reason = None
            if not isinstance(request.prompt, str) or request.prompt_embeds is not None:
                reason = "non_single_text"
            elif (request.n != 1 or request.use_beam_search
                  or getattr(params, "n", None) != 1):
                reason = "multiple_outputs_or_beam"
            elif request.truncate_prompt_tokens is not None:
                reason = "truncation"
            elif request.add_special_tokens:
                reason = "special_tokens"
            elif lora is not None or adapter is not None:
                reason = "adapter"
            elif (getattr(params, "max_tokens", None) != 2048
                  or getattr(params, "guided_decoding", None) is not None):
                reason = "max_tokens_or_guided"
            elif request.kv_transfer_params:
                reason = "remote_kv"
            elif (normalized.get("prompt") != request.prompt
                  or "prompt_token_ids" not in engine_prompt
                  or normalized.get("prompt_token_ids") != engine_prompt["prompt_token_ids"]):
                reason = "normalized_input_mismatch"
            if reason:
                outcome = "skipped"
                self.counts["skip_" + reason] += 1
                self.emit("shadow_static_skip", {"engine_request_id": request_id,
                                                 "reason": reason})
                return
            tokens = len(engine_prompt["prompt_token_ids"])
            require(tokens > 0, "empty token input")
            self.counts["static_attempts"] += 1
            lengths, timings = self.predictor.static(normalized["prompt"], tokens)
            require(valid_lengths(lengths), "invalid static result")
            params.extra_args = dict(params.extra_args or {})
            params.extra_args.pop(INTERNAL_KEY, None)
            params.extra_args[INTERNAL_KEY] = StaticEnvelope(tokens, 2048, lengths)
            self.counts["static_success"] += 1
            outcome = "success"
            self.emit("shadow_static", {"engine_request_id": request_id,
                "prompt_tokens": tokens, "max_tokens": 2048,
                "length_quantiles": lengths, **timings,
                "static_compute_ns": time.perf_counter_ns() - start})
        except Exception as exc:
            outcome = "failed"
            if getattr(params, "extra_args", None):
                params.extra_args.pop(INTERNAL_KEY, None)
            self.counts["static_errors"] += 1
            self.emit("shadow_static_error", {"engine_request_id": request_id,
                                              "error_type": type(exc).__name__})
        finally:
            self.counts["static_outcome_" + outcome] += 1
            self.observe_elapsed("static_foreground", start, outcome,
                                 total_counter="static_total_ns")

    def sweep(self, waiting, running, kv_manager, token_budget,
              scheduled_timestamp, decision_batch_id=None):
        """Read-only bounded traversal; never allocate KV or reorder either queue."""
        start = time.perf_counter_ns()
        self.batch += 1
        self.counts["waiting_opportunities"] += 1
        snapshot = None
        candidates = []
        outcome = "failed"
        try:
            running_count, kv_usage = len(running), float(kv_manager.usage)
            snapshot = {"decision_batch_id": decision_batch_id or
                        f"{self.writer.instance_id}:{self.batch}",
                        "scheduled_timestamp": scheduled_timestamp,
                        "running_count": running_count, "kv_usage": kv_usage,
                        "token_budget": int(token_budget)}
            state_read = time.perf_counter_ns()
            require(math.isfinite(kv_usage) and 0 <= kv_usage <= 1,
                    "invalid KV usage")
            if (self.batch - 1) % self.config.every:
                self.counts["frequency_skips"] += 1
                outcome = "interval_skip"
                return snapshot, ()
            for r in islice(waiting, self.config.scan_limit):
                self.counts["scanned_opportunities"] += 1
                if getattr(r, "r1_shadow_admitted", False):
                    self.counts["skip_already_admitted"] += 1
                    continue
                # PREEMPTED is also excluded if an unexpected legacy Request
                # lacks the admitted flag. WAITING_FOR_* is not yet eligible.
                if r.status.name != "WAITING":
                    self.counts["skip_status"] += 1
                    continue
                if not valid_lengths(getattr(r, "r1_shadow_lengths", None)):
                    self.counts["skip_no_static"] += 1
                    continue
                candidates.append(r)
                if len(candidates) == self.config.candidates:
                    self.counts["cap_reached_sweeps"] += 1
                    break
            if not candidates:
                self.counts["empty_sweeps"] += 1
                outcome = "empty"
                return snapshot, ()
            matrix = self.predictor.assemble(candidates, running_count, kv_usage)
            assembled = time.perf_counter_ns()
            self.counts["dynamic_attempts"] += 1
            predictions = self.predictor.dynamic(matrix)
            require(len(predictions) == len(candidates), "batch prediction length mismatch")
            for values in predictions:
                require(len(values) == 3 and all(math.isfinite(x) and x >= 0 for x in values)
                        and values == sorted(values), "invalid time quantiles")
            predicted = time.perf_counter_ns()
            records = [{"engine_request_id": r.request_id,
                        "prompt_tokens": r.num_prompt_tokens,
                        "max_tokens": r.max_tokens,
                        "length_quantiles": tuple(r.r1_shadow_lengths),
                        "time_quantiles": tuple(values)}
                       for r, values in zip(candidates, predictions)]
            for r in candidates:
                r.r1_shadow_predicted = True
            self.counts["dynamic_success"] += 1
            self.counts["candidate_predictions"] += len(candidates)
            outcome = "success"
            self.emit("shadow_candidates", {"snapshot": snapshot,
                "candidate_count": len(candidates), "candidates": records,
                "extract_assemble_ns": assembled - start,
                "state_read_ns": state_read - start,
                "time_head_ns": predicted - assembled,
                "shadow_compute_ns": time.perf_counter_ns() - start})
            if self.config.sort:
                self.sort_candidates(snapshot, candidates, predictions)
            return snapshot, tuple(r.request_id for r in candidates)
        except Exception as exc:
            outcome = "failed"
            self.counts["dynamic_errors"] += 1
            self.emit("shadow_dynamic_error", {"snapshot": snapshot,
                "candidate_count": len(candidates), "error_type": type(exc).__name__})
            return snapshot, ()
        finally:
            self.counts["dynamic_outcome_" + outcome] += 1
            # An interval skip does not inspect candidates: None means unknown,
            # rather than falsely declaring that the waiting cohort was empty.
            count = None if outcome == "interval_skip" else len(candidates)
            self.observe_elapsed("dynamic_foreground", start, outcome, count,
                                 total_counter="shadow_total_ns")

    def sort_candidates(self, snapshot, candidates, predictions):
        """Record a separate ordering; failures do not change prediction success."""
        start = time.perf_counter_ns()
        self.counts["sort_attempts"] += 1
        try:
            require(len(candidates) == len(predictions) and len(candidates) > 0,
                    "invalid sorting inputs")
            key_start = time.perf_counter_ns()
            keys = [values[1] for values in predictions]
            key_end = time.perf_counter_ns()
            indices = stable_sort_indices(keys)
            sort_end = time.perf_counter_ns()
            ids = [candidates[index].request_id for index in indices]
            ids_end = time.perf_counter_ns()
            record = {"decision_batch_id": snapshot["decision_batch_id"],
                      "rule": SORT_RULE, "timing_version": SORT_TIMING_VERSION,
                      "candidate_count": len(candidates),
                      "sorted_request_ids": ids,
                      "order_changed": indices != list(range(len(candidates))),
                      "key_construct_ns": key_end - key_start,
                      "stable_sort_ns": sort_end - key_end,
                      "id_construct_ns": ids_end - sort_end}
            record["sort_total_ns"] = time.perf_counter_ns() - start
        except Exception as exc:
            self.counts["sort_errors"] += 1
            self.emit("shadow_sort_error", {
                "decision_batch_id": snapshot["decision_batch_id"],
                "error_type": type(exc).__name__})
            return
        self.counts["sort_success"] += 1
        self.emit("shadow_sort", record)

    def admit(self, request, decision):
        start = time.perf_counter_ns()
        outcome = "failed"
        try:
            if getattr(request, "r1_shadow_lengths", None) is None:
                outcome = "no_static"
                return
            if getattr(request, "r1_shadow_admitted", False):
                outcome = "already_admitted"
                return
            request.r1_shadow_admitted = True
            self.counts["first_admissions"] += 1
            snapshot, ids = decision if decision is not None else (None, ())
            self.emit("shadow_first_admission", {"engine_request_id": request.request_id,
                "snapshot": snapshot, "predicted_in_this_batch": request.request_id in ids})
            outcome = "admitted"
        finally:
            self.observe_elapsed("first_admission", start, outcome)

    def finish(self, request):
        start = time.perf_counter_ns()
        outcome = "failed"
        try:
            self.counts["terminal_requests"] += 1
            self.emit("shadow_terminal", {"engine_request_id": request.request_id,
                "had_static": getattr(request, "r1_shadow_lengths", None) is not None,
                "was_candidate": getattr(request, "r1_shadow_predicted", False),
                "was_admitted": getattr(request, "r1_shadow_admitted", False),
                "status": request.status.name})
            outcome = "completed"
        finally:
            self.observe_elapsed("terminal", start, outcome)

    def close(self):
        if not self.closed:
            self.closed = True
            start = time.perf_counter_ns()
            values = {"counts": dict(self.counts),
                      "foreground_distributions": self.duration_report()}
            values["summary_build_ns"] = time.perf_counter_ns() - start
            self.emit("shadow_summary", values)
        complete = self.writer.close()
        return complete and not self.counts["log_errors"]


def create_shadow(vllm_config, role):
    if os.environ.get("VLLM_R1_ADMISSION_MODE", "off") != "off":
        from vllm.v1.core.sched.r1_admission import (AdmissionConfig,
                                                    unsupported_reason)
        AdmissionConfig.from_env()
        reason = unsupported_reason(vllm_config)
        if reason is not None:
            logger.warning("R1 admission and shadow disabled; native fallback: %s", reason)
            return None
    config = ShadowConfig.from_env()
    if config is None:
        return None
    start = time.perf_counter_ns()
    validate_service(vllm_config)
    predictor = Predictor(config.model, role)
    # Import/reuse only after enabled; collector switch / ID policy is irrelevant.
    from vllm.v1.metrics.request_collector import CollectionConfig, RecordWriter
    writer = RecordWriter(CollectionConfig(config.run_id, config.directory),
        "shadow-" + role, {"model_sha256": MODEL_SHA256,
            "model": vllm_config.model_config.model,
            "tokenizer": vllm_config.model_config.tokenizer,
            "model_revision": vllm_config.model_config.revision,
            "tokenizer_revision": vllm_config.model_config.tokenizer_revision,
            "dtype": str(vllm_config.model_config.dtype),
            "quantiles": QUANTILES, "max_tokens": 2048,
            "candidates": config.candidates, "scan_limit": config.scan_limit,
            "every": config.every, "threadpools": predictor.threadpools,
            "shadow_sort_enabled": config.sort, "shadow_sort_rule": SORT_RULE,
            "shadow_sort_timing_version": SORT_TIMING_VERSION,
            "environment": predictor.environment,
            "startup_load_ns": time.perf_counter_ns() - start})
    return Shadow(config, predictor, writer)
