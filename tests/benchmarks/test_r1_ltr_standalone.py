"""Low-cost checks for the standalone scorer; no vLLM or model loading."""
import importlib.util
import sys
import unittest
from pathlib import Path

p = Path(__file__).resolve().parents[2] / 'vllm/v1/metrics/r1_ltr_predictor.py'
spec = importlib.util.spec_from_file_location('r1_ltr_predictor_test', p)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class StandaloneScorerTests(unittest.TestCase):
    def test_import_does_not_load_model(self):
        self.assertEqual(module.MODEL_REVISION, '39df2b41ffe88d5ed967c6035d3838b5b5960379')
        self.assertEqual(len(module.TOKENIZER_SHA256), 5)

    def test_missing_model_is_rejected(self):
        with self.assertRaises(FileNotFoundError):
            module.PaperLTRScorer(p.parent / 'not-a-model', p.parent / 'not-a-tokenizer')

    def test_invalid_request_rejected_before_inference(self):
        scorer = module.PaperLTRScorer.__new__(module.PaperLTRScorer)
        for request_id, prompt in (('', 'text'), ('id', None)):
            with self.subTest(request_id=request_id), self.assertRaises(ValueError):
                scorer.score(request_id, prompt)


class AsyncScorerTests(unittest.IsolatedAsyncioTestCase):
    async def test_identity_bound_and_duplicate_id_rejected(self):
        import asyncio
        import threading
        started, unblock = threading.Event(), threading.Event()
        class Fake:
            def score(self, request_id, prompt):
                started.set()
                unblock.wait(2)
                return module.ScoreResult(request_id, float(len(prompt)), 1, 1)
        scorer = module.BoundedAsyncLTRScorer(Fake(), max_inflight=1)
        try:
            task = asyncio.create_task(scorer.score('one', 'abc', 1))
            await asyncio.to_thread(started.wait, 1)
            with self.assertRaisesRegex(module.ScoringUnavailable, 'duplicate'):
                await scorer.score('one', 'different', 1)
            unblock.set()
            result = await task
            self.assertEqual((result.request_id, result.score), ('one', 3))
        finally:
            unblock.set()
            await scorer.close()

    async def test_timeout_does_not_release_running_slot(self):
        import asyncio
        import threading
        started, unblock = threading.Event(), threading.Event()
        class Fake:
            def score(self, request_id, prompt):
                started.set()
                unblock.wait(2)
                return module.ScoreResult(request_id, 1., 1, 1)
        scorer = module.BoundedAsyncLTRScorer(Fake(), max_inflight=1)
        try:
            task = asyncio.create_task(scorer.score('one', 'text', 0.05))
            await asyncio.to_thread(started.wait, 1)
            with self.assertRaises(asyncio.TimeoutError):
                await task
            with self.assertRaisesRegex(module.ScoringUnavailable, 'queue_full'):
                await scorer.score('two', 'text', 0.05)
            unblock.set()
        finally:
            unblock.set()
            await scorer.close()

    async def test_wrong_identity_is_rejected(self):
        class Fake:
            def score(self, request_id, prompt):
                return module.ScoreResult('other', 1., 1, 1)
        scorer = module.BoundedAsyncLTRScorer(Fake())
        try:
            with self.assertRaisesRegex(module.ScoringUnavailable, 'identity'):
                await scorer.score('expected', 'text', 1)
        finally:
            await scorer.close()

class InternalScoreTests(unittest.TestCase):
    def test_typed_envelope_is_consumed_only_for_matching_request(self):
        from types import SimpleNamespace
        for value, expected in ((module.ScoreResult('one', -1.5, 1, 1), -1.5),
                                (module.ScoreResult('other', 2., 1, 1), None),
                                (module.ScoreResult('one', float('nan'), 1, 1), None),
                                ({'request_id': 'one', 'score': 3.}, None)):
            params = SimpleNamespace(extra_args={module.LTR_INTERNAL_KEY: value, 'keep': 4})
            self.assertEqual(module.consume_score(params, 'one'), expected)
            self.assertEqual(params.extra_args, {'keep': 4})

class AsyncCancellationTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancelled_running_request_holds_capacity_and_no_reuse(self):
        import asyncio
        import threading
        started, release = threading.Event(), threading.Event()
        class Fake:
            def score(self, request_id, prompt):
                started.set()
                release.wait(2)
                return module.ScoreResult(request_id, 3., 1, 1)
        worker = module.BoundedAsyncLTRScorer(Fake(), 1)
        try:
            task = asyncio.create_task(worker.score('cancelled', 'first', 1))
            await asyncio.to_thread(started.wait, 1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            with self.assertRaisesRegex(module.ScoringUnavailable, 'queue_full'):
                await worker.score('other', 'second', 1)
            with self.assertRaisesRegex(module.ScoringUnavailable, 'duplicate'):
                await worker.score('cancelled', 'replacement', 1)
        finally:
            release.set()
            await worker.close()
