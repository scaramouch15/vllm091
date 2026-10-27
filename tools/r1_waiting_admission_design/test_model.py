"""CPU-only contract tests for the isolated state model."""
import unittest

from model import Prediction, Request, run_round


def ids(requests):
    return tuple(r.request_id for r in requests)


class WaitingModelTests(unittest.TestCase):
    def run_case(self, requests, scores=None, *, round_id=1, limit=10,
                 capacity=10, tokens=10, opportunity=True, preempted=False,
                 prediction=None, prediction_failed=False):
        if prediction is None and scores is not None:
            prediction = Prediction(round_id, tuple(scores))
        result = run_round(requests, round_id=round_id, limit=limit,
                           capacity=capacity, token_budget=tokens,
                           prediction=prediction, prediction_failed=prediction_failed,
                           opportunity=opportunity,
                           preempted=preempted)
        self.assertEqual(sorted(ids(requests)),
                         sorted((*result.admitted, *ids(result.waiting))))
        self.assertEqual(len(set((*result.admitted, *ids(result.waiting)))), len(requests))
        return result

    def test_sizes_zero_one_two_and_large_stable_head(self):
        for n in (0, 1, 2, 40):
            requests = [Request(str(i)) for i in range(n)]
            scores = [100] + [1] * max(0, n - 1) if n else []
            out = self.run_case(requests, scores, limit=n, capacity=n, tokens=n)
            self.assertEqual(out.admitted, tuple(str(i) for i in range(n)))
        requests = [Request(x) for x in "abcde"]
        out = self.run_case(requests, [9, 3, 1, 1, 2])
        self.assertEqual(out.attempted, ("a", "c", "d", "e", "b"))

    def test_no_reorder_skip_failed_invalid_stale(self):
        requests = [Request(x) for x in "abc"]
        for prediction, opportunity in ((None, True), (Prediction(1, (3, 1, 2)), False),
                (Prediction(0, (3, 1, 2)), True), (Prediction(1, (3, float("nan"), 2)), True),
                (Prediction(1, (3, float("inf"), 2)), True), (Prediction(1, (3,)), True)):
            with self.subTest(prediction=prediction, opportunity=opportunity):
                out = self.run_case(requests, prediction=prediction, opportunity=opportunity, capacity=2)
                self.assertEqual(out.decision, "native")
                self.assertEqual(out.admitted, ("a", "b"))
                self.assertEqual(ids(out.waiting), ("c",))
        preempted = self.run_case(requests, [3, 1, 2], preempted=True)
        self.assertEqual(preempted.stop, "preempted")
        self.assertEqual(ids(preempted.waiting), ("a", "b", "c"))
        failure = self.run_case(requests, prediction=Prediction(1, (9, 3, 1)),
                                prediction_failed=True, capacity=2)
        self.assertEqual(failure.decision, "native")
        self.assertEqual(failure.admitted, ("a", "b"))

    def test_capacity_token_and_kv_failure_restore_original(self):
        for capacity, tokens, expected in ((0, 10, ()), (10, 0, ()), (1, 10, ("a",)),
                                           (10, 2, ("a", "c")), (10, 10, ("a", "c", "b"))):
            out = self.run_case([Request(x) for x in "abc"], [9, 4, 1],
                                capacity=capacity, tokens=tokens)
            self.assertEqual(out.admitted, expected)
            self.assertEqual(ids(out.waiting), tuple(x for x in "abc" if x not in expected))
        head_failure = [Request("a", native_action="kv_fail"), Request("b"), Request("c")]
        out = self.run_case(head_failure, [9, 3, 1])
        self.assertEqual((out.attempted, out.admitted, out.stop), (("a",), (), "kv_failure"))
        self.assertEqual(ids(out.waiting), ("a", "b", "c"))
        middle = [Request("a"), Request("b"), Request("c", native_action="kv_fail"), Request("d")]
        out = self.run_case(middle, [9, 3, 1, 2])
        self.assertEqual((out.attempted, out.admitted, out.stop), (("a", "c"), ("a",), "kv_failure"))
        self.assertEqual(ids(out.waiting), ("b", "c", "d"))

    def test_barrier_cutoff_and_native_skip_reinsertion_after_window(self):
        requests = [Request("a"), Request("b"), Request("fsm", False, "skip"),
                    Request("remote", False, "skip"), Request("z")]
        out = self.run_case(requests, [2, 1], tokens=10)
        self.assertEqual(out.selected_window, ("a", "b"))
        self.assertEqual(out.attempted, ("a", "b", "fsm", "remote", "z"))
        self.assertEqual(out.admitted, ("a", "b", "z"))
        self.assertEqual(out.skipped, ("fsm", "remote"))
        self.assertEqual(ids(out.waiting), ("fsm", "remote"))
        first_special = [Request("fsm", False, "skip"), Request("a"), Request("b")]
        out = self.run_case(first_special, [1, 2])
        self.assertEqual(out.decision, "native")
        self.assertEqual(out.attempted, ("fsm", "a", "b"))
        self.assertEqual(ids(out.waiting), ("fsm",))
        partial = self.run_case(requests, [2, 1], capacity=1)
        self.assertEqual(ids(partial.waiting), ("b", "fsm", "remote", "z"))
        self.assertEqual(partial.skipped, ())
        tail_stop = self.run_case(requests, [2, 1], capacity=2)
        self.assertEqual(ids(tail_stop.waiting), ("fsm", "remote", "z"))
        self.assertEqual(tail_stop.skipped, ())

    def test_preempted_barrier_does_not_cross_and_native_resumes(self):
        requests = [Request("a"), Request("resume", False), Request("b")]
        out = self.run_case(requests, [10], capacity=3)
        self.assertEqual(out.selected_window, ("a",))
        self.assertEqual(out.admitted, ("a", "resume", "b"))
        out = self.run_case(requests[1:], [1, 0], capacity=2)
        self.assertEqual(out.decision, "native")
        self.assertEqual(out.admitted, ("resume", "b"))

    def test_multiround_head_protection_and_new_arrivals(self):
        waiting = tuple(Request(x) for x in "abcd")
        first = self.run_case(waiting, [9, 5, 1, 2], capacity=1)
        self.assertEqual(first.admitted, ("a",))
        waiting = first.waiting + (Request("new"),)
        second = self.run_case(waiting, [99, 8, 7, 0], round_id=2, capacity=1)
        self.assertEqual(second.admitted, ("b",))
        self.assertEqual(ids(second.waiting), ("c", "d", "new"))
        stale = self.run_case(second.waiting, round_id=3, capacity=1,
                              prediction=Prediction(2, (99, 1, 0)))
        self.assertEqual(stale.decision, "native")
        self.assertEqual(stale.admitted, ("c",))

    def test_unsupported_candidate_skip_rejected(self):
        with self.assertRaisesRegex(ValueError, "cannot be candidate"):
            self.run_case([Request("x", True, "skip")], [1])


if __name__ == "__main__":
    unittest.main()
