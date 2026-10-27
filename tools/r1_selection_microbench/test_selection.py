"""Behavioral tests for the isolated selection experiment."""
import random
import unittest

from selection import METHODS, fresh_waiting, maintain_window, pure_select, validate_scores


class SelectionTest(unittest.TestCase):
    def test_validation(self):
        validate_scores([])
        validate_scores([0, 1.5, 200])
        for score in (float("nan"), float("inf"), -float("inf"), -1, True, "1"):
            with self.subTest(score=score), self.assertRaises(ValueError):
                validate_scores([score])

    def check_case(self, scores, n, limit, capacity, failure):
        validate_scores(scores)
        k = len(scores)
        entries = [(score, pos, pos) for pos, score in enumerate(scores)]
        expected_order = sorted(range(k), key=lambda pos: (scores[pos], pos))
        expected_tried = []
        expected_admitted = []
        for position in expected_order:
            if len(expected_tried) >= limit or len(expected_admitted) >= capacity:
                break
            expected_tried.append(position)
            if failure is not None and len(expected_tried) - 1 == failure:
                break
            expected_admitted.append(position)
        expected_waiting = [x for x in range(n) if x not in expected_admitted]
        for name, method in METHODS.items():
            with self.subTest(method=name, scores=scores, limit=limit,
                              capacity=capacity, failure=failure):
                original = list(entries)
                self.assertEqual([x[2] for x in pure_select(entries, limit, method)],
                                 expected_order[:limit])
                self.assertEqual(entries, original)
                waiting = fresh_waiting(n)
                tried, admitted = maintain_window(waiting, scores, limit, capacity,
                                                  failure, method)
                self.assertEqual(tried, expected_tried)
                self.assertEqual(admitted, expected_admitted)
                self.assertEqual(list(waiting), expected_waiting)
                self.assertEqual(list(waiting)[- (n-k):], list(range(k, n)) if n > k else list(waiting))
                self.assertEqual(sorted(admitted + list(waiting)), list(range(n)))
                self.assertFalse(set(admitted) & set(waiting))

    def test_distributions_and_stopping(self):
        rng = random.Random(20261027)
        distributions = [[], [3], [2] * 32, [i // 4 for i in range(32)],
                         list(range(32)), list(range(32, 0, -1)),
                         [rng.random() for _ in range(32)]]
        for scores in distributions:
            k = len(scores)
            cases = [(0, 0, None), (k, 0, None), (k, k, None),
                     (k, k // 2, None), (min(4, k), k, None)]
            if k:
                cases += [(k, k, 0), (k, k, k // 2)]
            for limit, capacity, failure in cases:
                self.check_case(scores, k + 7, limit, capacity, failure)

    def test_full_window_no_tail_and_first_failure(self):
        self.check_case([3, 1, 1, 2], 4, 4, 4, 0)


if __name__ == "__main__":
    unittest.main()
