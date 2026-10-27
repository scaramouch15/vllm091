"""Standalone selection experiment, unrelated to the vLLM scheduler."""
from collections import deque
import heapq
import math


def validate_scores(scores):
    """Called by the experiment before timing; never trust unvalidated input."""
    for score in scores:
        if (isinstance(score, bool) or not isinstance(score, (int, float))
                or not math.isfinite(score) or score < 0):
            raise ValueError("scores must be finite nonnegative numbers")


class SortedSelector:
    def __init__(self, entries):
        self.items = sorted(entries, reverse=True)

    def pop(self):
        return self.items.pop()


class ScanSelector:
    def __init__(self, entries):
        self.items = list(entries)

    def pop(self):
        index = min(range(len(self.items)), key=self.items.__getitem__)
        return self.items.pop(index)


class HeapSelector:
    def __init__(self, entries):
        self.items = list(entries)
        heapq.heapify(self.items)

    def pop(self):
        return heapq.heappop(self.items)


METHODS = {"sort": SortedSelector, "scan": ScanSelector, "heap": HeapSelector}


def pure_select(entries, attempts, selector):
    """Build a fresh structure and take exactly attempts entries.

    entries must contain validated (score, original_position, request_id)
    tuples with unique positions. Both preparation and validation are outside
    this timed function. Structure construction and the result list are timed.
    """
    data = selector(entries)
    return [data.pop() for _ in range(attempts)]


def maintain_window(waiting, scores, attempt_limit, capacity, fail_at, selector):
    """Mutate an independent deque; simulate stop on first failed allocation.

    `scores` are validated before timing. fail_at is a zero-based attempt
    index or None. Positions are unique, so the third tuple element is never
    used as a tie-breaker. Preserve the original order of all non-admitted
    prefix requests and leave the unexamined deque tail in place.
    """
    window = [waiting.popleft() for _ in scores]
    entries = [(score, index, request) for index, (score, request)
               in enumerate(zip(scores, window))]
    data = selector(entries)
    tried, admitted = [], []
    accepted = [False] * len(window)
    while len(tried) < attempt_limit and len(admitted) < capacity:
        entry = data.pop()
        tried.append(entry[2])
        if fail_at is not None and len(tried) - 1 == fail_at:
            break
        accepted[entry[1]] = True
        admitted.append(entry[2])
    waiting.extendleft(window[i] for i in range(len(window) - 1, -1, -1)
                       if not accepted[i])
    return tried, admitted


def fresh_waiting(n):
    return deque(range(n))
