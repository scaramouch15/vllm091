"""Independent state model, not an implementation of the vLLM Scheduler.

It models a bounded prefix, stop-on-failure and a simplified native skip
deque. It does not model real KV allocation, connector, encoder, or threading.
"""
from collections import deque
from dataclasses import dataclass
from itertools import islice
import math


@dataclass(frozen=True)
class Request:
    request_id: str
    eligible: bool = True
    # Simplified native effect when outside a candidate window.
    native_action: str = "admit"  # admit, skip, or kv_fail
    tokens: int = 1


@dataclass(frozen=True)
class Prediction:
    round_id: int
    scores: tuple


@dataclass(frozen=True)
class Outcome:
    waiting: tuple
    attempted: tuple
    admitted: tuple
    skipped: tuple
    selected_window: tuple
    decision: str
    stop: str
    token_left: int


def eligible_prefix(waiting, limit):
    """Inspect <= limit positions, ending immediately at first ineligible."""
    prefix = []
    for request in islice(waiting, limit):
        if not request.eligible:
            break
        prefix.append(request)
    return prefix


def run_round(waiting, *, round_id, limit, capacity, token_budget,
              prediction=None, prediction_failed=False, opportunity=True, preempted=False):
    """Model one synchronous round; caller appends/aborts only between rounds.

    `prediction` is artificial and must be for this round and this exact
    prefix length; it is never stored. capacity/tokens are abstract counts.
    `prediction_failed` represents a failure before any queue change.
    Any native_action=skip is assumed not ready throughout this round.
    """
    if limit < 0 or capacity < 0 or token_budget < 0:
        raise ValueError("negative limit, capacity or budget")
    work = deque(waiting)
    if len({r.request_id for r in work}) != len(work):
        raise ValueError("request IDs must be unique")
    if preempted:
        return Outcome(tuple(work), (), (), (), (), "preempted", "preempted", token_budget)
    candidates = eligible_prefix(work, limit) if opportunity and capacity and token_budget else []
    decision = "native"
    ordered = []
    if (not prediction_failed and candidates and isinstance(prediction, Prediction)
            and prediction.round_id == round_id):
        scores = prediction.scores
        valid = (len(scores) == len(candidates) and
                 all(not isinstance(s, bool) and isinstance(s, (int, float))
                     and math.isfinite(s) and s >= 0 for s in scores))
        if valid:
            # The first request remains first, even when it has a high score.
            tail = sorted(range(1, len(candidates)), key=lambda i: (scores[i], i))
            ordered = [candidates[i] for i in [0, *tail]]
            decision = "reordered"
    if ordered:
        for _ in candidates:
            work.popleft()
        work.extendleft(reversed(ordered))
    candidate_ids = {r.request_id for r in candidates} if ordered else set()
    tried, admitted, skipped = [], [], deque()
    stop = "empty"
    while work:
        if len(admitted) >= capacity:
            stop = "capacity"
            break
        if token_budget <= 0:
            stop = "tokens"
            break
        request = work[0]
        tried.append(request.request_id)
        if request.native_action == "skip":
            if request.request_id in candidate_ids:
                raise ValueError("a native-skip request cannot be candidate-eligible")
            skipped.appendleft(work.popleft())
            continue
        if request.native_action == "kv_fail":
            stop = "kv_failure"
            break
        if request.native_action != "admit" or request.tokens <= 0:
            raise ValueError("unsupported simplified native action")
        work.popleft()
        admitted.append(request.request_id)
        token_budget -= min(token_budget, request.tokens)
    if ordered:
        # Restore window survivors before the untouched tail; when a tail skip
        # occurred the whole window was consumed, so no survivors coexist.
        accepted = set(admitted)
        survivors = [r for r in candidates if r.request_id not in accepted]
        if survivors and skipped:
            raise AssertionError("tail cannot be visited before window ends")
        tail = [r for r in work if r.request_id not in candidate_ids]
        work = deque(survivors + tail)
    work.extendleft(skipped)  # Native appendleft + extendleft double reversal.
    return Outcome(tuple(work), tuple(tried), tuple(admitted),
                   tuple(r.request_id for r in reversed(skipped)),
                   tuple(r.request_id for r in candidates) if ordered else (),
                   decision, stop, token_budget)
