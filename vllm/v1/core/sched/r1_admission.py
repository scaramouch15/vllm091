"""Opt-in RC1 waiting admission experiment; CPU prediction is provided by shadow."""
from dataclasses import dataclass
from itertools import islice
import math
import os
from pathlib import Path
import time
import uuid


@dataclass(frozen=True)
class AdmissionConfig:
    mode: str
    every: int
    prefix_limit: int
    skip_single_slot: bool = False
    head_policy: str = "protected"
    time_observation: bool = False
    wait_limit_seconds: float = None
    budget_seconds: float = None

    def __post_init__(self):
        if self.head_policy not in ("protected", "ranked", "conditional_wait", "conditional_risk"):
            raise ValueError("invalid VLLM_R1_ADMISSION_HEAD_POLICY")
        if self.mode in ("observe", "apply") and self.head_policy != "protected" and self.skip_single_slot:
            raise ValueError("ranked or conditional head policy conflicts with SKIP_SINGLE_SLOT=1")
        if self.mode in ("observe", "apply") and self.head_policy.startswith("conditional_"):
            if not self.time_observation:
                raise ValueError("conditional head policy requires TIME_OBSERVE=1")
            if (self.wait_limit_seconds is None or not math.isfinite(self.wait_limit_seconds)
                    or self.wait_limit_seconds <= 0):
                raise ValueError("conditional head policy requires finite WAIT_LIMIT_SECONDS > 0")
            if self.head_policy == "conditional_risk" and (
                    self.budget_seconds is None or not math.isfinite(self.budget_seconds)
                    or self.budget_seconds <= self.wait_limit_seconds):
                raise ValueError("conditional_risk requires finite BUDGET_SECONDS > WAIT_LIMIT_SECONDS > 0")

    @classmethod
    def from_env(cls):
        mode = os.environ.get("VLLM_R1_ADMISSION_MODE", "off")
        if mode not in ("off", "observe", "apply"):
            raise ValueError("VLLM_R1_ADMISSION_MODE must be off, observe or apply")
        try:
            every = int(os.environ.get("VLLM_R1_ADMISSION_EVERY", "8"))
            limit = int(os.environ.get("VLLM_R1_ADMISSION_PREFIX_LIMIT", "32"))
        except ValueError as exc:
            raise ValueError("admission EVERY and PREFIX_LIMIT must be integers") from exc
        if every <= 0 or not 1 <= limit <= 4096:
            raise ValueError("admission requires EVERY > 0 and 1 <= PREFIX_LIMIT <= 4096")
        skip = os.environ.get("VLLM_R1_ADMISSION_SKIP_SINGLE_SLOT", "0")
        if skip not in ("0", "1"):
            raise ValueError("VLLM_R1_ADMISSION_SKIP_SINGLE_SLOT must be 0 or 1")
        head_policy = os.environ.get("VLLM_R1_ADMISSION_HEAD_POLICY", "protected")
        timing = os.environ.get("VLLM_R1_ADMISSION_TIME_OBSERVE", "0")
        if timing not in ("0", "1"):
            raise ValueError("VLLM_R1_ADMISSION_TIME_OBSERVE must be 0 or 1")
        wait = budget = None
        if mode in ("observe", "apply") and head_policy in ("conditional_wait", "conditional_risk"):
            try:
                wait = float(os.environ["VLLM_R1_ADMISSION_WAIT_LIMIT_SECONDS"])
                if head_policy == "conditional_risk":
                    budget = float(os.environ["VLLM_R1_ADMISSION_BUDGET_SECONDS"])
            except (KeyError, ValueError) as exc:
                raise ValueError("conditional policy requires explicit numeric WAIT_LIMIT_SECONDS and, for risk, BUDGET_SECONDS") from exc
        return cls(mode, every, limit, skip == "1", head_policy, timing == "1", wait, budget)


def unsupported_reason(config):
    """Fail closed for configurations outside the reviewed RC1 envelope."""
    model = config.model_config
    parallel = config.parallel_config
    sched = config.scheduler_config
    if sched.policy != "fcfs" or sched.scheduler_cls != "vllm.v1.core.sched.scheduler.Scheduler":
        return "scheduler_not_native_fcfs"
    if model.is_multimodal_model or model.is_encoder_decoder:
        return "not_text_decoder"
    if (Path(model.model.rstrip("/")).name != "Meta-Llama-3-8B-Instruct" or
            Path(model.tokenizer.rstrip("/")).name != "Meta-Llama-3-8B-Instruct"):
        return "shadow_model_or_tokenizer"
    hf = model.hf_config
    if (hf.model_type != "llama" or hf.hidden_size != 4096 or
            hf.num_hidden_layers != 32 or hf.vocab_size != 128256):
        return "shadow_model_architecture"
    if (model.encoder_config or {}).get("do_lower_case", False):
        return "shadow_encoder_lowercase"
    if (parallel.data_parallel_size != 1 or parallel.pipeline_parallel_size != 1
            or parallel.tensor_parallel_size != 1):
        return "parallelism_not_one"
    if config.kv_transfer_config is not None:
        return "kv_connector"
    if config.speculative_config is not None:
        return "speculative_decoding"
    if config.lora_config is not None:
        return "lora_config"
    return None


@dataclass(frozen=True)
class Decision:
    instance_id: str
    schedule_id: int
    candidate_ids: tuple
    scores: tuple


class AdmissionController:
    def __init__(self, config, shadow, valid_lengths):
        self.config, self.shadow, self.valid_lengths = config, shadow, valid_lengths
        self.instance_id = uuid.uuid4().hex
        self.schedules = self.opportunities = self.prediction_successes = 0

    def begin(self):
        self.schedules += 1
        frame = {"schema_version": 1, "mode": self.config.mode,
                "every": self.config.every, "prefix_limit": self.config.prefix_limit,
                "skip_single_slot": self.config.skip_single_slot,
                "head_policy": self.config.head_policy,
                "score_rule": ("head_then_p90_ascending_stable_v1"
                               if self.config.head_policy == "protected" else
                               "all_p90_ascending_stable_v1"),
                "instance_id": self.instance_id, "schedule_id": self.schedules,
                "opportunity_id": None, "waiting_size": None, "checked": 0,
                "running_count": None, "kv_usage": None, "token_budget": None,
                "available_running_slots": None,
                "candidate_count": 0, "candidate_ids": (), "sorted_ids": (),
                "scores_p90": (), "prefix_stop": "not_scanned",
                "order_changed": False, "applied_order_changed": False,
                "window_attempts": [], "window_admitted": [],
                "tail_attempt_count": 0, "tail_admitted": [],
                "reason": None, "exit_reason": None,
                "trigger_prefix_ns": 0, "prediction_ns": 0,
                "decision_ns": 0, "apply_ns": 0, "restore_ns": 0,
                "_candidates": (), "_candidate_set": frozenset(), "_applied": False}
        if self.config.head_policy in ("conditional_wait", "conditional_risk"):
            frame.update(conditional_policy=self.config.head_policy, ranked_head_displaced=None,
                         ranked_head_position=None, wait_seconds=None,
                         wait_limit_seconds=self.config.wait_limit_seconds,
                         budget_seconds=self.config.budget_seconds, wait_due=None,
                         risk_due=None, selected_policy=None, condition_fallback=None,
                         condition_check_ns=0, head_adjust_ns=0, condition_timing_version=1)
        return frame

    def eligible(self, r):
        # Missing optional admitted flag is rejected, never defaulted to eligible.
        return (r.status.name == "WAITING" and self.valid_lengths(r.r1_shadow_lengths)
                and getattr(r, "r1_shadow_admitted", None) is False
                and r.num_output_tokens == 0 and r.num_computed_tokens == 0
                and not r.has_encoder_inputs and not r.mm_inputs and not r.mm_positions
                and r.lora_request is None and not r.use_structured_output
                and r.kv_transfer_params is None)

    def observe_schedule_time(self, frame, timestamp):
        start = time.perf_counter_ns()
        frame["scheduled_timestamp"] = timestamp
        frame["head_queued_timestamp"] = None
        frame["head_queued_status"] = "not_scanned"
        frame["time_observation_ns"] = time.perf_counter_ns() - start

    def observe_head_time(self, frame, request, log_stats):
        start = time.perf_counter_ns()
        value = getattr(request, "r1_first_queued_timestamp", None)
        frame["head_queued_timestamp"] = value
        frame["head_queued_status"] = ("available" if value is not None else
                                       "log_stats_disabled" if not log_stats else
                                       "queued_event_unavailable")
        elapsed = time.perf_counter_ns() - start
        frame["time_observation_ns"] += elapsed
        return elapsed

    def make_decision(self, frame, candidates, running_count, kv_usage):
        matrix = self.shadow.predictor.assemble(candidates, running_count, kv_usage)
        values = self.shadow.predictor.dynamic(matrix)
        if len(values) != len(candidates):
            raise ValueError("invalid prediction count")
        for row in values:
            if (len(row) != 3 or any(isinstance(x, bool) or not isinstance(x, (int, float))
                                     or not math.isfinite(x) or x < 0 for x in row)
                    or list(row) != sorted(row)):
                raise ValueError("invalid predicted quantiles")
        return Decision(self.instance_id, frame["schedule_id"],
                        tuple(r.request_id for r in candidates),
                        tuple(row[1] for row in values))

    def prepare(self, scheduler, frame, preempted, token_budget, timestamp):
        start = time.perf_counter_ns()
        frame["waiting_size"] = len(scheduler.waiting)
        frame["token_budget"] = token_budget
        frame["available_running_slots"] = (scheduler.max_num_running_reqs -
                                            len(scheduler.running))
        reason = ("preempted" if preempted else
                  "empty_waiting" if not scheduler.waiting else
                  "token_budget" if token_budget <= 0 else
                  "capacity" if len(scheduler.running) >= scheduler.max_num_running_reqs else None)
        if reason:
            frame["reason"] = reason
            frame["trigger_prefix_ns"] = time.perf_counter_ns() - start
            return None
        self.opportunities += 1
        frame["opportunity_id"] = self.opportunities
        if (self.opportunities - 1) % self.config.every:
            frame["reason"] = "interval_skip"
            frame["trigger_prefix_ns"] = time.perf_counter_ns() - start
            return None
        candidates = []
        for r in islice(scheduler.waiting, self.config.prefix_limit):
            frame["checked"] += 1
            if not self.eligible(r):
                frame["prefix_stop"] = "ineligible"
                break
            candidates.append(r)
        if frame["prefix_stop"] == "not_scanned":
            frame["prefix_stop"] = ("limit" if frame["checked"] == self.config.prefix_limit
                                    and frame["waiting_size"] > frame["checked"] else "end")
        frame["candidate_count"] = len(candidates)
        frame["trigger_prefix_ns"] = time.perf_counter_ns() - start
        head_observation_ns = 0
        if candidates and self.config.time_observation:
            head_observation_ns = self.observe_head_time(frame, candidates[0], scheduler.log_stats)
        elif self.config.time_observation:
            observation_start = time.perf_counter_ns()
            frame["head_queued_status"] = "no_eligible_head"
            frame["time_observation_ns"] += time.perf_counter_ns() - observation_start
        if not candidates:
            frame["reason"] = "empty_prefix"
            return None
        frame["_candidates"] = tuple(candidates)
        frame["candidate_ids"] = tuple(r.request_id for r in candidates)
        frame["_candidate_set"] = frozenset(frame["candidate_ids"])
        if self.config.skip_single_slot and frame["available_running_slots"] == 1:
            # Preserve the real pre-admission snapshot separately from the empty
            # set of IDs predicted in this batch; no model or ordering work.
            try:
                running_count = len(scheduler.running)
                kv_usage = float(scheduler.kv_cache_manager.usage)
                if not math.isfinite(kv_usage) or not 0 <= kv_usage <= 1:
                    raise ValueError("invalid KV usage")
                frame["running_count"], frame["kv_usage"] = running_count, kv_usage
                snapshot = {"decision_batch_id": f"{self.instance_id}:{frame['schedule_id']}",
                            "scheduled_timestamp": timestamp, "running_count": running_count,
                            "kv_usage": kv_usage, "token_budget": int(token_budget)}
            except Exception as exc:
                frame["reason"] = "snapshot_failed"
                frame["error_type"] = type(exc).__name__
                frame["trigger_prefix_ns"] = time.perf_counter_ns() - start - head_observation_ns
                return None
            frame["reason"] = "single_slot_skip"
            frame["trigger_prefix_ns"] = time.perf_counter_ns() - start - head_observation_ns
            return snapshot, ()
        started = time.perf_counter_ns()
        try:
            running_count, kv_usage = len(scheduler.running), float(scheduler.kv_cache_manager.usage)
            if not math.isfinite(kv_usage) or not 0 <= kv_usage <= 1:
                raise ValueError("invalid KV usage")
            frame["running_count"], frame["kv_usage"] = running_count, kv_usage
            decision = self.make_decision(frame, candidates, running_count, kv_usage)
        except Exception as exc:
            frame["prediction_ns"] = time.perf_counter_ns() - started
            frame["reason"] = "prediction_failed"
            frame["error_type"] = type(exc).__name__
            return None
        frame["prediction_ns"] = time.perf_counter_ns() - started
        self.prediction_successes += 1
        started = time.perf_counter_ns()
        try:
            actual = tuple(islice(scheduler.waiting, len(candidates)))
            ids = frame["candidate_ids"]
            if (not isinstance(decision, Decision) or decision.instance_id != self.instance_id
                    or decision.schedule_id != frame["schedule_id"]
                    or decision.candidate_ids != ids or len(set(ids)) != len(ids)
                    or len(decision.scores) != len(ids)
                    or any(isinstance(s, bool) or not isinstance(s, (float, int))
                           or not math.isfinite(s) or s < 0 for s in decision.scores)
                    or len(actual) != len(candidates)
                    or any(a is not b for a, b in zip(actual, candidates))):
                raise ValueError("decision identity or prefix mismatch")
            if self.config.head_policy == "protected":
                order = [0, *sorted(range(1, len(ids)), key=lambda i: (decision.scores[i], i))]
            else:
                order = sorted(range(len(ids)), key=lambda i: (decision.scores[i], i))
            ordered = tuple(candidates[i] for i in order)
            frame["sorted_ids"] = tuple(r.request_id for r in ordered)
            frame["scores_p90"] = decision.scores
            frame["order_changed"] = frame["sorted_ids"] != ids
        except Exception as exc:
            frame["decision_ns"] = time.perf_counter_ns() - started
            frame["reason"] = "invalid_decision"
            frame["error_type"] = type(exc).__name__
            return None
        frame["decision_ns"] = time.perf_counter_ns() - started
        if self.config.head_policy in ("conditional_wait", "conditional_risk"):
            check_start = time.perf_counter_ns()
            head_position = order.index(0)
            frame["ranked_head_position"] = head_position
            frame["ranked_head_displaced"] = head_position != 0
            selected = "ranked"
            if head_position:
                queued = frame.get("head_queued_timestamp")
                if queued is None:
                    frame["condition_fallback"] = "queued_timestamp_missing"
                elif (isinstance(queued, bool) or not isinstance(queued, (int, float))
                      or not math.isfinite(queued)):
                    frame["condition_fallback"] = "queued_timestamp_invalid"
                    frame["head_queued_timestamp"] = None
                    frame["head_queued_status"] = "queued_timestamp_invalid"
                elif queued > timestamp:
                    frame["condition_fallback"] = "queued_timestamp_after_schedule"
                else:
                    age = timestamp - queued
                    frame["wait_seconds"] = age
                    frame["wait_due"] = age >= self.config.wait_limit_seconds
                    if self.config.head_policy == "conditional_risk":
                        frame["risk_due"] = self.config.budget_seconds - age <= decision.scores[0]
                    if frame["wait_due"] or frame["risk_due"]:
                        selected = "protected"
                if frame["condition_fallback"] is not None:
                    selected = "protected"
            frame["selected_policy"] = selected
            frame["condition_check_ns"] = time.perf_counter_ns() - check_start
            if selected == "protected":
                adjust_start = time.perf_counter_ns()
                order.pop(head_position)
                order.insert(0, 0)
                ordered = tuple(candidates[i] for i in order)
                frame["sorted_ids"] = tuple(r.request_id for r in ordered)
                frame["order_changed"] = frame["sorted_ids"] != ids
                frame["head_adjust_ns"] = time.perf_counter_ns() - adjust_start
        for r in candidates:
            r.r1_shadow_predicted = True
        frame["reason"] = "success"
        if self.config.mode == "apply":
            started = time.perf_counter_ns()
            for _ in candidates:
                scheduler.waiting.popleft()
            scheduler.waiting.extendleft(reversed(ordered))
            frame["_applied"] = True
            frame["applied_order_changed"] = frame["order_changed"]
            frame["apply_ns"] = time.perf_counter_ns() - started
        snapshot = {"decision_batch_id": f"{self.instance_id}:{frame['schedule_id']}",
                    "scheduled_timestamp": timestamp, "running_count": running_count,
                    "kv_usage": kv_usage, "token_budget": int(token_budget)}
        return snapshot, ids

    def attempt(self, frame, request):
        if request.request_id in frame["_candidate_set"]:
            frame["window_attempts"].append(request.request_id)
        else:
            frame["tail_attempt_count"] += 1

    def admitted(self, frame, request):
        key = "window_admitted" if request.request_id in frame["_candidate_set"] else "tail_admitted"
        frame[key].append(request.request_id)

    def restore(self, scheduler, frame, skipped):
        if not frame["_applied"]:
            return
        started = time.perf_counter_ns()
        admitted = set(frame["window_admitted"])
        remaining = tuple(r for r in frame["_candidates"] if r.request_id not in admitted)
        actual = tuple(islice(scheduler.waiting, len(remaining)))
        if (remaining and skipped) or len(actual) != len(remaining) or \
                {id(r) for r in actual} != {id(r) for r in remaining}:
            raise RuntimeError("RC1 window restoration invariant violated")
        for _ in remaining:
            scheduler.waiting.popleft()
        scheduler.waiting.extendleft(reversed(remaining))
        frame["restore_ns"] = time.perf_counter_ns() - started

    def finish(self, frame, start):
        log_start = time.perf_counter_ns()
        frame["prediction_successes"] = self.prediction_successes
        frame["window_A"] = len(frame["window_attempts"])
        frame["window_M"] = len(frame["window_admitted"])
        frame["tail_A"] = frame["tail_attempt_count"]
        frame["tail_M"] = len(frame["tail_admitted"])
        frame["actual_admission_order"] = frame["window_admitted"] + frame["tail_admitted"]
        record = {key: value for key, value in frame.items() if not key.startswith("_")}
        self.shadow.emit("r1_admission_v1", record)
        self.shadow.observe_elapsed("admission_log_foreground_v1", log_start, "complete")
        self.shadow.observe_elapsed("admission_schedule_foreground_v1", start, "complete")
