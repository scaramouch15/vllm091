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
        return cls(mode, every, limit, skip == "1")


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
        return {"schema_version": 1, "mode": self.config.mode,
                "every": self.config.every, "prefix_limit": self.config.prefix_limit,
                "skip_single_slot": self.config.skip_single_slot,
                "score_rule": "head_then_p90_ascending_stable_v1",
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

    def eligible(self, r):
        # Missing optional admitted flag is rejected, never defaulted to eligible.
        return (r.status.name == "WAITING" and self.valid_lengths(r.r1_shadow_lengths)
                and getattr(r, "r1_shadow_admitted", None) is False
                and r.num_output_tokens == 0 and r.num_computed_tokens == 0
                and not r.has_encoder_inputs and not r.mm_inputs and not r.mm_positions
                and r.lora_request is None and not r.use_structured_output
                and r.kv_transfer_params is None)

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
                frame["trigger_prefix_ns"] = time.perf_counter_ns() - start
                return None
            frame["reason"] = "single_slot_skip"
            frame["trigger_prefix_ns"] = time.perf_counter_ns() - start
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
            order = [0, *sorted(range(1, len(ids)), key=lambda i: (decision.scores[i], i))]
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
