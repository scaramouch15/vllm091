"""CPU tests using actual Scheduler methods and controllable stubs."""
from dataclasses import replace
import builtins
import os
import sys
import types
import unittest
from unittest.mock import Mock, patch

import test_r1_admission as base


class ConditionalTests(base.AdmissionTests):
    def configured(self, policy="conditional_wait", mode="apply", wait=5., budget=10., stamp=95.):
        obj, pred, writer = self.setup_scheduler(mode=mode)
        obj.r1_admission.config = replace(obj.r1_admission.config, head_policy=policy,
            time_observation=True, wait_limit_seconds=wait,
            budget_seconds=budget if policy == "conditional_risk" else None)
        for request in obj.waiting:
            request.r1_first_queued_timestamp = stamp
        obj.max_num_running_reqs = 1
        return obj, pred, writer

    def run_at_100(self, obj):
        with patch.object(base.native.time, "monotonic", return_value=100.):
            return obj.schedule()

    def test_wait_threshold_and_unneeded(self):
        for stamp, expected in ((95., "protected"), (96., "ranked")):
            obj, pred, writer = self.configured(stamp=stamp)
            self.run_at_100(obj)
            row = self.event(writer)
            self.assertEqual(row["selected_policy"], expected)
            self.assertEqual(row["wait_due"], stamp == 95.)
            self.assertIsNone(row["risk_due"])
            self.assertEqual(row["actual_admission_order"], ["a" if expected == "protected" else "c"])
            self.assertEqual(pred.calls, 1)
        obj, pred, writer = self.configured()
        pred.scores["a"] = 0
        self.run_at_100(obj)
        row = self.event(writer)
        self.assertFalse(row["ranked_head_displaced"])
        self.assertEqual(row["ranked_head_position"], 0)
        self.assertIsNone(row["wait_due"])
        self.assertIsNone(row["wait_seconds"])

    def test_risk_both_equal_and_over_budget(self):
        for stamp, wait_due, risk_due in ((95., True, True), (96., False, True), (85., True, True)):
            obj, pred, writer = self.configured(policy="conditional_risk", stamp=stamp)
            pred.scores["a"] = 6.
            self.run_at_100(obj)
            row = self.event(writer)
            self.assertEqual((row["wait_due"], row["risk_due"]), (wait_due, risk_due))
            self.assertEqual(row["selected_policy"], "protected")
            self.assertEqual(row["actual_admission_order"], ["a"])

    def test_missing_invalid_and_future_queued(self):
        for value, reason in ((None, "queued_timestamp_missing"),
                              (float("nan"), "queued_timestamp_invalid"),
                              (float("inf"), "queued_timestamp_invalid"),
                              (101., "queued_timestamp_after_schedule")):
            obj, _, writer = self.configured(stamp=value)
            self.run_at_100(obj)
            row = self.event(writer)
            self.assertEqual(row["condition_fallback"], reason)
            self.assertEqual(row["selected_policy"], "protected")
            self.assertIsNone(row["wait_due"])
            self.assertEqual(row["actual_admission_order"], ["a"])

    def test_one_sort_observe_and_restore(self):
        for mode in ("observe", "apply"):
            obj, pred, writer = self.configured(mode=mode)
            obj.max_num_running_reqs = 2
            calls = []
            def counted(values, *args, **kwargs):
                if isinstance(values, range): calls.append(1)
                return builtins.sorted(values, *args, **kwargs)
            with patch.object(base.A, "sorted", side_effect=counted, create=True):
                self.run_at_100(obj)
            row = self.event(writer)
            self.assertEqual((pred.calls, len(calls)), (1, 1))
            self.assertEqual(row["sorted_ids"], ("a", "c", "d", "b"))
            self.assertEqual(row["actual_admission_order"], ["a", "c"] if mode == "apply" else ["a", "b"])
            self.assertEqual([r.request_id for r in obj.waiting], ["b", "d"] if mode == "apply" else ["c", "d"])
            self.assertEqual(row["condition_timing_version"], 2)

    def test_risk_all_boolean_combinations_and_no_unneeded_search(self):
        class NoIndexList(list):
            def index(self, *args):
                raise AssertionError("unexpected head search")
        # Synthetic thresholds and scores only, not recommended experiment parameters.
        for wait_due, risk_due, stamp, score in (
                (True, False, 95., 1.), (False, False, 99., 1.),
                (False, True, 96., 6.), (True, True, 95., 6.)):
            for mode in ("observe", "apply"):
                with self.subTest(wait_due=wait_due, risk_due=risk_due, mode=mode):
                    obj, pred, writer = self.configured(policy="conditional_risk", mode=mode, stamp=stamp)
                    pred.scores = {"a": score, "b": .1, "c": .2, "d": .3}
                    searches_forbidden = not (wait_due or risk_due)
                    calls = []
                    def counted(values, *args, **kwargs):
                        result = builtins.sorted(values, *args, **kwargs)
                        if isinstance(values, range):
                            calls.append(1)
                            if searches_forbidden:
                                return NoIndexList(result)
                        return result
                    with patch.object(base.A, "sorted", side_effect=counted, create=True):
                        self.run_at_100(obj)
                    row = self.event(writer)
                    protected = wait_due or risk_due
                    self.assertEqual((row["wait_due"], row["risk_due"]), (wait_due, risk_due))
                    self.assertEqual(row["selected_policy"], "protected" if protected else "ranked")
                    self.assertTrue(row["ranked_head_displaced"])
                    self.assertEqual(row["ranked_head_position"], 3 if protected else None)
                    self.assertEqual(row["sorted_ids"][0], "a" if protected else "b")
                    self.assertEqual(row["actual_admission_order"], ["a" if mode == "observe" or protected else "b"])
                    self.assertEqual((pred.calls, len(calls)), (1, 1))
                    self.assertEqual(row["condition_timing_version"], 2)

    def test_failure_and_frequency_native_fallback(self):
        obj, pred, writer = self.configured()
        obj.r1_admission.config = replace(obj.r1_admission.config, every=2)
        pred.dynamic = Mock(side_effect=ValueError("failure"))
        obj.kv_cache_manager.fail.add("a")
        self.run_at_100(obj)
        self.assertEqual(self.event(writer)["reason"], "prediction_failed")
        self.run_at_100(obj)
        row = self.event(writer)
        self.assertEqual(row["reason"], "interval_skip")
        self.assertIsNone(row["selected_policy"])
        pred.dynamic.assert_called_once()
        self.assertEqual([r.request_id for r in obj.waiting], list("abcd"))

    def test_conditional_native_resource_and_special_boundaries(self):
        for fail, budget, expected, remaining in (("a", 20, [], list("abcd")),
                                                 ("c", 20, ["a"], ["b", "c", "d"]),
                                                 (None, 6, ["a", "c"], ["b", "d"])):
            obj, _, writer = self.configured()
            obj.max_num_running_reqs = 8
            obj.max_num_scheduled_tokens = budget
            if fail: obj.kv_cache_manager.fail.add(fail)
            self.run_at_100(obj)
            self.assertEqual(self.event(writer)["actual_admission_order"], expected)
            self.assertEqual([r.request_id for r in obj.waiting], remaining)
        obj, pred, writer = self.configured()
        obj.waiting[2].status = base.native.N["RequestStatus"].WAITING_FOR_FSM
        obj.waiting[2].structured_output_request = None
        obj.max_num_running_reqs = 8
        self.run_at_100(obj)
        self.assertEqual(self.event(writer)["candidate_ids"], ("a", "b"))
        self.assertEqual([r.request_id for r in obj.waiting], ["c"])
        self.assertEqual(pred.calls, 1)
        obj, pred, writer = self.configured()
        running = base.req("running", base.native.N["RequestStatus"].RUNNING)
        obj.running.append(running); obj.requests["running"] = running
        obj.max_num_running_reqs = 8
        obj.kv_cache_manager.fail.add("running")
        self.run_at_100(obj)
        self.assertEqual(self.event(writer)["reason"], "preempted")
        self.assertEqual(pred.calls, 0)

    def test_scheduler_rejects_conditional_without_native_stats(self):
        ns = base.native.scheduler_namespace()
        ns.update(EventPublisherFactory=types.SimpleNamespace(create=lambda *args: None),
                  compute_encoder_budget=lambda **kwargs: (0, 0),
                  EncoderCacheManager=lambda **kwargs: types.SimpleNamespace(),
                  KVCacheManager=lambda **kwargs: base.native.FakeKV())
        cfg = types.SimpleNamespace(
            scheduler_config=types.SimpleNamespace(max_num_seqs=8, max_num_batched_tokens=20, max_model_len=100),
            cache_config=types.SimpleNamespace(num_gpu_blocks=100, block_size=16,
                enable_prefix_caching=False, prefix_caching_hash_algo="builtin"),
            model_config=types.SimpleNamespace(), lora_config=None, kv_events_config=None,
            kv_transfer_config=None, speculative_config=None,
            parallel_config=types.SimpleNamespace(data_parallel_rank=0))
        env = {"VLLM_R1_ADMISSION_MODE": "apply", "VLLM_R1_ADMISSION_HEAD_POLICY": "conditional_wait",
               "VLLM_R1_ADMISSION_TIME_OBSERVE": "1", "VLLM_R1_ADMISSION_WAIT_LIMIT_SECONDS": "5"}
        with patch.dict(sys.modules, {"vllm.v1.core.sched.r1_admission": base.A}), \
             patch.dict(os.environ, env, clear=True), self.assertRaisesRegex(ValueError, "log_stats"):
            ns["Scheduler"](cfg, types.SimpleNamespace(), types.SimpleNamespace(), log_stats=False)

    def test_config_requirements_and_default(self):
        A = base.A
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(A.AdmissionConfig.from_env().head_policy, "protected")
        with patch.dict(os.environ, {"VLLM_R1_ADMISSION_MODE": "off",
                                    "VLLM_R1_ADMISSION_HEAD_POLICY": "conditional_risk"}, clear=True):
            self.assertIsNone(A.AdmissionConfig.from_env().budget_seconds)
        env = {"VLLM_R1_ADMISSION_MODE": "apply", "VLLM_R1_ADMISSION_HEAD_POLICY": "conditional_risk",
               "VLLM_R1_ADMISSION_TIME_OBSERVE": "1", "VLLM_R1_ADMISSION_WAIT_LIMIT_SECONDS": "5",
               "VLLM_R1_ADMISSION_BUDGET_SECONDS": "10"}
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(A.AdmissionConfig.from_env().budget_seconds, 10.)
        for changes in ({"VLLM_R1_ADMISSION_WAIT_LIMIT_SECONDS": "0"},
                        {"VLLM_R1_ADMISSION_WAIT_LIMIT_SECONDS": "nan"},
                        {"VLLM_R1_ADMISSION_BUDGET_SECONDS": "5"},
                        {"VLLM_R1_ADMISSION_BUDGET_SECONDS": "inf"},
                        {"VLLM_R1_ADMISSION_TIME_OBSERVE": "0"},
                        {"VLLM_R1_ADMISSION_SKIP_SINGLE_SLOT": "1"}):
            with patch.dict(os.environ, dict(env, **changes), clear=True), self.assertRaises(ValueError):
                A.AdmissionConfig.from_env()
        for key in ("VLLM_R1_ADMISSION_WAIT_LIMIT_SECONDS", "VLLM_R1_ADMISSION_BUDGET_SECONDS"):
            incomplete = dict(env); del incomplete[key]
            with patch.dict(os.environ, incomplete, clear=True), self.assertRaises(ValueError):
                A.AdmissionConfig.from_env()


if __name__ == "__main__":
    unittest.main()
