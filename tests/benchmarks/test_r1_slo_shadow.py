"""SLO shadow integration using Scheduler methods and controlled CPU stubs."""
from dataclasses import replace
import math
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import test_r1_admission as base


class SloShadowTests(base.AdmissionTests):
    def setup_slo(self, names="abcd", scores=None, enabled=True, stats=True):
        obj, predictor, writer = self.setup_scheduler(mode="observe", names="", scores=scores)
        obj.r1_admission.config = replace(obj.r1_admission.config, slo_observation=enabled)
        obj.log_stats = stats
        values = {"chat.ttft": 2., "chat.tpot": .03, "code.e2el": 30., "summary.e2el": 20.}
        obj.r1_type_slo = SimpleNamespace(snapshot=lambda: {"run_id": "r", "config_sha256": "digest", "config_seconds": values})
        with patch.object(base.native.time, "monotonic", return_value=100.):
            for name in names:
                r = base.req(name)
                r.r1_request_type = {"a": "chat", "b": "code", "c": "summary", "d": "code"}[name]
                obj.add_request(r)
        return obj, predictor, writer

    def test_slo_observation_and_native_order(self):
        obj, predictor, writer = self.setup_slo()
        obj.max_num_running_reqs = 1
        with patch.object(base.native.time, "monotonic", return_value=105.):
            obj.schedule()
        row = self.event(writer)
        obs = row["slo_shadow_v1"]
        self.assertEqual((obs["candidate_count"], obs["valid_prediction_count"]), (4, 4))
        self.assertEqual(obs["candidate_source"], "admission_contiguous_prefix")
        self.assertTrue(obs["same_type_comparable_pair"])
        self.assertTrue(obs["same_type_reference_differs"])
        a, b, c, d = obs["candidates"]
        self.assertEqual((a["request_type"], a["wait_s"], a["chat_ttft_slo_s"]), ("chat", 5., 2.))
        self.assertEqual(a["ttft_risk_status"], "not_estimable")
        self.assertEqual(a["tpot_risk_status"], "not_estimable")
        self.assertIsNone(a["reference_slack_s"])
        self.assertEqual((b["wait_plus_p90_s"], b["reference_slack_s"]), (8., 22.))
        self.assertEqual((c["wait_plus_p90_s"], c["reference_slack_s"]), (6., 14.))
        self.assertEqual(row["actual_admission_order"], ["a"])
        self.assertEqual([r.request_id for r in obj.waiting], list("bcd"))
        self.assertEqual(predictor.calls, 1)
        self.assertGreaterEqual(row["slo_construct_ns"], 0)

    def test_missing_time_invalid_prediction_single_and_barrier(self):
        for value, expected in ((None, "queued_time_unavailable"), (float("nan"), "queued_time_invalid"), (110., "queued_time_invalid")):
            obj, _, writer = self.setup_slo(names="a")
            obj.waiting[0].r1_first_queued_timestamp = value
            with patch.object(base.native.time, "monotonic", return_value=105.): obj.schedule()
            obs = self.event(writer)["slo_shadow_v1"]
            self.assertEqual(obs["candidate_count"], 1)
            self.assertFalse(obs["same_type_comparable_pair"])
            self.assertEqual(obs["candidates"][0]["wait_status"], expected)
        obj, predictor, writer = self.setup_slo(names="ab")
        predictor.dynamic = lambda _: [[float("nan"), 1., 2.]] * 2
        obj.schedule()
        obs = self.event(writer)["slo_shadow_v1"]
        self.assertEqual(obs["reason"], "prediction_failed")
        self.assertEqual(obs["valid_prediction_count"], 0)
        self.assertTrue(all(r["inference_p90_s"] is None for r in obs["candidates"]))
        obj, predictor, writer = self.setup_slo(names="abc")
        obj.waiting[1].max_tokens = 128
        obj.waiting[1].r1_shadow_lengths = None
        obj.schedule()
        obs = self.event(writer)["slo_shadow_v1"]
        self.assertEqual(obs["candidate_count"], 1)
        self.assertEqual(obs["first_ineligible"]["reason"], "unsupported_max_tokens")
        self.assertEqual(obs["window_outside_checked"], 1)

    def test_disabled_stats_and_interval_skip(self):
        obj, _, writer = self.setup_slo(names="a", stats=False)
        obj.schedule()
        self.assertEqual(self.event(writer)["slo_shadow_v1"]["candidates"][0]["wait_status"], "native_stats_disabled")
        obj, _, writer = self.setup_slo(names="a", enabled=False)
        obj.schedule()
        self.assertNotIn("slo_shadow_v1", self.event(writer))
        obj, _, writer = self.setup_slo(names="ab")
        obj.r1_admission.config = replace(obj.r1_admission.config, every=8)
        obj.r1_admission.opportunities = 1
        obj.schedule()
        obs = self.event(writer)["slo_shadow_v1"]
        self.assertEqual(obs["reason"], "interval_skip")
        self.assertIsNone(obs["candidate_count"])
        self.assertIsNone(obs["same_type_reference_differs"])

    def test_preempted_admitted_barriers_and_writer_loss(self):
        status = base.native.N["RequestStatus"]
        for request_status, admitted, reason in ((status.PREEMPTED, True, "preempted"),
                                                  (status.WAITING, True, "already_admitted")):
            obj, predictor, writer = self.setup_slo(names="ab")
            obj.waiting[0].status = request_status
            obj.waiting[0].r1_shadow_admitted = admitted
            obj.max_num_running_reqs = 1
            obj.schedule()
            obs = self.event(writer)["slo_shadow_v1"]
            self.assertEqual(obs["first_ineligible"]["reason"], reason)
            self.assertEqual(predictor.calls, 0)
        obj, _, writer = self.setup_slo(names="a")
        writer.emit = lambda *args: False
        obj.schedule()
        self.assertGreaterEqual(obj.r1_shadow.counts["log_dropped"], 1)

    def test_enabled_and_disabled_observe_identical_and_isolated_error(self):
        outcomes = []
        for enabled in (False, True):
            obj, predictor, writer = self.setup_slo(enabled=enabled)
            obj.max_num_running_reqs = 1
            with patch.object(base.native.time, "monotonic", return_value=105.):
                obj.schedule()
            row = self.event(writer)
            outcomes.append((row["actual_admission_order"], [r.request_id for r in obj.waiting], predictor.calls))
        self.assertEqual(outcomes[0], outcomes[1])
        obj, predictor, writer = self.setup_slo(names="ab")
        obj.r1_admission.build_slo_observation = lambda frame: (_ for _ in ()).throw(ValueError("bad"))
        obj.schedule()
        self.assertEqual(self.event(writer)["slo_shadow_v1"]["reason"], "observation_error")
        self.assertEqual(predictor.calls, 1)
        self.assertEqual(self.event(writer)["actual_admission_order"], ["a", "b"])
        obj, _, writer = self.setup_slo(names="a")
        obj.r1_type_slo = SimpleNamespace(snapshot=lambda: (_ for _ in ()).throw(ValueError("bad")))
        obj.schedule()
        obs = self.event(writer)["slo_shadow_v1"]
        self.assertEqual(obs["config_error_type"], "ValueError")
        self.assertEqual(obs["candidates"][0]["missing_reason"], "config_unavailable")
        self.assertEqual(self.event(writer)["actual_admission_order"], ["a"])

    def test_config_requires_observe_and_valid_switch(self):
        with patch.dict(os.environ, {"VLLM_R1_SLO_SHADOW": "1"}, clear=True):
            with self.assertRaisesRegex(ValueError, "observe"): base.A.AdmissionConfig.from_env()
        with patch.dict(os.environ, {"VLLM_R1_SLO_SHADOW": "bad", "VLLM_R1_ADMISSION_MODE": "observe"}, clear=True):
            with self.assertRaisesRegex(ValueError, "SLO_SHADOW"): base.A.AdmissionConfig.from_env()
        with patch.dict(os.environ, {"VLLM_R1_SLO_SHADOW": "1", "VLLM_R1_ADMISSION_MODE": "observe"}, clear=True):
            self.assertTrue(base.A.AdmissionConfig.from_env().slo_observation)

if __name__ == "__main__":
    unittest.main()
