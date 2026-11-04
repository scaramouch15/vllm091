"""Reference order uses SLO observations, never the Scheduler queue."""
from dataclasses import replace
from unittest.mock import patch
import unittest
import test_r1_slo_shadow as base


class ReferenceOrderTests(base.SloShadowTests):
    def observe(self, obj, writer):
        obj.max_num_running_reqs = 1
        with patch.object(base.base.native.time, "monotonic", return_value=105.):
            obj.schedule()
        row = self.event(writer)
        return row, row["slo_shadow_v1"]

    def test_cross_type_stable_tie_and_negative_slack(self):
        obj, predictor, writer = self.setup_slo(names="bcd", scores={"b": 13, "c": 3, "d": 40})
        row, obs = self.observe(obj, writer)
        self.assertEqual([r["reference_slack_s"] for r in obs["candidates"]], [12., 12., -15.])
        self.assertEqual(obs["reference_order_status"], "complete")
        self.assertEqual(obs["reference_order_original_ids"], ["b", "c", "d"])
        self.assertEqual(obs["reference_order_comparable_ids"], ["b", "c", "d"])
        self.assertEqual(obs["reference_order_subset_ids"], ["d", "b", "c"])
        self.assertTrue(obs["reference_order_changed"])
        self.assertIsNone(obs["reference_comparison_reason"])
        self.assertEqual(row["actual_admission_order"], ["b"])
        self.assertEqual([r.request_id for r in obj.waiting], ["c", "d"])
        self.assertEqual(predictor.calls, 1)
        self.assertGreaterEqual(obs["reference_sort_ns"], 0)

    def test_partial_chat_and_missing_time_do_not_suggest_complete_order(self):
        obj, _, writer = self.setup_slo(names="abcd")
        obj.waiting[2].r1_first_queued_timestamp = None
        _, obs = self.observe(obj, writer)
        self.assertEqual(obs["reference_order_status"], "partial")
        self.assertEqual(obs["reference_order_original_ids"], list("abcd"))
        self.assertEqual(obs["reference_order_comparable_ids"], ["b", "d"])
        self.assertEqual(obs["reference_order_subset_ids"], ["b", "d"])
        self.assertIsNone(obs["reference_order_changed"])
        self.assertEqual(obs["reference_comparison_reason"], "partial_candidate_set")
        self.assertEqual({x["request_id"]: x["reason"] for x in obs["reference_order_unavailable"]},
                         {"a": "chat_e2e_reference_not_estimable", "c": "queued_time_unavailable"})

    def test_one_comparable_and_invalid_type(self):
        obj, _, writer = self.setup_slo(names="abc")
        obj.waiting[2].r1_request_type = None
        _, obs = self.observe(obj, writer)
        self.assertEqual(obs["reference_order_status"], "partial")
        self.assertEqual(obs["reference_order_subset_ids"], ["b"])
        self.assertEqual(obs["reference_comparison_reason"], "fewer_than_two_comparable")
        self.assertIsNone(obs["reference_order_changed"])
        self.assertEqual(obs["reference_order_unavailable"][1]["reason"], "type_missing_or_invalid")

    def test_missing_slo_and_failed_prediction_are_explicit(self):
        from types import SimpleNamespace
        obj, _, writer = self.setup_slo(names="bc")
        obj.r1_type_slo = SimpleNamespace(snapshot=lambda: {
            "run_id": "r", "config_sha256": "d", "config_seconds": {"summary.e2el": 20.}})
        _, obs = self.observe(obj, writer)
        self.assertEqual(obs["reference_order_status"], "partial")
        self.assertEqual(obs["reference_order_subset_ids"], ["c"])
        self.assertEqual(obs["reference_order_unavailable"][0]["reason"], "slo_missing")
        obj, predictor, writer = self.setup_slo(names="bc")
        predictor.dynamic = lambda _: [[float("nan"), 1., 2.]] * 2
        _, obs = self.observe(obj, writer)
        self.assertEqual(obs["reference_order_status"], "partial")
        self.assertEqual(obs["reference_order_subset_ids"], [])
        self.assertEqual(obs["reference_comparison_reason"], "fewer_than_two_comparable")
        self.assertTrue(all(x["reason"] == "prediction_prediction_failed"
                            for x in obs["reference_order_unavailable"]))

    def test_single_empty_and_unscanned_are_distinct(self):
        obj, _, writer = self.setup_slo(names="b")
        _, obs = self.observe(obj, writer)
        self.assertEqual(obs["reference_order_status"], "complete")
        self.assertEqual(obs["reference_comparison_reason"], "fewer_than_two_candidates")
        self.assertIsNone(obs["reference_order_changed"])
        obj, _, writer = self.setup_slo(names="a")
        obj.waiting[0].r1_shadow_lengths = None
        _, obs = self.observe(obj, writer)
        self.assertEqual(obs["reference_order_status"], "empty")
        self.assertEqual(obs["reference_order_original_ids"], [])
        self.assertEqual(obs["reference_comparison_reason"], "fewer_than_two_candidates")
        obj, _, writer = self.setup_slo(names="ab")
        obj.r1_admission.config = replace(obj.r1_admission.config, every=8)
        obj.r1_admission.opportunities = 1
        _, obs = self.observe(obj, writer)
        self.assertEqual(obs["reference_order_status"], "unscanned")
        self.assertIsNone(obs["reference_order_original_ids"])

    def test_disabled_no_reference_fields(self):
        obj, _, writer = self.setup_slo(enabled=False)
        obj.schedule()
        row = self.event(writer)
        self.assertNotIn("slo_shadow_v1", row)
        self.assertNotIn("reference_sort_ns", row)

if __name__ == "__main__":
    unittest.main()
