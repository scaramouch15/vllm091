"""CPU integration tests for native QUEUED time reuse."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import test_r1_admission as base


class TimeTests(base.AdmissionTests):
    def setup_time(self, policy="ranked", enabled=True, stats=True):
        obj, predictor, writer = self.setup_scheduler(names="", head_policy=policy)
        obj.r1_admission.config = replace(obj.r1_admission.config, time_observation=enabled)
        obj.log_stats = stats
        for name in "abc":
            obj.add_request(base.req(name))
        return obj, predictor, writer

    def test_native_queued_timestamp_survives_take_events_and_sort(self):
        with patch.object(base.native.time, "monotonic", return_value=100.0):
            obj, _, writer = self.setup_time()
        head = obj.waiting[0]
        self.assertEqual(head.r1_first_queued_timestamp, head.events[0].timestamp)
        self.assertEqual(head.events[0].type, base.native.N["EngineCoreEventType"].QUEUED)
        self.assertEqual(head.take_events()[0].timestamp, 100.0)
        self.assertEqual(head.events, [])
        obj.max_num_running_reqs = 1
        with patch.object(base.native.time, "monotonic", return_value=105.0):
            obj.schedule()
        row = self.event(writer)
        self.assertEqual((row["head_queued_timestamp"], row["scheduled_timestamp"],
                          row["head_queued_status"]), (100.0, 105.0, "available"))
        self.assertEqual([r.request_id for r in obj.waiting], ["a", "b"])
        self.assertEqual(head.r1_first_queued_timestamp, 100.0)

    def test_enabled_disabled_preserve_both_policies_and_missing(self):
        for policy in ("ranked", "protected"):
            outcomes = []
            for enabled in (False, True):
                obj, _, writer = self.setup_time(policy, enabled)
                obj.max_num_running_reqs = 1
                obj.schedule()
                row = self.event(writer)
                outcomes.append((row["sorted_ids"], row["actual_admission_order"],
                                 [r.request_id for r in obj.waiting]))
                self.assertEqual("scheduled_timestamp" in row, enabled)
            self.assertEqual(outcomes[0], outcomes[1])
        obj, _, writer = self.setup_time(stats=False)
        obj.schedule()
        self.assertEqual(self.event(writer)["head_queued_status"], "log_stats_disabled")
        self.assertIsNone(self.event(writer)["head_queued_timestamp"])
        obj, _, writer = self.setup_time()
        del obj.waiting[0].r1_first_queued_timestamp
        obj.schedule()
        self.assertEqual(self.event(writer)["head_queued_status"], "queued_event_unavailable")

    def test_first_admission_and_abort_keep_native_records(self):
        obj, _, writer = self.setup_time()
        obj.max_num_running_reqs = 1
        obj.schedule()
        decision = self.event(writer)
        admission = next(v for k, v in writer.rows if k == "shadow_first_admission")
        self.assertEqual(admission["snapshot"]["decision_batch_id"],
                         f"{decision['instance_id']}:{decision['schedule_id']}")
        admitted = decision["actual_admission_order"][0]
        obj.finish_requests(admitted, base.native.N["RequestStatus"].FINISHED_ABORTED)
        terminal = next(v for k, v in writer.rows if k == "shadow_terminal")
        self.assertEqual(terminal["status"], "FINISHED_ABORTED")

    def test_skipped_reinsert_keeps_queued_timestamp(self):
        obj, _, writer = self.setup_time()
        head = obj.waiting[0]
        stamp = head.r1_first_queued_timestamp
        head.status = base.native.N["RequestStatus"].WAITING_FOR_FSM
        head.structured_output_request = None
        obj.max_num_running_reqs = 1
        obj.schedule()
        self.assertIs(obj.waiting[0], head)
        self.assertEqual(head.r1_first_queued_timestamp, stamp)
        self.assertEqual(self.event(writer)["head_queued_status"], "no_eligible_head")

    def test_terminal_reasons_and_native_failure(self):
        status = base.native.N["RequestStatus"]
        for finish in (status.FINISHED_STOPPED, status.FINISHED_LENGTH_CAPPED, status.FINISHED_ABORTED):
            obj, _, writer = self.setup_time()
            obj.max_num_running_reqs = 1
            obj.schedule()
            request_id = self.event(writer)["actual_admission_order"][0]
            obj.finish_requests(request_id, finish)
            row = next(v for k, v in writer.rows if k == "shadow_terminal")
            self.assertEqual(row["status"], finish.name)
        obj, _, writer = self.setup_time()
        obj.kv_cache_manager.allocate_slots = base.Mock(side_effect=RuntimeError("native KV"))
        with self.assertRaisesRegex(RuntimeError, "native KV"):
            obj.schedule()
        self.assertFalse(any(k == "shadow_terminal" for k, _ in writer.rows))


if __name__ == "__main__":
    unittest.main()
