"""Actual Scheduler-method CPU integration for cached output-length SJF.

The scheduler and model use the existing native stubs; no GPU or server is used.
"""
import os
import unittest
from unittest.mock import Mock, patch
import test_r1_admission as base


class LengthSJFTests(unittest.TestCase):
    setup_scheduler = base.AdmissionTests.setup_scheduler
    event = base.AdmissionTests.event

    def setup_case(self, mode="apply", names="abcd", limit=32):
        obj, pred, writer = self.setup_scheduler(mode=mode, names=names,
                                                   head_policy="length_sjf", limit=limit)
        for req, score in zip(obj.waiting, (9., 3., 1., 1.)):
            req.r1_shadow_lengths = (score, score, score)
        pred.assemble = Mock(side_effect=AssertionError("dynamic assembly called"))
        pred.dynamic = Mock(side_effect=AssertionError("dynamic prediction called"))
        obj.r1_shadow.observe_prefill = Mock(side_effect=AssertionError("prefill called"))
        return obj, pred, writer

    def test_apply_stable_single_slot_and_snapshot(self):
        obj, pred, writer = self.setup_case()
        obj.max_num_running_reqs = 1
        obj.schedule()
        row = self.event(writer)
        self.assertEqual(row['candidate_ids'], tuple('abcd'))
        self.assertEqual(row['sorted_ids'], ('c', 'd', 'b', 'a'))
        self.assertEqual(row['length_p50_scores'], (9., 3., 1., 1.))
        self.assertEqual(row['score_rule'], 'predicted_output_length_p50_sjf_v1')
        self.assertEqual(row['scores_p90'], ())
        self.assertEqual(row['prediction_ns'], 0)
        self.assertEqual(row['prediction_successes'], 0)
        self.assertEqual(row['actual_admission_order'], ['c'])
        self.assertEqual([r.request_id for r in obj.waiting], ['a', 'b', 'd'])
        events = [r for k, r in writer.rows if k == 'shadow_first_admission']
        self.assertEqual(len(events), 1)
        self.assertFalse(events[0]['predicted_in_this_batch'])
        self.assertIsNotNone(events[0]['snapshot'])
        pred.assemble.assert_not_called()
        pred.dynamic.assert_not_called()
        obj.r1_shadow.observe_prefill.assert_not_called()

    def test_observe_does_not_reorder(self):
        obj, _, writer = self.setup_case(mode='observe')
        obj.max_num_running_reqs = 2
        obj.schedule()
        row = self.event(writer)
        self.assertEqual(row['sorted_ids'], ('c', 'd', 'b', 'a'))
        self.assertFalse(row['applied_order_changed'])
        self.assertEqual(row['actual_admission_order'], ['a', 'b'])
        self.assertEqual([r.request_id for r in obj.waiting], ['c', 'd'])

    def test_capacity_token_and_kv_restore(self):
        for failed, capacity, budget, admitted, waiting in (
            (None, 2, 20, ['c', 'd'], ['a', 'b']),
            (None, 8, 6, ['c', 'd'], ['a', 'b']),
            ('c', 8, 20, [], ['a', 'b', 'c', 'd']),
            ('d', 8, 20, ['c'], ['a', 'b', 'd'])):
            with self.subTest(failed=failed, budget=budget):
                obj, _, writer = self.setup_case()
                obj.max_num_running_reqs, obj.max_num_scheduled_tokens = capacity, budget
                if failed:
                    obj.kv_cache_manager.fail.add(failed)
                obj.schedule()
                self.assertEqual([r.request_id for r in obj.running], admitted)
                self.assertEqual([r.request_id for r in obj.waiting], waiting)
                self.assertEqual(sorted(admitted + waiting), list('abcd'))
                if failed:
                    self.assertEqual(self.event(writer)['exit_reason'], 'kv_allocation_failed')

    def test_empty_single_and_prefix_barriers(self):
        obj, _, writer = self.setup_case(names='')
        obj.schedule()
        self.assertEqual(self.event(writer)['reason'], 'empty_waiting')
        obj, _, writer = self.setup_case(names='a')
        obj.schedule()
        self.assertEqual(self.event(writer)['sorted_ids'], ('a',))
        for value in (None, (float('nan'), 1., 2.), (float('inf'), 1., 2.),
                      (-1., 1., 2.), (True, 1., 2.)):
            with self.subTest(value=value):
                obj, _, writer = self.setup_case()
                obj.waiting[1].r1_shadow_lengths = value
                obj.max_num_running_reqs = 1
                obj.schedule()
                row = self.event(writer)
                self.assertEqual(row['candidate_ids'], ('a',))
                self.assertEqual(row['prefix_stop'], 'ineligible')
                self.assertEqual(row['actual_admission_order'], ['a'])
        for attr, value in (('r1_shadow_admitted', True), ('num_computed_tokens', 1)):
            obj, _, writer = self.setup_case()
            setattr(obj.waiting[1], attr, value)
            obj.max_num_running_reqs = 1
            obj.schedule()
            self.assertEqual(self.event(writer)['candidate_ids'], ('a',))
        obj, _, writer = self.setup_case()
        obj.waiting[0].status = base.native.N['RequestStatus'].WAITING_FOR_FSM
        obj.waiting[0].structured_output_request = None
        obj.max_num_running_reqs = 1
        obj.schedule()
        self.assertEqual(self.event(writer)['reason'], 'empty_prefix')
        self.assertEqual([r.request_id for r in obj.running], ['b'])
        self.assertEqual([r.request_id for r in obj.waiting], ['a', 'c', 'd'])

    def test_invalid_decision_and_cache_failure_fallback(self):
        for kind in ('identity', 'exception'):
            obj, _, writer = self.setup_case()
            obj.max_num_running_reqs = 1
            if kind == 'identity':
                original = obj.r1_admission.make_length_decision
                def wrong(frame, candidates):
                    d = original(frame, candidates)
                    return base.A.Decision(d.instance_id, d.schedule_id, tuple(reversed(d.candidate_ids)), d.scores)
                obj.r1_admission.make_length_decision = wrong
            else:
                obj.r1_admission.make_length_decision = Mock(side_effect=ValueError('bad cache'))
            obj.schedule()
            self.assertEqual(self.event(writer)['reason'], 'invalid_decision' if kind == 'identity' else 'snapshot_or_cached_length_failed')
            self.assertEqual([r.request_id for r in obj.running], ['a'])

    def test_config_default_and_incompatible(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(base.A.AdmissionConfig.from_env().head_policy, 'protected')
        for extra in ({'VLLM_R1_ADMISSION_SKIP_SINGLE_SLOT': '1'},
                      {'VLLM_R1_ADMISSION_FEASIBLE_PROTECTION': '1'},
                      {'VLLM_R1_SLO_SHADOW': '1'}):
            env = {'VLLM_R1_ADMISSION_MODE': 'observe', 'VLLM_R1_ADMISSION_HEAD_POLICY': 'length_sjf', **extra}
            with patch.dict(os.environ, env, clear=True), self.assertRaises(ValueError):
                base.A.AdmissionConfig.from_env()

    def test_preempted_request_is_barrier_and_interval_is_preserved(self):
        obj, _, writer = self.setup_case()
        obj.waiting[1].status = base.native.N['RequestStatus'].PREEMPTED
        obj.max_num_running_reqs = 1
        obj.schedule()
        self.assertEqual(self.event(writer)['candidate_ids'], ('a',))
        self.assertEqual(self.event(writer)['prefix_stop_detail'], 'preempted')
        obj, _, writer = self.setup_case()
        obj.r1_admission = base.A.AdmissionController(
            base.A.AdmissionConfig('apply', 2, 32, head_policy='length_sjf'),
            obj.r1_shadow, base.S.valid_lengths)
        obj.kv_cache_manager.fail.update(['c', 'a'])
        obj.schedule()
        self.assertEqual(self.event(writer)['opportunity_id'], 1)
        obj.schedule()
        self.assertEqual(self.event(writer)['opportunity_id'], 2)
        self.assertEqual(self.event(writer)['reason'], 'interval_skip')

    def test_preempted_running_does_not_sort(self):
        obj, pred, writer = self.setup_case()
        running = base.req('running', base.native.N['RequestStatus'].RUNNING)
        obj.running.append(running)
        obj.requests['running'] = running
        obj.kv_cache_manager.fail.add('running')
        obj.schedule()
        self.assertEqual(self.event(writer)['reason'], 'preempted')
        self.assertEqual([r.request_id for r in obj.waiting], ['running', 'a', 'b', 'c', 'd'])
        pred.dynamic.assert_not_called()


if __name__ == '__main__':
    unittest.main()
