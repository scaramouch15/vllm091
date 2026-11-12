"""Scheduler-method tests; controlled resources and logger, no real EngineCore."""
import unittest
import os
from unittest.mock import Mock
from unittest.mock import patch
import test_r1_admission as base

class LTRAdmissionTests(unittest.TestCase):
    setup_scheduler = base.AdmissionTests.setup_scheduler
    event = base.AdmissionTests.event

    def case(self, mode='apply'):
        obj, pred, writer = self.setup_scheduler(mode=mode, head_policy='paper_ltr')
        obj.r1_shadow = None
        class Sink:
            def emit(self, kind, values):
                writer.emit(kind, values)
            def observe_elapsed(self, *args):
                pass
        obj.r1_admission.shadow = Sink()
        for req, score in zip(obj.waiting, (-1., 3., 3., 0.)):
            req.r1_shadow_lengths = None
            req.r1_ltr_score = score
            req.r1_ltr_admitted = False
        pred.dynamic = Mock(side_effect=AssertionError('B dynamic called'))
        return obj, pred, writer

    def test_apply_and_observe_stable_without_b(self):
        for mode, actual in [('apply', ['b', 'c']), ('observe', ['a', 'b'])]:
            with self.subTest(mode=mode):
                obj, pred, writer = self.case(mode)
                obj.max_num_running_reqs = 2
                obj.schedule()
                row = self.event(writer)
                self.assertEqual(row['sorted_ids'], ('b', 'c', 'd', 'a'))
                self.assertEqual(row['actual_admission_order'], actual)
                self.assertEqual([r.request_id for r in obj.running], actual)
                self.assertEqual([r.request_id for r in obj.waiting], [x for x in 'abcd' if x not in actual])
                pred.dynamic.assert_not_called()

    def test_configuration_conflicts(self):
        env = {'VLLM_R1_ADMISSION_MODE': 'apply',
               'VLLM_R1_ADMISSION_HEAD_POLICY': 'paper_ltr'}
        for extra in ({'VLLM_R1_ADMISSION_SKIP_SINGLE_SLOT': '1'},
                      {'VLLM_R1_ADMISSION_FEASIBLE_PROTECTION': '1'},
                      {'VLLM_R1_FAIRNESS_APPLY': '1'}):
            with patch.dict(os.environ, {**env, **extra}, clear=True):
                with self.assertRaises(ValueError):
                    base.A.AdmissionConfig.from_env()

    def test_single_candidate_and_running_preemption(self):
        obj, pred, writer = self.case()
        while len(obj.waiting) > 1:
            obj.waiting.pop()
        obj.schedule()
        self.assertEqual(self.event(writer)['sorted_ids'], ('a',))
        pred.dynamic.assert_not_called()
        obj, pred, writer = self.case()
        running = base.req('running', base.native.N['RequestStatus'].RUNNING)
        obj.running.append(running)
        obj.requests['running'] = running
        obj.kv_cache_manager.fail.add('running')
        obj.schedule()
        self.assertEqual(self.event(writer)['reason'], 'preempted')
        self.assertEqual([r.request_id for r in obj.waiting], ['running', 'a', 'b', 'c', 'd'])
        pred.dynamic.assert_not_called()

    def test_invalid_prefix_and_kv_restore(self):
        for value in (None, float('nan'), float('inf')):
            obj, pred, writer = self.case()
            obj.waiting[1].r1_ltr_score = value
            obj.max_num_running_reqs = 1
            obj.schedule()
            self.assertEqual(self.event(writer)['candidate_ids'], ('a',))
            self.assertEqual([r.request_id for r in obj.running], ['a'])
            pred.dynamic.assert_not_called()
        obj, _, writer = self.case()
        obj.kv_cache_manager.fail.add('c')
        obj.schedule()
        self.assertEqual([r.request_id for r in obj.running], ['b'])
        self.assertEqual([r.request_id for r in obj.waiting], ['a', 'c', 'd'])
        self.assertEqual(self.event(writer)['exit_reason'], 'kv_allocation_failed')

    def test_preempted_is_never_ranked(self):
        obj, pred, writer = self.case()
        obj.waiting[1].status = base.native.N['RequestStatus'].PREEMPTED
        obj.max_num_running_reqs = 1
        obj.schedule()
        self.assertEqual(self.event(writer)['candidate_ids'], ('a',))
        pred.dynamic.assert_not_called()

    def test_duplicate_identity_falls_back_before_queue_application(self):
        obj, _, writer = self.case()
        obj.waiting[1].request_id = 'a'
        obj.max_num_running_reqs = 1
        obj.schedule()
        self.assertEqual(self.event(writer)['reason'], 'invalid_decision')
        self.assertFalse(self.event(writer)['applied_order_changed'])

    def test_token_budget_and_skipped_barrier(self):
        obj, _, writer = self.case()
        obj.max_num_scheduled_tokens = 3
        obj.schedule()
        self.assertEqual(self.event(writer)['window_admitted'], ['b'])
        self.assertEqual([r.request_id for r in obj.waiting], ['a', 'c', 'd'])

    def test_changed_prefix_falls_back_before_application(self):
        from collections import deque
        obj, _, writer = self.case()
        class ChangedPrefix(deque):
            calls = 0
            def __iter__(self):
                self.calls += 1
                if self.calls == 2:
                    first = self.popleft()
                    second = self.popleft()
                    self.appendleft(first)
                    self.appendleft(second)
                return super().__iter__()
        obj.waiting = ChangedPrefix(obj.waiting)
        obj.max_num_running_reqs = 1
        obj.schedule()
        self.assertEqual(self.event(writer)['reason'], 'invalid_decision')
        self.assertFalse(self.event(writer)['applied_order_changed'])
        self.assertEqual([r.request_id for r in obj.running], ['b'])
        obj, _, writer = self.case()
        obj.waiting[0].status = base.native.N['RequestStatus'].WAITING_FOR_FSM
        obj.waiting[0].structured_output_request = None
        obj.max_num_running_reqs = 1
        obj.schedule()
        self.assertEqual(self.event(writer)['reason'], 'empty_prefix')
        self.assertEqual([r.request_id for r in obj.running], ['b'])
        self.assertEqual([r.request_id for r in obj.waiting], ['a', 'c', 'd'])
