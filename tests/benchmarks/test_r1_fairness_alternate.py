"""CPU tests for optional fairness/feasibility alternation."""
from dataclasses import replace
import os
import unittest
from unittest.mock import patch
import test_r1_fairness_apply as base

A = base.A

class AlternateTests(unittest.TestCase):
    def setup_case(self):
        h = base.FairnessApplyTests()
        harness, obj, pred, writer = h.setup_case()
        obj.r1_admission.config = replace(obj.r1_admission.config, fairness_alternate=True)
        obj.waiting[0].r1_first_queued_timestamp = 70.
        return harness, obj, writer

    def test_interleaves_and_preserves_orders(self):
        def r(id, p, w, score): return dict(request_id=id, position=p, wait_s=w, score=score)
        rows = [r('p1',0,30,-2),r('f1',1,1,.4),r('p2',2,31,-3),r('f2',3,1,.2),r('i',4,1,-1)]
        self.assertEqual(A.alternate_fairness_order(rows,30,'feasible'),('alternate',('f2','p1','f1','p2','i'),True))
        self.assertEqual(A.alternate_fairness_order(rows,30,'protected'),('alternate',('p1','f2','p2','f1','i'),True))
        self.assertEqual(A.alternate_fairness_order(rows[:1],30,'feasible')[1],('p1',))

    def test_feasible_order_uses_absolute_slack_without_changing_groups(self):
        rows = [dict(request_id='code', position=0, wait_s=15.95, score=.139,
                     slo_s=30., prediction_p90_s=9.89),
                dict(request_id='chat', position=1, wait_s=1.389, score=.254,
                     slo_s=2., prediction_p90_s=.103),
                dict(request_id='protected', position=2, wait_s=31., score=-1.)]
        self.assertEqual(A.alternate_fairness_order(rows, 30., 'feasible')[1][0], 'code')
        self.assertEqual(A.alternate_fairness_order(rows, 30., 'feasible', 'absolute')[1],
                         ('chat', 'protected', 'code'))
        self.assertEqual(A.alternate_fairness_order(rows, 30., 'feasible', 'critical', .6)[1],
                         ('chat', 'protected', 'code'))
        self.assertEqual(A.alternate_fairness_order(rows[:2], 30., 'feasible', 'absolute')[1],
                         ('chat', 'code'))

    def test_one_slot_switches_only_after_actual_admission(self):
        h,obj,w = self.setup_case(); obj.max_num_running_reqs=1
        h.run100(obj); e=h.event(w)
        self.assertEqual(e['actual_admission_order'],['b'])
        self.assertEqual(e['fairness_alternate_turn_before'],'feasible')
        self.assertEqual(e['fairness_alternate_turn_after'],'protected')
        # A fresh scheduler shares only the controller turn, showing the next conflict.
        h2,obj2,w2 = self.setup_case(); obj2.max_num_running_reqs=1
        obj2.r1_admission.fairness_next=obj.r1_admission.fairness_next
        h2.run100(obj2); e2=h2.event(w2)
        self.assertEqual(e2['actual_admission_order'],['a'])
        self.assertEqual(e2['fairness_alternate_turn_after'],'feasible')

    def test_two_slots_interleave_and_no_admission_does_not_advance(self):
        h,obj,w=self.setup_case();obj.max_num_running_reqs=2
        h.run100(obj);e=h.event(w)
        self.assertEqual(e['actual_admission_order'],['b','a'])
        self.assertEqual(e['fairness_alternate_turn_after'],'feasible')
        h,obj,w=self.setup_case();obj.max_num_running_reqs=1;obj.kv_cache_manager.fail.add('b')
        h.run100(obj);e=h.event(w)
        self.assertEqual(e['actual_admission_order'],[])
        self.assertEqual(e['fairness_alternate_turn_after'],'feasible')

    def test_bad_inputs_fall_back_without_advancing(self):
        h,obj,w=self.setup_case();obj.max_num_running_reqs=1;obj.waiting[0].r1_request_type=None
        h.run100(obj);e=h.event(w)
        self.assertNotEqual(e['reason'],'success')
        self.assertEqual(e['fairness_alternate_turn_after'],'feasible')

    def test_switch_is_opt_in_and_requires_fairness_apply(self):
        with patch.dict(os.environ,{},clear=True): self.assertFalse(A.AdmissionConfig.from_env().fairness_alternate)
        for value in ('1','bad'):
            with patch.dict(os.environ,{'VLLM_R1_FAIRNESS_ALTERNATE':value},clear=True):
                with self.assertRaises(ValueError): A.AdmissionConfig.from_env()

    def test_middle_kv_failure_updates_only_for_success(self):
        h,obj,w=self.setup_case();obj.max_num_running_reqs=2
        obj.kv_cache_manager.fail.add('a')
        h.run100(obj);e=h.event(w)
        self.assertEqual(e['window_attempts'], ['b','a'])
        self.assertEqual(e['window_admitted'], ['b'])
        self.assertEqual(e['fairness_alternate_turn_after'], 'protected')
        self.assertEqual([r.request_id for r in obj.waiting], list('acd'))

    def test_skipped_tail_and_preemption_do_not_add_turns(self):
        h,obj,w=self.setup_case()
        obj.waiting[2].status=base.native.N['RequestStatus'].WAITING_FOR_FSM
        obj.waiting[2].structured_output_request=None
        h.run100(obj);e=h.event(w)
        self.assertEqual(e['candidate_ids'], ('a','b'))
        self.assertEqual(e['window_admitted'], ['b','a'])
        self.assertEqual(e['tail_admitted'], ['d'])
        self.assertEqual(e['fairness_alternate_turn_after'], 'feasible')
        self.assertEqual([r.request_id for r in obj.waiting], ['c'])
        h,obj,w=self.setup_case()
        running=base.base.base.req('running',base.native.N['RequestStatus'].RUNNING)
        obj.running.append(running);obj.requests['running']=running
        obj.kv_cache_manager.fail.add('running')
        h.run100(obj);e=h.event(w)
        self.assertEqual(e['reason'],'preempted')
        self.assertEqual(e['fairness_alternate_turn_after'],'feasible')
        self.assertEqual([r.request_id for r in obj.waiting],['running','a','b','c','d'])

    def test_no_competing_group_does_not_advance_turn(self):
        h,obj,w=self.setup_case();obj.max_num_running_reqs=1
        obj.waiting[0].r1_first_queued_timestamp=99.
        h.run100(obj);e=h.event(w)
        self.assertFalse(e['fairness_alternate_competing'])
        self.assertEqual(len(e['window_admitted']),1)
        self.assertEqual(e['fairness_alternate_turn_after'],'feasible')

    def test_native_exception_after_admission_advances_turn(self):
        h,obj,w=self.setup_case();obj.max_num_running_reqs=2
        original=obj.kv_cache_manager.allocate_slots
        calls=[]
        def fail_second(*args,**kwargs):
            calls.append(1)
            if len(calls)==2:
                raise RuntimeError('native KV exception')
            return original(*args,**kwargs)
        obj.kv_cache_manager.allocate_slots=fail_second
        with self.assertRaisesRegex(RuntimeError,'native KV exception'):
            h.run100(obj)
        self.assertEqual([r.request_id for r in obj.running],['b'])
        self.assertEqual(obj.r1_admission.fairness_next,'protected')
        self.assertFalse(any(kind=='r1_admission_v1' for kind,_ in w.rows))

    def test_duplicate_admission_notification_does_not_advance_twice(self):
        h,obj,w=self.setup_case()
        controller=obj.r1_admission
        frame=controller.begin()
        frame['_candidate_set']=frozenset(('a','b'))
        frame['_applied']=True
        frame['fairness_alternate_competing']=True
        frame['_fairness_groups']={'a':'protected','b':'feasible'}
        a,b=obj.waiting[0],obj.waiting[1]
        controller.admitted(frame,b)
        self.assertEqual(controller.fairness_next,'protected')
        controller.admitted(frame,a)
        self.assertEqual(controller.fairness_next,'feasible')
        controller.admitted(frame,b)
        self.assertEqual(controller.fairness_next,'feasible')

if __name__=='__main__': unittest.main()
