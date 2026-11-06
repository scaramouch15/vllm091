"""Actual Scheduler admission path with controlled CPU predictor and KV stubs."""
from dataclasses import replace
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_r1_admission as base
import test_r1_type_rank_shadow as rank


class TypeAdmissionTests(base.AdmissionTests):
    def make_type(self, mode='apply', names='abcd'):
        obj, pred, writer = self.setup_scheduler(mode=mode, names=names)
        obj.r1_admission.config = replace(obj.r1_admission.config, head_policy='type_slo', time_observation=True)
        shadow = obj.r1_shadow
        shadow.config = replace(shadow.config, type_rank_shadow=True, type_rank_wait_limit=30.)
        shadow.prefill_predictor = rank.Prefill()
        obj.r1_type_slo = types.SimpleNamespace(snapshot=lambda: {'run_id':'unit', 'config_sha256':'digest',
            'config_seconds':{'chat.ttft':10.,'chat.tpot':.1,'code.e2el':100.,'summary.e2el':100.}})
        for r, kind in zip(obj.waiting, ('code', 'chat', 'summary', 'code')):
            r.r1_request_type=kind; r.r1_first_queued_timestamp=99.
        return obj, pred, writer

    def run100(self,obj):
        with patch.object(base.native.time,'monotonic',return_value=100.): obj.schedule()

    def test_apply_and_observe_and_one_batch(self):
        for mode, expected in (('apply',['b','a']),('observe',['a','b'])):
            obj,pred,writer=self.make_type(mode); obj.max_num_running_reqs=2
            self.run100(obj); row=self.event(writer)
            self.assertEqual(row['sorted_ids'],('b','a','c','d'))
            self.assertEqual(row['actual_admission_order'],expected)
            self.assertEqual([r.request_id for r in obj.waiting], [x for x in 'abcd' if x not in expected])
            self.assertEqual(pred.calls,1); self.assertEqual(obj.r1_shadow.prefill_predictor.calls,1)
            self.assertEqual([r['request_type'] for r in row['type_rank_candidates']],['code','chat','summary','code'])
            self.assertAlmostEqual(row['type_rank_candidates'][1]['score'],.7)
            self.assertAlmostEqual(row['type_rank_candidates'][0]['score'],.90)

    def test_missing_and_failure_return_native(self):
        for case in ('type','time','config','prefill','b'):
            obj,pred,writer=self.make_type();obj.max_num_running_reqs=1
            if case=='type': obj.waiting[0].r1_request_type=None
            if case=='time': obj.waiting[0].r1_first_queued_timestamp=None
            if case=='config': obj.r1_type_slo=None
            if case=='prefill': obj.r1_shadow.prefill_predictor=rank.Prefill(True)
            if case=='b': pred.dynamic=lambda rows: (_ for _ in ()).throw(ValueError('bad'))
            self.run100(obj); row=self.event(writer)
            self.assertNotEqual(row['reason'],'success'); self.assertEqual(row['actual_admission_order'],['a'])
            self.assertEqual([r.request_id for r in obj.waiting],list('bcd'))

    def test_protection_and_equal_stability(self):
        obj,pred,writer=self.make_type();obj.max_num_running_reqs=2
        obj.waiting[0].r1_first_queued_timestamp=70.
        self.run100(obj);row=self.event(writer)
        self.assertEqual(row['sorted_ids'][:2],('a','b'))
        self.assertEqual(row['type_rank_candidates'][0]['protection_reason'],'wait_limit')
        obj,pred,writer=self.make_type(names='ac')
        for r in obj.waiting: r.r1_request_type='code'
        pred.scores={'a':1,'c':1}; self.run100(obj)
        self.assertEqual(self.event(writer)['sorted_ids'],('a','c'))

    def test_negative_slack_and_invalid_identity(self):
        obj,pred,writer=self.make_type();obj.max_num_running_reqs=1
        obj.r1_type_slo=types.SimpleNamespace(snapshot=lambda:{'config_seconds':{
            'chat.ttft':1.,'chat.tpot':.1,'code.e2el':100.,'summary.e2el':100.}})
        self.run100(obj);row=self.event(writer)
        self.assertLess(row['type_rank_candidates'][1]['score'],0)
        self.assertEqual(row['actual_admission_order'],['b'])
        obj,pred,writer=self.make_type();obj.max_num_running_reqs=1
        original=obj.r1_admission.make_decision
        def wrong(*args):
            value=original(*args)
            return replace(value,candidate_ids=tuple(reversed(value.candidate_ids)))
        obj.r1_admission.make_decision=wrong
        self.run100(obj)
        self.assertEqual(self.event(writer)['reason'],'invalid_decision')
        self.assertEqual(self.event(writer)['actual_admission_order'],['a'])

    def test_kv_failure_and_token_restore(self):
        for failed, budget, admitted, remaining in (('b',20,[],list('abcd')),('a',20,['b'],list('acd')),(None,3,['b'],list('acd'))):
            obj,_,writer=self.make_type();obj.max_num_scheduled_tokens=budget
            if failed: obj.kv_cache_manager.fail.add(failed)
            self.run100(obj)
            self.assertEqual([r.request_id for r in obj.running],admitted)
            self.assertEqual([r.request_id for r in obj.waiting],remaining)
            self.assertEqual(sorted(admitted+remaining),list('abcd'))

    def test_barrier_single_and_interval(self):
        obj,pred,writer=self.make_type(names='a');self.run100(obj)
        self.assertEqual(self.event(writer)['sorted_ids'],('a',))
        obj,pred,writer=self.make_type();obj.waiting[0].status=base.native.N['RequestStatus'].WAITING_FOR_FSM
        obj.waiting[0].structured_output_request=None;obj.max_num_running_reqs=1
        self.run100(obj);self.assertEqual(self.event(writer)['reason'],'empty_prefix')
        self.assertEqual([r.request_id for r in obj.waiting],list('acd'))
        self.assertEqual(pred.calls,0)

    def test_skipped_tail_and_preemption(self):
        obj,pred,writer=self.make_type()
        obj.waiting[2].status=base.native.N['RequestStatus'].WAITING_FOR_FSM
        obj.waiting[2].structured_output_request=None
        self.run100(obj)
        self.assertEqual(self.event(writer)['candidate_ids'],('a','b'))
        self.assertEqual(self.event(writer)['actual_admission_order'],['b','a','d'])
        self.assertEqual([r.request_id for r in obj.waiting],['c'])
        obj,pred,writer=self.make_type()
        running=base.req('running',base.native.N['RequestStatus'].RUNNING)
        obj.running.append(running);obj.requests['running']=running
        obj.kv_cache_manager.fail.add('running');self.run100(obj)
        self.assertEqual(self.event(writer)['reason'],'preempted')
        self.assertEqual(pred.calls,0)
        self.assertEqual([r.request_id for r in obj.waiting],['running','a','b','c','d'])

    def test_config_mode_gates(self):
        env={'VLLM_R1_SHADOW':'1','VLLM_R1_SHADOW_MODEL':'x','VLLM_R1_SHADOW_RUN_ID':'x',
             'VLLM_R1_SHADOW_DIR':'x','VLLM_R1_TYPE_RANK_SHADOW':'1','VLLM_R1_TYPE_SLO':'1',
             'VLLM_R1_PREFILL_SHADOW':'1','VLLM_R1_PREFILL_MODEL':'x',
             'VLLM_R1_PREFILL_MODEL_SHA256':'a'*64,'VLLM_R1_PREFILL_PROVENANCE':'report',
             'VLLM_R1_ADMISSION_MODE':'apply','VLLM_R1_ADMISSION_HEAD_POLICY':'type_slo',
             'VLLM_R1_ADMISSION_TIME_OBSERVE':'1'}
        with patch.dict(os.environ,env,clear=True):
            self.assertEqual(base.A.AdmissionConfig.from_env().head_policy,'type_slo')
            self.assertTrue(base.S.ShadowConfig.from_env().type_rank_shadow)
        with patch.dict(os.environ,dict(env,VLLM_R1_ADMISSION_SKIP_SINGLE_SLOT='1'),clear=True), self.assertRaises(ValueError):
            base.A.AdmissionConfig.from_env()
        with patch.dict(os.environ,dict(env,VLLM_R1_ADMISSION_TIME_OBSERVE='0'),clear=True), self.assertRaises(ValueError):
            base.A.AdmissionConfig.from_env()

if __name__ == '__main__': unittest.main()
