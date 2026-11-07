"""Fairness apply integration tests using the real Scheduler test harness."""
from dataclasses import replace
import os
import unittest
from unittest.mock import patch
import test_r1_type_slo_admission as base

A = base.base.A
native = base.base.native

class FairnessApplyTests(unittest.TestCase):
    def setup_case(self, names='abcd'):
        h=base.TypeAdmissionTests()
        obj,pred,writer=h.make_type(names=names)
        obj.r1_admission.config=replace(obj.r1_admission.config, fairness_apply=True,
                                         fairness_wait_limit_seconds=30.)
        return h,obj,pred,writer

    def test_actual_order_equals_shared_fairness_function_and_restore(self):
        h,obj,pred,writer=self.setup_case();obj.max_num_running_reqs=2
        h.run100(obj); row=h.event(writer)
        branch, planned=A.observe_fairness_order(row['type_rank_candidates'],30.)
        self.assertEqual(row['fairness_apply_branch'],branch)
        self.assertEqual(row['sorted_ids'],planned)
        self.assertEqual(row['sorted_ids'],('b','a','c','d'))
        self.assertEqual(row['actual_admission_order'],['b','a'])
        self.assertEqual([r.request_id for r in obj.waiting],['c','d'])
        self.assertTrue(row['applied_order_changed'])
        self.assertTrue(row['actual_window_order_changed'])
        self.assertEqual((pred.calls,obj.r1_shadow.prefill_predictor.calls),(1,1))
        self.assertEqual(row['score_rule'],'type_slo_fairness_apply_v1')

    def test_fixed_protection_and_nonnegative_before_negative(self):
        h,obj,pred,writer=self.setup_case();obj.max_num_running_reqs=1
        obj.waiting[0].r1_first_queued_timestamp=70.
        h.run100(obj);row=h.event(writer)
        self.assertEqual(row['fairness_apply_branch'],'fairness')
        self.assertEqual(row['actual_admission_order'],['a'])
        h,obj,pred,writer=self.setup_case();obj.max_num_running_reqs=1
        obj.r1_type_slo=base.types.SimpleNamespace(snapshot=lambda:{'config_seconds':{
            'chat.ttft':2.,'chat.tpot':.1,'code.e2el':100.,'summary.e2el':100.}})
        h.run100(obj);row=h.event(writer)
        self.assertLess(row['type_rank_candidates'][1]['score'],0)
        self.assertEqual(row['fairness_apply_branch'],'feasible')
        self.assertNotEqual(row['actual_admission_order'],['b'])

    def test_all_negative_and_equal_scores_are_stable(self):
        for scores,expected in (({'a':120.,'b':200.,'c':150.,'d':180.},'all_infeasible_fcfs'),
                                ({'a':1.,'b':1.,'c':1.,'d':1.},'feasible')):
            h,obj,pred,writer=self.setup_case();obj.max_num_running_reqs=1
            for r in obj.waiting:r.r1_request_type='code'
            pred.scores=scores;h.run100(obj);row=h.event(writer)
            self.assertEqual(row['fairness_apply_branch'],expected)
            self.assertEqual(row['sorted_ids'],tuple('abcd'))
            self.assertEqual(row['actual_admission_order'],['a'])

    def test_missing_predictor_or_identity_falls_back(self):
        for condition in ('type','time','prefill','b','identity','bad_order'):
            h,obj,pred,writer=self.setup_case();obj.max_num_running_reqs=1
            if condition=='type':obj.waiting[0].r1_request_type=None
            if condition=='time':obj.waiting[0].r1_first_queued_timestamp=None
            if condition=='prefill':obj.r1_shadow.prefill_predictor.fail=True
            if condition=='b':pred.dynamic=lambda matrix: (_ for _ in ()).throw(ValueError('bad'))
            if condition=='identity':
                original=obj.r1_admission.make_decision
                def wrong(*args):
                    d=original(*args);return replace(d,candidate_ids=tuple(reversed(d.candidate_ids)))
                obj.r1_admission.make_decision=wrong
            if condition=='bad_order':
                target=patch.object(A,'observe_fairness_order',return_value=('feasible',('a','a','c','d')))
                target.start();self.addCleanup(target.stop)
            h.run100(obj);row=h.event(writer)
            self.assertNotEqual(row['reason'],'success',condition)
            self.assertEqual(row['actual_admission_order'],['a'],condition)
            self.assertEqual([r.request_id for r in obj.waiting],list('bcd'),condition)

    def test_kv_token_skipped_and_preemption(self):
        for failed,budget,admitted,remaining in (('b',20,[],list('abcd')),
                                                  ('a',20,['b'],list('acd')),
                                                  (None,3,['b'],list('acd'))):
            h,obj,pred,writer=self.setup_case();obj.max_num_scheduled_tokens=budget
            if failed:obj.kv_cache_manager.fail.add(failed)
            h.run100(obj)
            self.assertEqual([r.request_id for r in obj.running],admitted)
            self.assertEqual([r.request_id for r in obj.waiting],remaining)
        h,obj,pred,writer=self.setup_case()
        obj.waiting[2].status=native.N['RequestStatus'].WAITING_FOR_FSM
        obj.waiting[2].structured_output_request=None
        h.run100(obj)
        self.assertEqual(h.event(writer)['candidate_ids'],('a','b'))
        self.assertEqual([r.request_id for r in obj.waiting],['c'])
        h,obj,pred,writer=self.setup_case()
        running=base.base.req('running',native.N['RequestStatus'].RUNNING)
        obj.running.append(running);obj.requests['running']=running;obj.kv_cache_manager.fail.add('running')
        h.run100(obj)
        self.assertEqual(h.event(writer)['reason'],'preempted');self.assertEqual(pred.calls,0)
        self.assertEqual([r.request_id for r in obj.waiting],['running','a','b','c','d'])

    def test_configuration_is_explicit_and_conflicts_fail(self):
        with patch.dict(os.environ,{},clear=True):self.assertFalse(A.AdmissionConfig.from_env().fairness_apply)
        env={'VLLM_R1_ADMISSION_MODE':'apply','VLLM_R1_ADMISSION_HEAD_POLICY':'type_slo',
             'VLLM_R1_ADMISSION_TIME_OBSERVE':'1','VLLM_R1_FAIRNESS_APPLY':'1',
             'VLLM_R1_FAIRNESS_WAIT_LIMIT_SECONDS':'30'}
        with patch.dict(os.environ,env,clear=True):self.assertTrue(A.AdmissionConfig.from_env().fairness_apply)
        for change in ({'VLLM_R1_ADMISSION_MODE':'off'}, {'VLLM_R1_ADMISSION_MODE':'observe'},
                       {'VLLM_R1_ADMISSION_HEAD_POLICY':'ranked'}, {'VLLM_R1_ADMISSION_TIME_OBSERVE':'0'},
                       {'VLLM_R1_ADMISSION_FEASIBLE_PROTECTION':'1'}, {'VLLM_R1_FAIRNESS_COMPARE_OBSERVE':'1'},
                       {'VLLM_R1_FAIRNESS_WAIT_LIMIT_SECONDS':'nan'}, {'VLLM_R1_FAIRNESS_WAIT_LIMIT_SECONDS':'inf'},
                       {'VLLM_R1_FAIRNESS_WAIT_LIMIT_SECONDS':'-1'}, {'VLLM_R1_FAIRNESS_WAIT_LIMIT_SECONDS':'0'},
                       {'VLLM_R1_ADMISSION_SKIP_SINGLE_SLOT':'1'},
                       {'VLLM_R1_FAIRNESS_APPLY':'bad'}):
            with self.subTest(change=change),patch.dict(os.environ,dict(env,**change),clear=True):
                with self.assertRaises(ValueError):A.AdmissionConfig.from_env()
        with self.assertRaises(ValueError):
            replace(A.AdmissionConfig.from_env(), mode='apply', head_policy='type_slo', time_observation=True, fairness_apply=True, fairness_wait_limit_seconds=True)
        missing=dict(env);del missing['VLLM_R1_FAIRNESS_WAIT_LIMIT_SECONDS']
        with patch.dict(os.environ,missing,clear=True),self.assertRaises(ValueError):A.AdmissionConfig.from_env()


    def test_mismatched_candidate_rows_fall_back(self):
        h,obj,pred,writer=self.setup_case();obj.max_num_running_reqs=1
        original=obj.r1_shadow.observe_type_rank
        def mismatched(*args, **kwargs):
            rows,ordered=original(*args,**kwargs)
            return list(reversed(rows)),ordered
        obj.r1_shadow.observe_type_rank=mismatched
        h.run100(obj);row=h.event(writer)
        self.assertEqual(row['reason'],'type_prediction_or_decision_failed')
        self.assertEqual(row['actual_admission_order'],['a'])
        self.assertEqual([r.request_id for r in obj.waiting],list('bcd'))

    def test_scheduler_does_not_ignore_new_switch_in_off_mode(self):
        import sys
        import types
        ns=native.scheduler_namespace()
        ns.update(EventPublisherFactory=types.SimpleNamespace(create=lambda *args: None),
                  compute_encoder_budget=lambda **kwargs: (0,0),
                  EncoderCacheManager=lambda **kwargs: types.SimpleNamespace(),
                  KVCacheManager=lambda **kwargs: native.FakeKV())
        cfg=types.SimpleNamespace(
            scheduler_config=types.SimpleNamespace(max_num_seqs=8,max_num_batched_tokens=20,max_model_len=100,
                policy='fcfs',scheduler_cls='vllm.v1.core.sched.scheduler.Scheduler'),
            cache_config=types.SimpleNamespace(num_gpu_blocks=100,block_size=16,enable_prefix_caching=False,prefix_caching_hash_algo='builtin'),
            model_config=types.SimpleNamespace(is_multimodal_model=False,is_encoder_decoder=False,
                model='Meta-Llama-3-8B-Instruct',tokenizer='Meta-Llama-3-8B-Instruct',
                hf_config=types.SimpleNamespace(model_type='llama',hidden_size=4096,num_hidden_layers=32,vocab_size=128256),encoder_config=None),
            lora_config=None,kv_events_config=None,kv_transfer_config=None,speculative_config=None,
            parallel_config=types.SimpleNamespace(data_parallel_rank=0,data_parallel_size=1,pipeline_parallel_size=1,tensor_parallel_size=1))
        with patch.dict(sys.modules,{'vllm.v1.core.sched.r1_admission':A}):
            for env in ({'VLLM_R1_FAIRNESS_APPLY':'1','VLLM_R1_FAIRNESS_WAIT_LIMIT_SECONDS':'30'},
                        {'VLLM_R1_FAIRNESS_APPLY':'invalid'}):
                with self.subTest(env=env),patch.dict(os.environ,env,clear=True),self.assertRaises(ValueError):
                    ns['Scheduler'](cfg,types.SimpleNamespace(),types.SimpleNamespace())

if __name__ == '__main__':unittest.main()
