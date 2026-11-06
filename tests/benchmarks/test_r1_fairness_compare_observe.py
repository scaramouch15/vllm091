"""Same-snapshot comparison must not alter the observe-mode waiting queue."""
from dataclasses import replace
import os
import unittest
from unittest.mock import patch

import test_r1_feasible_protection as existing

A = existing.base.base.A


class FairnessCompareTests(unittest.TestCase):
    def make(self):
        harness = existing.FeasibleProtectionTests()
        obj, predictor, writer = harness.configured()
        obj.r1_admission.config = replace(obj.r1_admission.config, mode='observe',
            fairness_compare_observe=True, fairness_wait_limit_seconds=30.)
        obj.waiting[1].r1_first_queued_timestamp = 89.
        return harness, obj, predictor, writer

    def test_independent_order_and_no_actual_reorder(self):
        h, obj, predictor, writer = self.make()
        h.run100(obj)
        row = h.event(writer)
        self.assertEqual(row['sorted_ids'][0], 'b')
        self.assertEqual(row['fairness_compare_order_ids'][0], 'a')
        self.assertTrue(row['fairness_compare_vs_plan_changed'])
        self.assertEqual(row['fairness_compare_branch'], 'feasible')
        self.assertEqual(row['actual_admission_order'], ['a'])
        self.assertEqual([r.request_id for r in obj.waiting], list('bcd'))
        self.assertEqual(predictor.calls, 1)
        self.assertEqual(obj.r1_shadow.prefill_predictor.calls, 1)

    def test_compare_error_cannot_change_plan_or_admission(self):
        h, obj, predictor, writer = self.make()
        with patch.object(A, 'observe_fairness_order', side_effect=RuntimeError('test')):
            h.run100(obj)
        row = h.event(writer)
        self.assertEqual(row['fairness_compare_status'], 'error')
        self.assertEqual(row['sorted_ids'][0], 'b')
        self.assertEqual(row['actual_admission_order'], ['a'])

    def test_fairness_and_all_infeasible_stability(self):
        rows = [{'request_id':'a','position':0,'wait_s':31.,'score':-1.},
                {'request_id':'b','position':1,'wait_s':32.,'score':-2.},
                {'request_id':'c','position':2,'wait_s':1.,'score':0.1}]
        self.assertEqual(A.observe_fairness_order(rows,30.), ('fairness', ('a','b','c')))
        rows[0]['wait_s'] = rows[1]['wait_s'] = 1.
        rows[2]['score'] = -1.
        self.assertEqual(A.observe_fairness_order(rows,30.), ('all_infeasible_fcfs', ('a','b','c')))

    def test_invalid_or_apply_config_rejected(self):
        for val in ('0', '-1', 'nan', 'inf', 'invalid'):
            with patch.dict(os.environ, {'VLLM_R1_ADMISSION_MODE':'observe',
                'VLLM_R1_ADMISSION_HEAD_POLICY':'type_slo',
                'VLLM_R1_ADMISSION_TIME_OBSERVE':'1',
                'VLLM_R1_ADMISSION_FEASIBLE_PROTECTION':'1',
                'VLLM_R1_FAIRNESS_COMPARE_OBSERVE':'1',
                'VLLM_R1_FAIRNESS_WAIT_LIMIT_SECONDS':val}, clear=True):
                with self.assertRaises(ValueError): A.AdmissionConfig.from_env()
        h, obj, predictor, writer = self.make()
        with self.assertRaises(ValueError):
            replace(obj.r1_admission.config, mode='apply')

    def test_missing_data_keeps_native_order_and_not_evaluated(self):
        h, obj, predictor, writer = self.make()
        obj.waiting[0].r1_request_type = None
        h.run100(obj)
        row = h.event(writer)
        self.assertEqual(row['fairness_compare_status'], 'not_evaluated')
        self.assertEqual(row['actual_admission_order'], ['a'])

    def test_observe_switch_is_checked_at_scheduler_startup(self):
        import sys
        import types
        native = existing.base.base.native
        ns = native.scheduler_namespace()
        ns.update(EventPublisherFactory=types.SimpleNamespace(create=lambda *args: None),
                  compute_encoder_budget=lambda **kwargs: (0, 0),
                  EncoderCacheManager=lambda **kwargs: types.SimpleNamespace(),
                  KVCacheManager=lambda **kwargs: native.FakeKV())
        cfg = types.SimpleNamespace(
            scheduler_config=types.SimpleNamespace(max_num_seqs=8, max_num_batched_tokens=20,
                max_model_len=100, policy='fcfs', scheduler_cls='vllm.v1.core.sched.scheduler.Scheduler'),
            cache_config=types.SimpleNamespace(num_gpu_blocks=100, block_size=16,
                enable_prefix_caching=False, prefix_caching_hash_algo='builtin'),
            model_config=types.SimpleNamespace(is_multimodal_model=False, is_encoder_decoder=False,
                model='Meta-Llama-3-8B-Instruct', tokenizer='Meta-Llama-3-8B-Instruct',
                hf_config=types.SimpleNamespace(model_type='llama', hidden_size=4096,
                    num_hidden_layers=32, vocab_size=128256), encoder_config=None),
            lora_config=None, kv_events_config=None, kv_transfer_config=None, speculative_config=None,
            parallel_config=types.SimpleNamespace(data_parallel_rank=0, data_parallel_size=1,
                pipeline_parallel_size=1, tensor_parallel_size=1))
        with patch.dict(sys.modules, {'vllm.v1.core.sched.r1_admission': A}):
            for env in ({'VLLM_R1_FAIRNESS_COMPARE_OBSERVE':'1',
                         'VLLM_R1_FAIRNESS_WAIT_LIMIT_SECONDS':'30'},
                        {'VLLM_R1_FAIRNESS_COMPARE_OBSERVE':'invalid'}):
                with self.subTest(env=env), patch.dict(os.environ, env, clear=True):
                    with self.assertRaises(ValueError):
                        ns['Scheduler'](cfg, types.SimpleNamespace(), types.SimpleNamespace())

    def test_predictor_failure_does_not_run_comparison(self):
        h, obj, predictor, writer = self.make()
        obj.r1_shadow.prefill_predictor.fail = True
        h.run100(obj)
        row = h.event(writer)
        self.assertEqual(row['reason'], 'type_prediction_or_decision_failed')
        self.assertEqual(row['fairness_compare_status'], 'not_evaluated')
        self.assertEqual(row['actual_admission_order'], ['a'])
        self.assertEqual([r.request_id for r in obj.waiting], list('bcd'))
        self.assertEqual(predictor.calls, 1)
        self.assertEqual(obj.r1_shadow.prefill_predictor.calls, 1)
