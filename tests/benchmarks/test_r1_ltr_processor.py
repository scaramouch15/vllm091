"""Actual Processor method extracted with dependency stubs; no real engine."""
import copy
import sys
import types
import unittest
from unittest.mock import Mock, patch
import test_request_collector as native
import test_r1_shadow as shadow_tests
import test_r1_ltr_standalone as ltr_tests

NS=types.SimpleNamespace
class LTRProcessorTests(unittest.TestCase):
    def test_processor_binds_typed_score_and_removes_forgery(self):
        namespace=dict(native.N, SamplingParams=NS, EngineCoreRequest=NS,
                       MULTIMODAL_REGISTRY=None, split_enc_dec_inputs=lambda x:(None,x))
        native.load_defs('vllm/v1/engine/processor.py',['Processor'],namespace)
        obj=namespace['Processor'].__new__(namespace['Processor'])
        obj.r1_shadow=None
        obj.model_config=NS(max_model_len=8192)
        obj.vllm_config=NS(parallel_config=NS(data_parallel_size=1))
        obj._validate_lora=obj._validate_params=obj._validate_model_inputs=Mock()
        obj.use_hash=False; obj.generation_config_fields={}
        obj.tokenizer=NS(get_lora_tokenizer=lambda _:None)
        obj.input_preprocessor=NS(preprocess=lambda *a,**kw:{'type':'token','prompt_token_ids':[1,2]},get_eos_token_id=lambda _:0)
        platform=types.ModuleType('vllm.platforms'); platform.current_platform=NS(validate_request=Mock())
        for value,expected in ((ltr_tests.module.ScoreResult('req',-0.5,1,1),-0.5),
                               (ltr_tests.module.ScoreResult('other',2.,1,1),None),
                               ({'request_id':'req','score':3},None)):
            params=native.params(max_tokens=2048,extra_args={ltr_tests.module.LTR_INTERNAL_KEY:value,'keep':1})
            params.clone=lambda: native.params(max_tokens=params.max_tokens,extra_args=copy.deepcopy(params.extra_args),update_from_generation_config=Mock(),update_from_tokenizer=Mock())
            with patch.dict(sys.modules,{'vllm.platforms':platform,'vllm.v1.metrics.r1_ltr_predictor':ltr_tests.module}):
                _,ecr=obj.process_inputs('req',{'prompt_token_ids':[1,2]},params)
            self.assertEqual(ecr.r1_ltr_score,expected)
            self.assertEqual(ecr.sampling_params.extra_args,{'keep':1})
            request = native.N['Request'].from_engine_core_request(NS(**vars(ecr), client_index=0))
            self.assertEqual(request.r1_ltr_score, expected)
            self.assertFalse(request.r1_ltr_admitted)
