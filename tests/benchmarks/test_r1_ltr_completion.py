"""Completion method tests with a fake frontend and a fake async scorer."""
import os
import sys
import types
import typing
import unittest
from unittest.mock import Mock, patch
import test_request_collector as native
import test_r1_shadow as fixtures
import test_r1_ltr_standalone as ltr_tests

NS=types.SimpleNamespace
class CompletionTests(unittest.TestCase):
    def test_completion_scores_normalized_text_once_and_binds_id(self):
        request=fixtures.completion_request(suffix=None,echo=False,stream=True,best_of=None,model='model',priority=0)
        params=native.params(max_tokens=2048)
        request.to_sampling_params=lambda *args:params
        class Fake:
            async def score(self,request_id,prompt,timeout):
                self.inputs=(request_id,prompt,timeout)
                return ltr_tests.module.ScoreResult(request_id,1.25,7,9)
        scorer=Fake()
        ns=dict(native.N,cast=typing.cast,Union=typing.Union,EmbedsPrompt=dict,TokensPrompt=dict,
                SamplingParams=NS,BeamSearchParams=type('Beam',(),{}),RequestResponseMetadata=lambda **kw:NS(**kw),
                is_embeds_prompt=lambda p:'prompt_embeds' in p,is_tokens_prompt=lambda p:'prompt_token_ids' in p,
                merge_async_iterators=lambda *args:object(),OpenAIServing=object,
                logger=NS(info=Mock(), warning=Mock()))
        native.load_defs('vllm/entrypoints/openai/serving_completion.py',['OpenAIServingCompletion'],ns)
        obj=ns['OpenAIServingCompletion'].__new__(ns['OpenAIServingCompletion'])
        obj.engine_client=NS(errored=False,r1_shadow=None,r1_ltr=scorer,r1_ltr_timeout=2.,generate=Mock(return_value=object()))
        async def noop(*args):return None
        async def preprocess(*args,**kw):return ([{'prompt':request.prompt,'prompt_token_ids':[1,2]}],[{'prompt_token_ids':[1,2]}])
        obj.engine_client.get_tokenizer=noop;obj._check_model=noop;obj._base_request_id=lambda _:'id'
        obj._maybe_get_adapters=lambda _:(None,None);obj._preprocess_completion=preprocess;obj._log_inputs=Mock()
        obj._get_model_name=lambda *args:'model';obj.completion_stream_generator=Mock(return_value='stream')
        obj.max_model_len=8192;obj.default_sampling_params={};obj.model_config=NS(logits_processor_pattern=None)
        with patch.dict(os.environ,{'VLLM_R1_TYPE_SLO':'0'}),patch.dict(sys.modules,{'vllm.v1.metrics.r1_ltr_predictor':ltr_tests.module}):
            self.assertEqual(native.run_async(obj.create_completion(request)),'stream')
        self.assertEqual(scorer.inputs,('cmpl-id-0',request.prompt,2.))
        self.assertEqual(params.extra_args[ltr_tests.module.LTR_INTERNAL_KEY].request_id,'cmpl-id-0')
