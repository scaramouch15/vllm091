"""CPU checks with real shadow/scheduler methods and fake model/resources."""
import dataclasses
import json
import math
import os
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_r1_shadow as base

S, native = base.S, base.native

class Prefill:
    sha256 = 'a' * 64
    def __init__(self, fail=False): self.fail, self.calls = fail, 0
    def predict(self, candidates, running, kv):
        self.calls += 1
        if self.fail: raise ValueError('failed')
        return [[1., 2., 3.] for _ in candidates]

class TypeRankTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=base.ROOT)
        self.addCleanup(self.temp.cleanup)
        self.config = S.ShadowConfig(Path(self.temp.name)/'b', 'unit', Path(self.temp.name),
                                     32, 128, 1, type_rank_shadow=True, type_rank_wait_limit=10.)
        self.writer = native.C.RecordWriter(native.C.CollectionConfig('unit',Path(self.temp.name)), 'shadow-core')
        self.obj = S.Shadow(self.config, base.FakePredictor(), self.writer, Prefill())
        self.addCleanup(self.obj.close)
        self.slos = {'config_seconds': {'chat.ttft': 10., 'chat.tpot': .1, 'code.e2el': 100., 'summary.e2el': 100.}, 'config_sha256': 'digest'}

    def req(self, name, kind, queued, tokens=1):
        r=base.request(name); r.r1_request_type=kind; r.r1_first_queued_timestamp=queued; r.num_prompt_tokens=tokens
        return r

    def records(self):
        self.obj.close()
        return [json.loads(x) for x in self.writer.path.read_text(encoding='utf8').splitlines()]

    def run_sweep(self, requests, slo=None):
        q=native.deque(requests); before=list(q)
        self.obj.sweep(q, [], types.SimpleNamespace(usage=.2), 20, 20., type_slo=self.slos if slo is None else slo)
        self.assertEqual(list(q), before)
        return next(x for x in self.records() if x['record_type']=='shadow_type_rank_v1')

    def test_types_scores_protection_and_no_queue_change(self):
        rows=[self.req('c','code',19.,10),self.req('h','chat',19.),self.req('s','summary',0.,1)]
        record=self.run_sweep(rows)
        self.assertEqual(record['original_ids'],['c','h','s'])
        self.assertEqual(record['comparable_order_ids'],['s','h','c'])
        self.assertAlmostEqual(record['candidates'][1]['score'],.7)
        self.assertAlmostEqual(record['candidates'][0]['score'],.88)
        self.assertEqual(record['candidates'][2]['protection_reason'],'wait_limit')
        self.assertEqual(record['candidates'][1]['tpot_status'],'not_estimable')

    def test_stable_ties(self):
        record=self.run_sweep([self.req('a','code',19.),self.req('b','summary',19.)])
        self.assertEqual(record['comparable_order_ids'],['a','b'])
        self.assertFalse(record['order_changed'])

    def test_missing_time_type_and_prediction_are_partial(self):
        self.obj.prefill_predictor=Prefill(True)
        record=self.run_sweep([self.req('a','chat',19.),self.req('b','code',float('nan')),self.req('c',None,19.)])
        self.assertEqual(record['status'],'partial'); self.assertIsNone(record['order_changed'])
        self.assertEqual([x['missing_reason'] for x in record['candidates']], ['prefill_prediction_unavailable','queued_time_missing_or_invalid','type_missing_or_invalid'])

    def test_missing_config_and_previously_admitted_excluded(self):
        admitted=self.req('old','code',19.); admitted.r1_shadow_admitted=True
        preempted=self.req('preempted','code',19.); preempted.status=native.N['RequestStatus'].PREEMPTED
        q=native.deque([admitted,preempted,self.req('new','code',19.)])
        self.obj.sweep(q,[],types.SimpleNamespace(usage=.2),20,20.,type_slo=None)
        record=next(x for x in self.records() if x['record_type']=='shadow_type_rank_v1')
        self.assertEqual(record['original_ids'],['new'])
        self.assertEqual(record['candidates'][0]['missing_reason'],'slo_config_unavailable')

    def test_b_prediction_failure_produces_no_ranking(self):
        self.obj.predictor.dynamic=lambda matrix: (_ for _ in ()).throw(ValueError('bad'))
        self.obj.sweep(native.deque([self.req('a','code',19.)]),[],types.SimpleNamespace(usage=.2),20,20.,type_slo=self.slos)
        kinds=[x['record_type'] for x in self.records()]
        self.assertIn('shadow_dynamic_error',kinds)
        self.assertNotIn('shadow_type_rank_v1',kinds)

    def test_scheduler_actual_admission_order_unchanged(self):
        queued=time.monotonic()-1
        requests=[self.req('a','code',queued,10),self.req('b','chat',queued)]
        scheduler=native.scheduler(waiting=requests);scheduler.r1_shadow=self.obj
        scheduler.r1_type_slo=types.SimpleNamespace(snapshot=lambda:self.slos)
        scheduler.schedule()
        self.assertEqual([r.request_id for r in scheduler.running],['a','b'])
        record=next(x for x in self.records() if x['record_type']=='shadow_type_rank_v1')
        self.assertEqual(record['comparable_order_ids'],['b','a'])

    def test_disabled_and_config_rejections(self):
        self.obj.config=dataclasses.replace(self.config,type_rank_shadow=False)
        self.obj.sweep(native.deque([self.req('a','code',19.)]),[],types.SimpleNamespace(usage=.2),20,20.)
        self.assertFalse(any(x['record_type']=='shadow_type_rank_v1' for x in self.records()))
        env={'VLLM_R1_SHADOW':'1','VLLM_R1_SHADOW_MODEL':'x','VLLM_R1_SHADOW_RUN_ID':'x','VLLM_R1_SHADOW_DIR':'x','VLLM_R1_TYPE_RANK_SHADOW':'1'}
        with patch.dict(os.environ,env,clear=True), self.assertRaises(ValueError): S.ShadowConfig.from_env()
        env.update(VLLM_R1_PREFILL_SHADOW='1',VLLM_R1_PREFILL_MODEL='x',VLLM_R1_PREFILL_MODEL_SHA256='a'*64,VLLM_R1_PREFILL_PROVENANCE='test',VLLM_R1_TYPE_SLO='1')
        with patch.dict(os.environ,env,clear=True):
            self.assertEqual(S.ShadowConfig.from_env().type_rank_wait_limit,30.)
        for limit in ('0','nan','inf','-1','bad'):
            with patch.dict(os.environ,dict(env,VLLM_R1_TYPE_RANK_WAIT_LIMIT_SECONDS=limit),clear=True), self.assertRaises(ValueError): S.ShadowConfig.from_env()

if __name__ == '__main__': unittest.main()
