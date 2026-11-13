"""Offline prototype tests; this is not a production microbatch implementation."""
import asyncio, importlib.util, sys, time, types, unittest
from pathlib import Path
p=Path(__file__).resolve().parents[2]/'tools/r1_ltr_v1/microbatch_queue_prototype.py'
spec=importlib.util.spec_from_file_location('microbatch_proto_test',p);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
import torch
class FakeTokenizer:
 def __call__(self,texts,**kwargs):return {'input_ids':torch.tensor([[len(s)] for s in texts])}
class FakeModel:
 def __init__(self,delay=0,fail=False):self.delay=delay;self.fail=fail
 def __call__(self,**kwargs):
  time.sleep(self.delay)
  if self.fail:raise ValueError('fake model failure')
  return types.SimpleNamespace(logits=kwargs['input_ids'].float())
class PrototypeTests(unittest.IsolatedAsyncioTestCase):
 async def test_mixed_batch_preserves_identity_and_scores(self):
  s=types.SimpleNamespace(tokenizer=FakeTokenizer(),model=FakeModel())
  texts=['a','bbbb','a','cc'];ref=[float(len(x)) for x in texts]
  r=await module.experiment(s,texts,ref,[1]*4,2,.02,10000,8,1.)
  self.assertEqual([x['status'] for x in r['rows']],['success']*4)
  self.assertEqual([x['score'] for x in r['rows']],ref)
  self.assertTrue(all(x['size']<=2 for x in r['batches']))
 async def test_bounded_queue_and_timeout_late_result(self):
  s=types.SimpleNamespace(tokenizer=FakeTokenizer(),model=FakeModel(delay=.1))
  r=await module.experiment(s,['a']*4,[1.]*4,[1]*4,2,.01,10000,2,.02)
  statuses=[x['status'] for x in r['rows']]
  self.assertEqual(statuses.count('queue_full'),2)
  self.assertEqual(statuses.count('timeout'),2)
  self.assertTrue(any(x.get('late_completion') for x in r['rows']))
  self.assertFalse(any('score' in x for x in r['rows']))
 async def test_model_error_fails_only_batch(self):
  s=types.SimpleNamespace(tokenizer=FakeTokenizer(),model=FakeModel(fail=True))
  r=await module.experiment(s,['a','bb'],[1.,2.],[1,2],2,.01,10000,2,1.)
  self.assertTrue(all(x['status']=='model_exception' for x in r['rows']))
 async def test_external_cancellation_waits_for_worker_shutdown(self):
  import threading
  started,release,finished=threading.Event(),threading.Event(),threading.Event()
  class SlowModel:
   def __call__(self,**kwargs):
    started.set();release.wait(2);finished.set()
    return types.SimpleNamespace(logits=kwargs['input_ids'].float())
  s=types.SimpleNamespace(tokenizer=FakeTokenizer(),model=SlowModel())
  task=asyncio.create_task(module.experiment(s,['a','bb'],[1.,2.],[1,2],2,.01,10000,2,5.))
  self.assertTrue(await asyncio.to_thread(started.wait,1))
  task.cancel();await asyncio.sleep(.01);self.assertFalse(task.done())
  release.set()
  with self.assertRaises(asyncio.CancelledError):await asyncio.wait_for(task,1)
  self.assertTrue(finished.is_set())
 async def test_nonfinite_score_rejects_batch(self):
  class NanModel:
   def __call__(self,**kwargs):return types.SimpleNamespace(logits=torch.full((len(kwargs['input_ids']),1),float('nan')))
  s=types.SimpleNamespace(tokenizer=FakeTokenizer(),model=NanModel())
  r=await module.experiment(s,['a','bb'],[1.,2.],[1,2],2,.01,10000,2,1.)
  self.assertTrue(all(x['status']=='model_exception' for x in r['rows']))
  self.assertFalse(any('score' in x for x in r['rows']))
