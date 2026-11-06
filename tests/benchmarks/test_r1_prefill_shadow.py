"""CPU tests with fake B/prefill heads and Scheduler resource stubs."""
import copy, hashlib, importlib.util, json, os, sys, tempfile, time, types, unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import test_r1_shadow as old
S=old.S; native=old.native
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('prefill_offline',ROOT/'tools/r1_shadow/analyze_prefill.py')
analysis=importlib.util.module_from_spec(spec);spec.loader.exec_module(analysis)
spec2=importlib.util.spec_from_file_location('prefill_compare',ROOT/'tools/r1_shadow/compare_prefill_runs.py')
compare=importlib.util.module_from_spec(spec2);spec2.loader.exec_module(compare)
class PF:
 def __init__(self,fail=False): self.calls=[];self.fail=fail;self.sha256='a'*64
 def predict(self,candidates,running,kv):
  self.calls.append(([r.request_id for r in candidates],running,kv))
  if self.fail:raise RuntimeError('private')
  return [[.1,.2,.3] for _ in candidates]
class Tests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory(prefix='r1-prefill-test-',dir=ROOT);self.addCleanup(self.tmp.cleanup);self.d=Path(self.tmp.name)
  self.split=self.d/'input_split.jsonl'
  rows=[{'input_hash':f'{i:064x}','prompt_hashes':[f'{i+2000:064x}'],
         'split':'train' if i<1447 else 'validation'} for i in range(1809)]
  self.split.write_text(''.join(json.dumps(row)+'\n' for row in rows),encoding='utf8')
  self.addCleanup(patch.stopall)
  patch.object(analysis,'SPLIT_SHA256',hashlib.sha256(self.split.read_bytes()).hexdigest()).start()
 def make(self,prefill=None,every=1,timing=False):
  config=S.ShadowConfig(self.d/'none','unit',self.d,32,128,every,False,None,timing)
  writer=native.C.RecordWriter(native.C.CollectionConfig('unit',self.d),'shadow-core',{'prefill_enabled':prefill is not None})
  obj=S.Shadow(config,old.FakePredictor(),writer,prefill);self.addCleanup(obj.close);return obj
 def test_config_gates(self):
  base={'VLLM_R1_SHADOW':'1','VLLM_R1_SHADOW_MODEL':'x','VLLM_R1_SHADOW_RUN_ID':'x','VLLM_R1_SHADOW_DIR':'x'}
  with patch.dict(os.environ,base,clear=True):self.assertIsNone(S.ShadowConfig.from_env().prefill_model)
  for extras in ({'VLLM_R1_PREFILL_SHADOW':'1'}, {'VLLM_R1_PREFILL_SHADOW':'bad'}, {'VLLM_R1_PREFILL_TIMING':'1','VLLM_R1_ADMISSION_MODE':'observe'}):
   with patch.dict(os.environ,dict(base,**extras),clear=True),self.assertRaises(ValueError):S.ShadowConfig.from_env()
  with patch.dict(os.environ,{'VLLM_R1_PREFILL_SHADOW':'1'},clear=True),self.assertRaises(ValueError):S.ShadowConfig.from_env()
  required=dict(base,VLLM_R1_PREFILL_SHADOW='1',VLLM_R1_PREFILL_MODEL='x',VLLM_R1_PREFILL_PROVENANCE='report-id')
  with patch.dict(os.environ,required,clear=True),self.assertRaises(ValueError):S.ShadowConfig.from_env()
  for value in ('', 'a'*63, 'g'*64):
   with patch.dict(os.environ,dict(required,VLLM_R1_PREFILL_MODEL_SHA256=value),clear=True),self.assertRaises(ValueError):S.ShadowConfig.from_env()
  with patch.dict(os.environ,dict(required,VLLM_R1_PREFILL_MODEL_SHA256='A'*64),clear=True):
   self.assertEqual(S.ShadowConfig.from_env().prefill_sha256,'a'*64)
  with patch.dict(os.environ,dict(required,VLLM_R1_PREFILL_MODEL_SHA256='a'*64,VLLM_R1_PREFILL_PROVENANCE=''),clear=True),self.assertRaises(ValueError):S.ShadowConfig.from_env()
 def test_scheduler_sweep_same_queue_and_log(self):
  pf=PF();obj=self.make(pf,every=2,timing=True);queue=[old.request('a'),old.request('b')]
  scheduler=native.scheduler(waiting=queue);scheduler.r1_shadow=obj
  scheduler.schedule()
  self.assertEqual([r.request_id for r in scheduler.running],['a','b'])
  self.assertEqual(len(pf.calls),1);self.assertEqual(len(obj.predictor.dynamic_calls),1)
  self.assertEqual(obj.counts['prefill_calls'],1)
  self.assertEqual(obj.duration_report()['overall']['schedule_wall_v1']['n'],1)
  obj.close();records=[json.loads(x) for x in obj.writer.path.read_text(encoding='utf8').splitlines()]
  pred=next(r for r in records if r['record_type']=='shadow_prefill_v1')
  self.assertEqual([x['engine_request_id'] for x in pred['candidates']],['a','b'])
  self.assertEqual(pred['target'],'first_scheduled_to_first_token_batch_seconds')
  self.assertEqual(pred['prefill_model_sha256'],'a'*64)
  self.assertEqual(pred['target_source'],'operator_attested_training_provenance')
  self.assertEqual(pred['candidate_count'],2)
 def test_failure_isolated_and_interval_skips(self):
  pf=PF(True);obj=self.make(pf,every=2);q=native.deque([old.request('a')]);before=list(q)
  self.assertEqual(obj.sweep(q,[],types.SimpleNamespace(usage=.2),20,3)[1],('a',))
  self.assertEqual(list(q),before);self.assertEqual(obj.counts['prefill_errors'],1)
  self.assertEqual(obj.sweep(q,[],types.SimpleNamespace(usage=.2),20,4)[1],())
  self.assertEqual(len(pf.calls),1)
  with patch.object(obj,'observe_prefill',side_effect=RuntimeError('private')):
   self.assertEqual(obj.sweep(q,[],types.SimpleNamespace(usage=.2),20,5)[1],('a',))
  self.assertEqual(obj.counts['prefill_instrumentation_errors'],1)
 def test_off_no_prefill_calls(self):
  obj=self.make();q=native.deque([old.request('a')]);obj.sweep(q,[],types.SimpleNamespace(usage=.2),20,3)
  self.assertEqual(obj.counts['prefill_calls'],0)
 def test_hash_before_deserialize_and_schema(self):
  model_path=self.d/'head.joblib';model_path.write_bytes(b'trusted-test')
  digest=hashlib.sha256(b'trusted-test').hexdigest(); fake=types.SimpleNamespace(load=unittest.mock.Mock())
  with patch.dict(sys.modules,{'joblib':fake}):
   with self.assertRaisesRegex(ValueError,'SHA256'):S.PrefillPredictor(model_path,'0'*64)
   fake.load.assert_not_called()
  estimator=types.SimpleNamespace(n_features_in_=3,predict=lambda x:[1.]*len(x))
  fake.load.return_value={'features':S.PREFILL_FIELDS,'quantiles':S.QUANTILES,'models':[estimator]*3}
  with patch.dict(sys.modules,{'joblib':fake}):
   head=S.PrefillPredictor(model_path,digest)
   self.assertEqual(fake.load.call_args.args[0].getvalue(),b'trusted-test')
   self.assertEqual(head.sha256,digest)
   self.assertEqual(head.predict([old.request('a')],0,.2),[[1.,1.,1.]])
   fake.load.return_value['features']=['kv_usage','running_count','num_prompt_tokens']
   with self.assertRaisesRegex(ValueError,'schema'):S.PrefillPredictor(model_path,digest)
   with self.assertRaisesRegex(ValueError,'cannot read'):S.PrefillPredictor(self.d/'missing',digest)
 def test_load_uses_verified_bytes_even_if_file_changes(self):
  model_path=self.d/'head.joblib';model_path.write_bytes(b'original')
  digest=hashlib.sha256(b'original').hexdigest()
  estimator=types.SimpleNamespace(n_features_in_=3,predict=lambda x:[1.]*len(x))
  artifact={'features':S.PREFILL_FIELDS,'quantiles':S.QUANTILES,'models':[estimator]*3}
  def load(stream):
   model_path.write_bytes(b'replaced')
   self.assertEqual(stream.read(),b'original')
   return artifact
  with patch.dict(sys.modules,{'joblib':types.SimpleNamespace(load=load)}):
   self.assertEqual(S.PrefillPredictor(model_path,digest).sha256,digest)
 def test_invalid_artifact_and_predictions(self):
  model_path=self.d/'head.joblib';model_path.write_bytes(b'x');digest=hashlib.sha256(b'x').hexdigest()
  estimator=types.SimpleNamespace(n_features_in_=3,predict=lambda x:[1.]*len(x))
  artifact={'features':S.PREFILL_FIELDS,'quantiles':S.QUANTILES,'models':[estimator]*3}
  fake=types.SimpleNamespace(load=unittest.mock.Mock())
  with patch.dict(sys.modules,{'joblib':fake}):
   for invalid in (dict(artifact,quantiles=[.5,.9]),dict(artifact,models=[estimator]),dict(artifact,models=[types.SimpleNamespace(n_features_in_=2,predict=lambda x:[]) ]*3)):
    fake.load.return_value=invalid
    with self.assertRaisesRegex(ValueError,'schema'):S.PrefillPredictor(model_path,digest)
   fake.load.return_value=dict(artifact,models=[types.SimpleNamespace(n_features_in_=3,predict=lambda x:[float('nan')]*len(x))]*3)
   with self.assertRaisesRegex(ValueError,'invalid prefill prediction'):S.PrefillPredictor(model_path,digest).predict([old.request('a')],0,.2)
   fake.load.return_value=dict(artifact,models=[types.SimpleNamespace(n_features_in_=3,predict=lambda x:[])]*3)
   with self.assertRaisesRegex(ValueError,'invalid prefill prediction'):S.PrefillPredictor(model_path,digest).predict([old.request('a')],0,.2)
 def test_same_batch_admission_join_and_loss(self):
  obj=self.make(PF());req=old.request('a');q=native.deque([req,old.request('b')])
  decision=obj.sweep(q,[],types.SimpleNamespace(usage=.2),20,3);obj.admit(req,decision)
  obj.observe_elapsed('schedule_wall_v1',time.perf_counter_ns(),'success');obj.close()
  f={'stats_status':'valid','timestamps':{'scheduled_ts':3.,'first_token_ts':3.2},'native_stats':{'prefill_time':.2}}
  fixed=json.loads(self.split.read_text(encoding='utf8').splitlines()[0]);sample={'request_index':0,'input_hash':fixed['input_hash'],'prompt_hash':fixed['prompt_hashes'][0],'task_type':'chat'}
  meta={'request_rate':4,'max_concurrency':16,'seed':1,'model':'m','burstiness':1,'sampling_params':{},'ignore_eos':False}
  merged={'recording_status':'complete','client_batch_counts_complete':True,'unassociated_records':[],'collection_complete':True,'files':[{'header':{'role':'client','metadata':meta}}],'requests':[{'run_id':'unit','engine_request_id':'a','status':['complete'],'attempts':[{'phase':'main','sample':sample}],'finished':[f]}]}
  p=self.d/'merged.json';p.write_text(json.dumps(merged),encoding='utf8')
  result=analysis.analyze(obj.writer.path,p,self.split)
  self.assertEqual(result['status'],'exploratory_eligible')
  self.assertEqual(result['counts']['matched_first_admission_finished'],1)
  self.assertEqual(result['sweep_opportunities'],1)
  self.assertEqual(result['prefill_calls'],1)
  self.assertEqual(result['candidate_count_distribution'],{2:1})
  self.assertEqual(result['b_candidate_count_distribution'],{2:1})
  self.assertEqual(result['counts']['candidate_predictions_without_same_decision_admission'],1)
  self.assertEqual(result['quality']['0.9']['n'],1)
  self.assertEqual(result['grouped_quality'][fixed['split']]['matched'],1)
  self.assertEqual(result['grouped_quality'][fixed['split']]['missing'],0)
  wrong=copy.deepcopy(merged);wrong['requests'][0]['attempts'][0]['sample']['input_hash']='unknown'
  p.write_text(json.dumps(wrong),encoding='utf8')
  with self.assertRaisesRegex(ValueError,'not in fixed split'):analysis.analyze(obj.writer.path,p,self.split)
  wrong=copy.deepcopy(merged);extra=copy.deepcopy(wrong['requests'][0]);extra['engine_request_id']='other';wrong['requests'].append(extra)
  p.write_text(json.dumps(wrong),encoding='utf8')
  with self.assertRaisesRegex(ValueError,'duplicate main input_hash'):analysis.analyze(obj.writer.path,p,self.split)
  p.write_text(json.dumps(merged),encoding='utf8')
  wrong_split=self.d/'wrong_split.jsonl';wrong_split.write_bytes(self.split.read_bytes()+b' ')
  with self.assertRaisesRegex(ValueError,'split SHA256 mismatch'):analysis.analyze(obj.writer.path,p,wrong_split)
  self.assertEqual(analysis.analyze(obj.writer.path,p,self.split,min_calls=2)['status'],'insufficient_samples')
  merged['collection_complete']=False;p.write_text(json.dumps(merged),encoding='utf8')
  with self.assertRaisesRegex(ValueError,'incomplete'):analysis.analyze(obj.writer.path,p,self.split)
  merged['collection_complete']=True
  merged['requests'][0]['finished'][0]['native_stats']['prefill_time']=.4
  p.write_text(json.dumps(merged),encoding='utf8')
  invalid=analysis.analyze(obj.writer.path,p,self.split)
  self.assertEqual(invalid['counts']['invalid_native_prefill'],1)
  self.assertIsNone(invalid['quality'])
  merged['requests'][0]['attempts'][0]['phase']='probe'
  p.write_text(json.dumps(merged),encoding='utf8')
  insufficient=analysis.analyze(obj.writer.path,p,self.split)
  self.assertEqual(insufficient['counts']['probe_excluded'],1)
  self.assertEqual(insufficient['status'],'insufficient_samples')
  lines=obj.writer.path.read_text(encoding='utf8').splitlines();footer=json.loads(lines[-1]);footer['records_complete']=False;lines[-1]=json.dumps(footer)
  damaged=self.d/'damaged.jsonl';damaged.write_text('\n'.join(lines)+'\n',encoding='utf8')
  with self.assertRaisesRegex(ValueError,'incomplete'):analysis.analyze(damaged,p,self.split)
 def test_failed_prefill_cannot_be_quality_eligible(self):
  obj=self.make(PF(True));obj.sweep(native.deque([old.request('a')]),[],types.SimpleNamespace(usage=.2),20,3)
  obj.observe_elapsed('schedule_wall_v1',time.perf_counter_ns(),'success');obj.close()
  data={'recording_status':'complete','client_batch_counts_complete':True,'unassociated_records':[],
        'collection_complete':True,'files':[],'requests':[]}
  path=self.d/'merged_error.json';path.write_text(json.dumps(data),encoding='utf8')
  with self.assertRaisesRegex(ValueError,'prefill prediction or shadow logging error'):
   analysis.analyze(obj.writer.path,path,self.split)
 def test_independent_run_comparison(self):
  dist={'schedule_wall_v1':{'n':50},'prefill_foreground_v1':{'n':40}}
  common={'status':'exploratory_eligible','main_identity_sha256':'x','main_request_count':1809,'fixed_split_sha256':'split',
          'request_rate':4,'max_concurrency':16,'client_config':{'seed':1},'shadow_summary':{'foreground_distributions':{'overall':dist}}}
  off=dict(common,run_id='off',prefill_enabled=False)
  on=dict(common,run_id='on',prefill_enabled=True)
  self.assertEqual(compare.compare(off,on)['status'],'exploratory_comparison')
  self.assertNotIn('paired_delta',compare.compare(off,on))
  self.assertEqual(compare.compare(off,dict(on,status='insufficient_samples'))['status'],'not_eligible')
  self.assertEqual(compare.compare(off,dict(on,run_id='off'))['status'],'not_eligible')
  self.assertEqual(compare.compare(off,dict(on,main_identity_sha256='other'))['status'],'not_eligible')
if __name__=='__main__':unittest.main()
