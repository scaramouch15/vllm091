import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
import sys
import subprocess
import types
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[2] / 'benchmarks' / 'r1_prefill_manifest.py'
spec = importlib.util.spec_from_file_location('r1_prefill_manifest_test', PATH)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

def h(x): return hashlib.sha256(x.encode()).hexdigest()
def row(i, prompt, kind='code', inp=None):
    return {'input_hash': inp or h('input'+str(i)), 'prompt_hash':h(prompt),
            'prompt':prompt, 'prompt_len':len(prompt), 'max_tokens':2048,
            'request_index':i, 'task_type':kind}

class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.path=Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        (self.path/'tokenizer.json').write_text('{}')
        self.source=self.path/'source'; self.source.write_text('source')
        self.rows=[row(0,'one'),row(1,'two','summary'),row(2,'three','chat')]
        self.run={'files':{'code':[{'source_file':str(self.source)}]},
                  'tokenizer':{'name':'test'},'tokenizer_id':str(self.path),
                  'tokenizer_mode':'auto','counts':{'code':3},'seed':12,
                  'mixed_output_len':2048}
    def part(self, rows=None, exclusions=None):
        return m.partition({'run':self.run,'requests':rows or self.rows},exclusions or {},12)
    def test_transitive_two_hash_group(self):
        rows=[row(0,'a',inp=h('x')),row(1,'b',inp=h('x')),row(2,'b',inp=h('y')),row(3,'c')]
        a=self.part(rows)
        self.assertTrue(any(len(s)==3 for s in a['splits'].values()))
        b=self.part(rows, {'old':({h('y')},set())})
        self.assertEqual(sum(len(s) for s in b['splits'].values()),1)
        self.assertEqual(b['summary']['exclusion_overlaps']['old']['candidate_rows'],1)
    def test_shortfall_and_duplicate(self):
        self.assertEqual(self.part()['summary']['shortfall'],1797)
        with self.assertRaises(ValueError): self.part(self.rows+[self.rows[0]])
    def test_replay_valid_and_reject_mismatch(self):
        approved=self.part()
        class Token: pass
        class Dataset:
            random_seed=12; counts={'code':3}; output_len=2048
            tokenizer_info={'name':'test'}; files=self.run['files']
            def sample(_,token,n):
                return [type('Req',(),{'metadata':r})() for r in self.rows]
        ds=Dataset(); split=next(k for k,v in approved['splits'].items() if v)
        self.assertTrue(m.replay(approved,ds,Token(),str(self.path),'auto',split))
        changed=copy.deepcopy(approved); changed['splits'][split][0]['prompt']='wrong'
        with self.assertRaises(ValueError): m.replay(changed,ds,Token(),str(self.path),'auto',split)
        changed=copy.deepcopy(approved); changed['splits'][split].append(changed['splits'][split][0])
        with self.assertRaises(ValueError): m.replay(changed,ds,Token(),str(self.path),'auto',split)
        changed=copy.deepcopy(approved); changed['splits'][split].append(row(99, 'extra'))
        with self.assertRaises(ValueError): m.replay(changed,ds,Token(),str(self.path),'auto',split)
        changed=copy.deepcopy(approved)
        another=next(k for k,v in changed['splits'].items() if k != split and v)
        changed['splits'][another].clear()
        with self.assertRaisesRegex(ValueError, 'count mismatch'):
            m.replay(changed,ds,Token(),str(self.path),'auto',split)
        changed=copy.deepcopy(approved); changed['splits'][split][0]['prompt_hash']=h('bad')
        with self.assertRaises(ValueError): m.replay(changed,ds,Token(),str(self.path),'auto',split)
        self.source.write_text('changed')
        with self.assertRaises(ValueError): m.replay(approved,ds,Token(),str(self.path),'auto',split)
    def report(self, sample):
        att={'phase':'main','run_id':'r1','engine_request_id':'e1','sample':sample}
        f={'run_id':'r1','engine_request_id':'e1','stats_status':'valid','timestamps':{'queued_ts':1.,'scheduled_ts':2.,'first_token_ts':3.,'last_token_ts':4.},'native_stats':{'prefill_time':1.}}
        main={'run_id':'r1','engine_request_id':'e1','attempts':[att],'admissions':[{'run_id':'r1','engine_request_id':'e1','snapshot':{'decision_batch_id':'b1'}}], 'finished':[f], 'client_results':[{'run_id':'r1','engine_request_id':'e1','success':True}], 'status':['complete']}
        probe={'attempts':[{'phase':'probe'}]}
        return {'collection_complete':True,'recording_status':'complete','client_batch_counts_complete':True,'unassociated_records':[],'requests':[main,probe]}
    def test_collection_probe_and_failures(self):
        approved={'splits':{'train':[self.rows[0]]}}
        r=self.report(self.rows[0]); result=m.validate_collections(approved,'train',[r])
        self.assertTrue(result['valid']); self.assertEqual(result['reports'][0]['probe_attempts'],1)
        broken=copy.deepcopy(r); broken['requests'][0]['finished'][0]['timestamps']['first_token_ts']=float('nan')
        self.assertFalse(m.validate_collections(approved,'train',[broken])['valid'])
        broken=copy.deepcopy(r); broken['requests'][0]['finished'][0]['native_stats']['prefill_time']=9
        self.assertFalse(m.validate_collections(approved,'train',[broken])['valid'])
        broken=copy.deepcopy(r); broken['requests'][0]['finished']=[]
        self.assertFalse(m.validate_collections(approved,'train',[broken])['valid'])
        broken=copy.deepcopy(r); broken['requests'][0]['finished'][0]['engine_request_id']='wrong'
        self.assertFalse(m.validate_collections(approved,'train',[broken])['valid'])
        broken=copy.deepcopy(r); broken['requests'][0]['admissions']=[]
        self.assertFalse(m.validate_collections(approved,'train',[broken])['valid'])
        broken=copy.deepcopy(r); broken['requests'][0]['attempts'] *= 2
        self.assertFalse(m.validate_collections(approved,'train',[broken])['valid'])
        broken=copy.deepcopy(r); broken['requests'][0]['attempts'][0]['sample']=self.rows[1]
        self.assertFalse(m.validate_collections(approved,'train',[broken])['valid'])
        broken=copy.deepcopy(r); broken['requests']=[]
        self.assertFalse(m.validate_collections(approved,'train',[broken])['valid'])
        broken=copy.deepcopy(r); broken['requests'].append(copy.deepcopy(broken['requests'][0]))
        result=m.validate_collections(approved,'train',[broken])
        self.assertIn('duplicate_engine_request_id', result['reports'][0]['errors'])
        broken=copy.deepcopy(r); broken['recording_status']='incomplete_or_unconfirmed'
        self.assertFalse(m.validate_collections(approved,'train',[broken])['valid'])

    def test_duplicate_run_across_reports(self):
        approved={'splits':{'train':[self.rows[0]]}}
        report=self.report(self.rows[0])
        result=m.validate_collections(approved,'train',[report,copy.deepcopy(report)])
        self.assertFalse(result['valid'])
        self.assertIn('duplicate_run_across_reports',result['reports'][1]['errors'])
        independent=copy.deepcopy(report)
        for row in independent['requests']:
            if row.get('run_id'): row['run_id']='r2'
            for key in ('attempts','admissions','finished','client_results'):
                for item in row.get(key,[]):
                    if item.get('run_id'): item['run_id']='r2'
        self.assertTrue(m.validate_collections(approved,'train',[report,independent])['valid'])

    def test_three_split_run_and_batch_isolation(self):
        def result(split, run, batch):
            return {'split':split,'valid':True,'approved_sha256':h('manifest'),
                    'runs':[run],'decision_batches':[[run,batch]]}
        train=result('train','r1','b1')
        cal=result('calibration','r1','b2')
        test=result('test','r3','b3')
        shared=m.compare_validations([train,cal,test])
        self.assertIn('run_id_cross_split:train:calibration',shared['errors'])
        self.assertFalse(shared['complete'])
        cal=result('calibration','r2','b2')
        partial=m.compare_validations([train,cal],require_all=False)
        self.assertTrue(partial['valid']); self.assertFalse(partial['complete'])
        missing=m.compare_validations([train,cal])
        self.assertIn('missing_split:test',missing['errors'])
        duplicate=m.compare_validations([train,cal,copy.deepcopy(cal)])
        self.assertIn('duplicate_split:calibration',duplicate['errors'])
        self.assertIn('missing_split:test',duplicate['errors'])
        self.assertTrue(m.compare_validations([train,cal,test])['complete'])
        bad=copy.deepcopy(test); bad['approved_sha256']=h('other')
        self.assertIn('approved_manifest_mismatch',m.compare_validations([train,cal,bad])['errors'])
        bad=copy.deepcopy(test); bad['valid']=False
        self.assertIn('invalid_validation:test',m.compare_validations([train,cal,bad])['errors'])
        bad=result('test','r1','b1')
        self.assertIn('batch_cross_split:train:test',m.compare_validations([train,cal,bad])['errors'])

    def test_compare_cli_rejects_incomplete_and_accepts_three(self):
        paths=[]
        for i, split in enumerate(m.SPLITS):
            path=self.path/(split+'.json')
            value={'split':split,'valid':True,'approved_sha256':h('manifest'),
                   'runs':[f'r{i}'],'decision_batches':[[f'r{i}','b']]}
            path.write_text(json.dumps(value), encoding='utf-8'); paths.append(str(path))
        output=self.path/'partial.json'
        p=subprocess.run([sys.executable,'-B',str(PATH),'compare','--validations',*paths[:2],
                          '--output',str(output)],capture_output=True,text=True)
        self.assertEqual(p.returncode,1)
        self.assertIn('missing_split:test',json.loads(output.read_text())['errors'])
        output=self.path/'complete.json'
        p=subprocess.run([sys.executable,'-B',str(PATH),'compare','--validations',*paths,
                          '--output',str(output)],capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr)
        self.assertTrue(json.loads(output.read_text())['complete'])

    def test_real_mixed_sampler_and_replay(self):
        # Load the production mixed implementation with only its native base
        # classes substituted; code JSONL and tokenizer inputs are synthetic.
        native=types.ModuleType('benchmark_dataset')
        @dataclass
        class SampleRequest:
            prompt: str
            prompt_len: int
            expected_output_len: int
        class Base:
            DEFAULT_SEED=0
            def __init__(self, random_seed=0): self.random_seed=random_seed
        native.SampleRequest=SampleRequest; native.BenchmarkDataset=Base
        path=PATH.parent/'mixed_workload_dataset.py'
        spec=importlib.util.spec_from_file_location('mixed_prefill_test',path)
        mixed=importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'benchmark_dataset':native,'mixed_prefill_test':mixed}):
            spec.loader.exec_module(mixed)
        source=self.path/'data.jsonl'
        source.write_text(''.join(json.dumps({'query':f'question {i}','resource':'r','lang':'python'})+'\n' for i in range(5)))
        class Token:
            name_or_path='fake'
            def get_chat_template(self): return 'template'
            def apply_chat_template(self, messages, **kwargs): return '<user>'+messages[0]['content']
            def __call__(self, prompt, **kwargs): return types.SimpleNamespace(input_ids=list(prompt))
        tokenizer=Token()
        def dataset(): return mixed.MixedWorkloadDataset(code_path=str(source),code_count=5,output_len=2048,random_seed=12)
        d=dataset(); candidates=d.sample(tokenizer,5)
        run={**d.report(),'tokenizer_id':str(self.path),'tokenizer_mode':'auto'}
        approved=m.partition({'run':run,'requests':[r.metadata for r in candidates]}, {},12)
        split=next(k for k,v in approved['splits'].items() if v)
        actual=m.replay(approved,dataset(),tokenizer,str(self.path),'auto',split)
        self.assertEqual([r.metadata for r in actual],approved['splits'][split])
        with self.assertRaises(ValueError): m.replay(approved,dataset(),tokenizer,str(self.path),'wrong',split)

if __name__=='__main__': unittest.main()
