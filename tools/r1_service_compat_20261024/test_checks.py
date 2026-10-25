"""Stdlib-only safety/plan checks, without importing the service or predictor."""
import ast
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from environment_probe import clean
from inspect_service import TARGET,classify,environment,source_check


def snapshot(extra=None):
    active={'numpy':'2.2.6','scipy':'1.15.3','torch':'2.7.0','vllm':'0.9.1'}
    active.update(extra or {})
    return {'python':'3.10.22','active':active}


def plan(items):
    return {'install':[{'metadata':{'name':k,'version':v}} for k,v in items.items()]}


class Checks(unittest.TestCase):
    def test_additive_three_packages(self):
        result=classify(snapshot(),plan({k:v for k,v in TARGET.items() if k not in ['numpy','scipy']}))
        self.assertTrue(result['additive_plan_only'])
        self.assertEqual({x['action'] for x in result['changes']},{'add'})

    def test_already_installed_empty_plan(self):
        self.assertTrue(classify(snapshot(TARGET),plan({}))['additive_plan_only'])

    def test_replacement_even_same_version_stops(self):
        self.assertFalse(classify(snapshot(TARGET),plan({'joblib':'1.5.3'}))['additive_plan_only'])
        self.assertFalse(classify(snapshot({'numpy':'1.26.4'}),plan(TARGET))['additive_plan_only'])

    def test_torch_or_extra_dependency_stops(self):
        for name in ['torch','nvidia-cublas-cu12','packaging']:
            additions={k:v for k,v in TARGET.items() if k not in ['numpy','scipy']}
            self.assertFalse(classify(snapshot(),plan(dict(additions,**{name:'99'})))['additive_plan_only'])

    def test_existing_conflict_or_missing_report_stops(self):
        additions={k:v for k,v in TARGET.items() if k not in ['numpy','scipy']}
        self.assertFalse(classify(snapshot(),plan(additions),pip_ok=False)['additive_plan_only'])
        self.assertFalse(classify(snapshot(),None)['additive_plan_only'])
        self.assertFalse(classify(snapshot(TARGET),{})['additive_plan_only'])

    def test_isolation_and_redaction(self):
        env=environment(True)
        self.assertEqual(env['PYTHONNOUSERSITE'],'1')
        self.assertNotIn('PYTHONHOME',env);self.assertNotIn('PYTHONPATH',env)
        self.assertEqual(clean('https://fixture:dummy@example.invalid/x?token=fixture'),'https://example.invalid/x')

    def test_python310_syntax_no_service_construction(self):
        for p in Path(__file__).parent.glob('*.py'):
            tree=ast.parse(p.read_text(encoding='utf-8'),feature_version=(3,10))
            compile(tree,str(p),'exec')
            for n in ast.walk(tree):
                if isinstance(n,ast.Call) and isinstance(n.func,ast.Name):
                    self.assertNotIn(n.func.id,['LLM','AsyncLLM','EngineCore','Scheduler'])

    def test_source_comparison_without_importing_vllm(self):
        explicit=os.environ.get('R1_COMPAT_TEST_REPO')
        if explicit:
            repo=Path(explicit).resolve()
            self.assertTrue((repo/'vllm/v1/request.py').is_file(),'R1_COMPAT_TEST_REPO must name the vLLM repository')
        else:
            repo=next((p for p in Path(__file__).resolve().parents if (p/'vllm/v1/request.py').is_file()),None)
            if repo is None:
                self.skipTest('Extracted kit: set R1_COMPAT_TEST_REPO to check the real source baseline')
        path=repo/'vllm/__init__.py'
        identity={'modules':{'vllm':{'file':str(path),'version':'0.9.1'}}}
        self.assertTrue(source_check(identity)['passed'])
        identity['modules']['vllm']['version']='0.9.10'
        self.assertFalse(source_check(identity)['passed'])

    def test_cuda_initialization_guards_with_fake_modules(self):
        import cpu_checks
        modules={name:SimpleNamespace(__file__='/fake/'+name+'.py',__version__='fake')
                 for name in ['torch','vllm','numpy','scipy','sklearn','joblib','threadpoolctl']}
        modules['torch'].cuda=SimpleNamespace(is_initialized=lambda:False,_lazy_init=lambda:None)
        modules['torch']._C=SimpleNamespace(_cuda_init=lambda:None,_cuda_getDeviceCount=lambda:1)
        modules['torch'].get_num_threads=lambda:1
        modules['torch'].get_num_interop_threads=lambda:1
        modules['threadpoolctl'].threadpool_info=lambda:[]
        with patch.dict(sys.modules,modules):
            result={}
            cpu_checks.check(SimpleNamespace(mode='identity',order='vllm-first',expected_prefix=None),result)
            self.assertFalse(result['cuda_initialized'])
            for fn in [modules['torch'].cuda._lazy_init,modules['torch']._C._cuda_init,modules['torch']._C._cuda_getDeviceCount]:
                with self.assertRaisesRegex(RuntimeError,'forbidden'):fn()


if __name__=='__main__':unittest.main(verbosity=2)
