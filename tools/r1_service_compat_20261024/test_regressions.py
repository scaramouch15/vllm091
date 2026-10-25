"""Targeted mocked/optimized failure tests; no real service imports or installs."""
import ast
import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import clone_validate as clone
import cpu_checks as cpu
from inspect_service import ROOT,source_check


def write(path,value):
    path.write_text(json.dumps(value),encoding='utf-8')


class OptimizedChecks(unittest.TestCase):
    def test_critical_cpu_failures_under_both_optimization_modes(self):
        code=r'''
import importlib,json,pathlib,platform,sys
from types import SimpleNamespace
import cpu_checks as cpu
case,out,root=sys.argv[1:]
out=pathlib.Path(out);root=pathlib.Path(root)
platform.python_version=lambda:'3.10.22'
versions={'numpy':'2.2.6','scipy':'1.15.3','sklearn':'1.7.2','joblib':'1.5.3','threadpoolctl':'3.6.0'}
modules={n:SimpleNamespace(__file__=sys.executable,__version__=versions.get(n,'fixture'))
         for n in ['torch','vllm','numpy','scipy','sklearn','joblib','threadpoolctl']}
modules['torch'].cuda=SimpleNamespace(is_initialized=lambda:case=='cuda',_lazy_init=lambda:None)
modules['torch']._C=SimpleNamespace(_cuda_init=lambda:None,_cuda_getDeviceCount=lambda:1)
modules['torch'].get_num_threads=lambda:1
modules['torch'].get_num_interop_threads=lambda:1
modules['threadpoolctl'].threadpool_info=lambda:[]
sys.modules.update(modules)
a=SimpleNamespace(mode='identity',order='vllm-first',expected_prefix=pathlib.Path(sys.prefix),out=out,migration_dir=root)
if case=='prefix':a.expected_prefix=root/'wrong-prefix'
if case=='executable':cpu.sys.executable=str(root/'outside-python')
if case=='package_path':modules['numpy'].__file__=str(root/'outside_numpy.py')
if case in ['version','python','hash','manifest_path']:a.mode='imports'
if case=='version':modules['sklearn'].__version__='wrong-version'
if case=='python':platform.python_version=lambda:'3.11.17'
if case in ['hash','manifest_path']:
    a.mode='predict'
    name='payload.txt' if case=='hash' else '../outside.txt'
    (root/'payload.txt').write_text('synthetic payload')
    (root/'files_sha256.json').write_text(json.dumps({name:'0'*64}))
if case=='identity':
    cpu.check=lambda a,r:cpu.validate_samples(list(range(1440)),list(reversed(range(1440))),lambda x:x)
if case=='prediction':
    checks={n:dict(failed_elements=0,failed_records=0,max_absolute_error=0.,max_relative_error=0.)
            for n in ['length','time','length_raw','time_raw']}
    checks['time']['failed_elements']=1
    cpu.check=lambda a,r:cpu.validate_predictions(checks)
cpu.execute(a)
'''
        cases=['prefix','executable','cuda','package_path','version','python','hash','manifest_path','identity','prediction']
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            for optimized in ['flag','environment']:
                for case in cases:
                    with self.subTest(optimized=optimized,case=case):
                        root=Path(directory)/(optimized+'_'+case);root.mkdir()
                        out=root/'result.json'
                        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
                        env.pop('PYTHONOPTIMIZE',None)
                        args=[sys.executable,'-B']
                        if optimized=='flag':args.append('-O')
                        else:env['PYTHONOPTIMIZE']='1'
                        p=subprocess.run([*args,'-c',code,case,str(out),str(root)],cwd=ROOT,env=env,
                                         capture_output=True,text=True)
                        self.assertNotEqual(p.returncode,0,p.stdout+p.stderr)
                        self.assertFalse(json.loads(out.read_text())['passed'])
                        self.assertNotIn('CPU check passed:',p.stdout)
                        self.assertIn('CheckError',p.stderr)

    def test_cpu_checks_has_no_removable_asserts(self):
        tree=ast.parse((ROOT/'cpu_checks.py').read_text(encoding='utf-8'))
        self.assertFalse(any(isinstance(n,ast.Assert) for n in ast.walk(tree)))


class CloneChecks(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ROOT);self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.inspection=self.root/'inspection';self.inspection.mkdir()
        self.prefix=self.root/'clone';self.source=self.root/'service';(self.source/'bin').mkdir(parents=True)
        (self.source/'bin/python').write_text('synthetic interpreter')
        self.repo=self.root/'service_repo';self.repo.mkdir()
        source_file=self.root/'source_code.py';source_file.write_text('# synthetic source\n')
        digest=hashlib.sha256(source_file.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
        names=json.loads((ROOT/'source_baseline.json').read_text())['sources']
        report=dict(schema_version=2,passed=True,checked=list(names),skipped=[],mismatched=[],
                    sources={n:dict(path=str(source_file),lf_sha256=digest) for n in names})
        self.assessment=dict(clone_install_eligible=True,service_python=str(self.source/'bin/python'),
            service_repo=str(self.repo),imported_sources={'isolated':report})
        self.before=dict(prefix=str(self.source),executable=str(self.source/'bin/python'),
                         active={'numpy':'2.2.6','scipy':'1.15.3'},selected={})
        write(self.inspection/'assessment.json',self.assessment)
        write(self.inspection/'isolated_environment.json',self.before)
        self.args=SimpleNamespace(inspection=self.inspection,prefix=self.prefix,out=self.root/'out',
            conda=self.root/'conda',migration_dir=self.root/'migration',max_copy_gib=1.)
        self.calls=[];self.bad=None;self.after_error=False;self.after_changed=False;self.after_location_changed=False

    def clone_snapshot(self):
        pip_file=self.prefix/'lib/site-packages/pip/__init__.py'
        pip_file.parent.mkdir(parents=True,exist_ok=True);pip_file.write_text('# synthetic pip')
        snapshot=dict(prefix=str(self.prefix),executable=str(self.prefix/'bin/python'),active=self.before['active'],
            selected={'pip':dict(version='fixture',distribution_location=str(pip_file.parent.parent),module_spec_origin=str(pip_file))},
            environment={'PIP_USER':'0','PIP_TARGET':None,'PIP_PREFIX':None,'PIP_CONFIG_FILE':os.devnull},
            pip_identity=dict(file=str(pip_file),version='fixture',target_prefix=str(self.prefix),
                scheme={k:str(self.prefix/k) for k in ['purelib','platlib','scripts','headers','data']}))
        if self.bad=='prefix':snapshot['prefix']=str(self.source)
        if self.bad=='executable':snapshot['executable']=str(self.source/'bin/python')
        if self.bad=='pip':snapshot['pip_identity']['file']=str(self.source/'pip.py')
        if self.bad=='pip_metadata':snapshot['selected']['pip']['distribution_location']=str(self.source)
        if self.bad=='target':snapshot['pip_identity']['scheme']['purelib']=str(self.source/'site-packages')
        return snapshot

    def fake_run(self,command,out,name,cwd,env):
        self.calls.append(name)
        if name=='source_size':return '1 synthetic-source'
        if name=='source_before':write(out/'source_before.json',self.before)
        elif name=='conda_clone':
            (self.prefix/'bin').mkdir(parents=True);(self.prefix/'bin/python').write_text('synthetic interpreter')
        elif name=='clone_before':write(out/'clone_before.json',self.clone_snapshot())
        elif name=='download_fixed_wheels':raise RuntimeError('primary-download-failure')
        elif name=='source_after':
            if self.after_error:raise RuntimeError('secondary-ending-check-failure')
            after=copy.deepcopy(self.before)
            if self.after_changed:after['active']['numpy']='changed'
            if self.after_location_changed:after['selected']={'pip':{'origin':'synthetic changed origin'}}
            write(out/'source_after.json',after)
        return ''

    def invoke(self):
        with patch.object(clone,'must_run',side_effect=self.fake_run), \
             patch.object(clone.shutil,'disk_usage',return_value=SimpleNamespace(free=10**15)), \
             patch.object(clone.Path,'home',return_value=self.root),contextlib.redirect_stdout(io.StringIO()):
            clone.validate(self.args)

    def test_bad_clone_identity_prevents_all_pip_actions(self):
        for bad in ['prefix','executable','pip','pip_metadata','target']:
            with self.subTest(bad=bad):
                # Every subcase has its own output/prefix; no previous result is overwritten.
                self.args.out=self.root/('out_'+bad);self.prefix=self.root/('clone_'+bad);self.args.prefix=self.prefix
                self.calls=[];self.bad=bad
                with self.assertRaisesRegex(ValueError,'Clone identity'):self.invoke()
                self.assertNotIn('download_fixed_wheels',self.calls)
                self.assertFalse(any('pip_check' in x or 'install' in x or 'dry_run' in x for x in self.calls))
                self.assertIn('source_after',self.calls)
                self.assertFalse(json.loads((self.args.out/'clone_identity.json').read_text())['passed'])

    def test_middle_failure_still_checks_source_after(self):
        with self.assertRaisesRegex(RuntimeError,'primary-download-failure'):self.invoke()
        self.assertIn('source_after',self.calls)
        result=json.loads((self.args.out/'completion.json').read_text())
        self.assertFalse(result['complete']);self.assertEqual(result['source_after']['status'],'unchanged')
        self.assertTrue(result['original_package_versions_unchanged'])

    def test_ending_failure_preserves_primary_error_and_marks_unknown(self):
        self.after_error=True
        with self.assertRaisesRegex(RuntimeError,'primary-download-failure'):self.invoke()
        failure=json.loads((self.args.out/'failure.json').read_text())
        self.assertIn('primary-download-failure',failure['error'])
        self.assertIn('secondary-ending-check-failure',failure['source_after']['error'])
        self.assertEqual(failure['source_after']['status'],'unknown')
        self.assertIsNone(failure['original_package_versions_unchanged'])
        self.assertFalse(json.loads((self.args.out/'completion.json').read_text())['complete'])

    def test_changed_source_is_not_reported_unchanged(self):
        self.after_changed=True
        with self.assertRaisesRegex(RuntimeError,'primary-download-failure'):self.invoke()
        result=json.loads((self.args.out/'completion.json').read_text())
        self.assertEqual(result['source_after']['status'],'changed')
        self.assertFalse(result['original_package_versions_unchanged'])

    def test_early_hold_still_records_unverified_source_after(self):
        self.assessment['clone_install_eligible']=False
        write(self.inspection/'assessment.json',self.assessment)
        with self.assertRaisesRegex(ValueError,'HOLD'):self.invoke()
        self.assertIn('source_after',self.calls)
        self.assertTrue((self.args.out/'source_after.json').is_file())
        result=json.loads((self.args.out/'completion.json').read_text())
        self.assertEqual(result['source_after']['status'],'unknown')
        self.assertIsNone(result['original_package_versions_unchanged'])

    def test_same_versions_but_changed_origin_is_not_unchanged_environment(self):
        self.after_location_changed=True
        with self.assertRaisesRegex(RuntimeError,'primary-download-failure'):self.invoke()
        result=json.loads((self.args.out/'completion.json').read_text())
        self.assertEqual(result['source_after']['status'],'changed')
        self.assertFalse(result['original_environment_snapshot_unchanged'])
        self.assertTrue(result['original_package_versions_unchanged'])

    def test_successful_work_requires_successful_ending_check(self):
        def completed(a,state):state.update(service=self.source/'bin/python',repo=self.repo,env={},baseline=self.before)
        with patch.object(clone,'perform_validation',side_effect=completed):self.invoke()
        self.assertTrue(json.loads((self.args.out/'completion.json').read_text())['complete'])

    def test_successful_work_but_failed_ending_check_is_incomplete(self):
        self.after_error=True
        def completed(a,state):state.update(service=self.source/'bin/python',repo=self.repo,env={},baseline=self.before)
        with patch.object(clone,'perform_validation',side_effect=completed):
            with self.assertRaisesRegex(RuntimeError,'Ending source verification is unknown'):self.invoke()
        result=json.loads((self.args.out/'completion.json').read_text())
        self.assertFalse(result['complete']);self.assertIsNone(result['original_package_versions_unchanged'])


class SourceChecks(unittest.TestCase):
    def test_benchmarks_use_service_repo_and_mismatches_fail(self):
        explicit=os.environ.get('R1_COMPAT_TEST_REPO')
        repo=Path(explicit).resolve() if explicit else next((p for p in ROOT.parents if (p/'vllm/v1/request.py').is_file()),None)
        if repo is None:self.skipTest('Set R1_COMPAT_TEST_REPO for real source fixtures')
        with tempfile.TemporaryDirectory(dir=ROOT) as d:
            imported=Path(d)/'imported';service=Path(d)/'service_repo'
            manifest=json.loads((ROOT/'source_baseline.json').read_text())['sources']
            for name in manifest:
                dest=(service if name.startswith('benchmarks/') else imported)/name
                dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(repo/name,dest)
            identity={'modules':{'vllm':{'file':str(imported/'vllm/__init__.py'),'version':'0.9.1'}}}
            valid=source_check(identity,service)
            self.assertTrue(valid['passed']);self.assertEqual(len(valid['checked']),len(manifest))
            missing_root=source_check(identity)
            self.assertFalse(missing_root['passed']);self.assertEqual(len(missing_root['skipped']),3)
            target=service/'benchmarks/benchmark_serving.py'
            with target.open('a',encoding='utf-8') as f:f.write('\n# synthetic benchmark mismatch\n')
            result=source_check(identity,service)
            self.assertFalse(result['passed'])
            self.assertEqual([x['file'] for x in result['mismatched']],['benchmarks/benchmark_serving.py'])
            self.assertEqual(result['sources']['benchmarks/benchmark_serving.py']['scope'],'service_repo_benchmark')


if __name__=='__main__':unittest.main(verbosity=2)
