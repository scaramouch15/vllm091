"""Read-only service audit and pip installation-plan classification; no install."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from environment_probe import canonical,clean

ROOT=Path(__file__).resolve().parent
TARGET={'numpy':'2.2.6','scipy':'1.15.3','scikit-learn':'1.7.2','joblib':'1.5.3','threadpoolctl':'3.6.0'}
HEAD='09c35e97b0605bd437c725562ca05afaacdb82e1'


def load(p):return json.loads(p.read_text(encoding='utf-8'))


def save(p,value):
    with p.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)


def environment(isolated=False,probe=False):
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='')
    if isolated:
        env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None);env['PYTHONNOUSERSITE']='1'
    if probe:
        for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[key]='1'
        env['PYTORCH_NVML_BASED_CUDA_CHECK']='1'
    return env


def run(command,out,name,cwd,env):
    command=list(map(str,command));started=dt.datetime.now(dt.timezone.utc).isoformat()
    p=subprocess.run(command,cwd=cwd,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                     text=True,encoding='utf-8',errors='replace')
    text=clean(p.stdout)
    (out/(name+'.log')).write_text(text,encoding='utf-8')
    with (out/'commands.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps(dict(command=clean(command),cwd=str(cwd),start_utc=started,
            end_utc=dt.datetime.now(dt.timezone.utc).isoformat(),exit_code=p.returncode),allow_nan=False)+'\n')
    print(name+': exit='+str(p.returncode),flush=True)
    if p.returncode:print(text[-1400:],flush=True)
    return p.returncode,text


def probe_command(python,script,*args):
    # Preserve cwd import lookup as with `python -m ...`, while executing a file.
    return [python,'-B','-c',"import runpy,sys; p=sys.argv.pop(1); runpy.run_path(p,run_name='__main__')",ROOT/script,*args]


def classify(snapshot,plan,pip_ok=True):
    reasons=[];changes=[]
    active=snapshot['active']
    if snapshot['python']!='3.10.22':reasons.append('Python is not 3.10.22; no Python replacement is authorized')
    for name in ['numpy','scipy']:
        if active.get(name)!=TARGET[name]:reasons.append(name+' does not already match the fixed service baseline')
    if not pip_ok:reasons.append('Existing pip check conflicts/failure must be reviewed first')
    if plan is None or not isinstance(plan.get('install'),list):
        reasons.append('pip dry-run/report unavailable, failed or unsupported; metadata-only, not install-approved')
    else:
        for item in plan.get('install',[]):
            name=canonical(item['metadata']['name']);version=item['metadata']['version'];old=active.get(name)
            change=dict(name=name,before=old,after=version,action='add' if old is None else 'replace')
            changes.append(change)
            if name not in TARGET or version!=TARGET.get(name):reasons.append('Unexpected planned package/version: '+name+'=='+version)
            if old is not None:reasons.append('Replacement/reinstallation of an existing package: '+name)
        projected=dict(active)
        projected.update({x['name']:x['after'] for x in changes})
        for name,version in TARGET.items():
            if projected.get(name)!=version:reasons.append('Fixed target unresolved: '+name)
    return dict(target=TARGET,changes=changes,hold_reasons=reasons,additive_plan_only=not reasons)


def source_check(identity,service_repo=None):
    expected=load(ROOT/'source_baseline.json')
    module=identity.get('modules',{}).get('vllm',{})
    root=Path(module['file']).resolve().parent if module.get('file') else None
    repo=Path(service_repo).resolve() if service_repo is not None else None
    records={};checked=[];skipped=[];mismatched=[]
    for name,wanted in expected['sources'].items():
        if name.startswith('vllm/'):
            base=root;relative=Path(name).relative_to('vllm');scope='imported_vllm'
        elif name.startswith('benchmarks/'):
            base=repo;relative=Path(name);scope='service_repo_benchmark'
        else:base=None;relative=Path(name);scope='unsupported_manifest_scope'
        record=dict(scope=scope,passed=False,expected_lf_sha256=wanted['lf_sha256'])
        records[name]=record
        if base is None:
            record['status']='skipped';record['reason']='source_root_unavailable'
            skipped.append(dict(file=name,reason=record['reason'],scope=scope));continue
        path=(base/relative).resolve();record['path']=str(path)
        if not path.is_relative_to(base):
            record.update(status='skipped',reason='path_outside_source_root')
            skipped.append(dict(file=name,reason=record['reason'],scope=scope));continue
        try:data=path.read_bytes()
        except OSError as e:
            record.update(status='mismatched',reason='source_unreadable',error=repr(e))
            mismatched.append(dict(file=name,reason=record['reason'],scope=scope));continue
        digest=hashlib.sha256(data.replace(b'\r\n',b'\n')).hexdigest()
        checked.append(name)
        record.update(sha256=hashlib.sha256(data).hexdigest(),lf_sha256=digest,
                      passed=digest==wanted['lf_sha256'],status='checked')
        if not record['passed']:
            record.update(status='mismatched',reason='hash_mismatch')
            mismatched.append(dict(file=name,reason='hash_mismatch',scope=scope))
    version=str(module.get('version','')).split('+')[0]
    return dict(schema_version=2,vllm_file=module.get('file'),vllm_version=module.get('version'),
                version_matches=version=='0.9.1',
                service_repo=str(repo) if repo else None,sources=records,checked=checked,skipped=skipped,mismatched=mismatched,
                expected_files=len(expected['sources']),
                passed=bool(records) and len(checked)==len(records) and not skipped and not mismatched and version=='0.9.1')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--service-python',required=True,type=Path);p.add_argument('--service-repo',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path);p.add_argument('--predictor-python',type=Path)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    save(a.out/'invocation.json',dict(argv=sys.argv,orchestrator=sys.executable,
         original_service_install_performed=False,probe_threads='1 only in disposable import subprocesses',
         original_environment=clean({k:os.environ.get(k) for k in ['PYTHONPATH','PYTHONHOME','PYTHONNOUSERSITE',
             'CUDA_VISIBLE_DEVICES','VLLM_USE_V1','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS','PIP_USER','PIP_TARGET','PIP_PREFIX']})))
    snapshots={};classifications={};identities={};sources={}
    for mode in ['native','isolated']:
        env=environment(mode=='isolated')
        rc,_=run(probe_command(a.service_python,'environment_probe.py','--out',a.out/(mode+'_environment.json'),
                 '--repo',a.service_repo),a.out,mode+'_metadata',a.service_repo,env)
        if rc:continue
        snap=load(a.out/(mode+'_environment.json'));snapshots[mode]=snap
        pip_rc,_=run([a.service_python,'-m','pip','check'],a.out,mode+'_pip_check',a.service_repo,env)
        run([a.service_python,'-m','pip','freeze','--all'],a.out,mode+'_pip_freeze',a.service_repo,env)
        run([a.service_python,'-m','pip','config','debug'],a.out,mode+'_pip_config',a.service_repo,env)
        _,help_text=run([a.service_python,'-m','pip','install','--help'],a.out,mode+'_pip_help',a.service_repo,env)
        plan=None
        if '--dry-run' in help_text and '--report' in help_text:
            report=a.out/(mode+'_pip_plan.json')
            rc,_=run([a.service_python,'-m','pip','install','--dry-run','--report',report,
                       '-r',ROOT/'predictor_requirements.txt','--disable-pip-version-check','--no-input'],
                      a.out,mode+'_pip_dry_run',a.service_repo,env)
            if report.exists():
                # Remove credentials/query strings from newly generated provenance only.
                plan=load(report)
                report.write_text(json.dumps(clean(plan),indent=2)+'\n',encoding='utf-8')
            if rc:plan=None
        classifications[mode]=classify(snap,plan,pip_rc==0)
        path=a.out/(mode+'_import_identity.json')
        rc,_=run(probe_command(a.service_python,'cpu_checks.py','--mode','identity','--out',path),
                 a.out,mode+'_import_identity',a.service_repo,environment(mode=='isolated',probe=True))
        if path.exists():
            identities[mode]=load(path);sources[mode]=source_check(identities[mode],a.service_repo)
    if a.predictor_python:
        run(probe_command(a.predictor_python,'environment_probe.py','--out',a.out/'predictor_B_metadata.json'),
            a.out,'predictor_B_metadata',a.service_repo,environment(True))
    holds=[]
    if set(snapshots)!= {'native','isolated'}:holds.append('Native/isolated metadata inspection did not both complete')
    else:
        for mode in snapshots:
            holds.extend(mode+': '+x for x in classifications[mode]['hold_reasons'])
            repo=snapshots[mode].get('repo',{})
            if repo.get('head',{}).get('stdout','').strip()!=HEAD:holds.append(mode+': server HEAD differs; review source changes')
            if not identities.get(mode,{}).get('passed'):holds.append(mode+': vLLM/torch CPU import failed')
            optional=identities.get(mode,{}).get('optional_import_errors',{})
            for dist,module in [('scikit-learn','sklearn'),('joblib','joblib'),('threadpoolctl','threadpoolctl')]:
                if snapshots[mode]['selected'][dist]['version'] is not None and module in optional:
                    holds.append(mode+': existing '+dist+' cannot import; review pre-existing failure')
            for name in TARGET:
                selected=snapshots[mode]['selected'][name]
                if selected['version'] is None and selected['module_spec_origin'] is not None:
                    holds.append(mode+': '+name+' module exists without distribution metadata; no blind overlay')
            if not sources.get(mode,{}).get('passed'):holds.append(mode+': imported vLLM source/version differs from local audit baseline')
            env=snapshots[mode]['environment']
            # PIP_TARGET/PREFIX/USER are checked from the orchestrator environment below.
        for name in TARGET:
            if snapshots['native']['selected'][name]!=snapshots['isolated']['selected'][name]:
                holds.append('Import/metadata source differs after isolation: '+name)
        for name in ['vllm','torch']:
            if identities.get('native',{}).get('modules',{}).get(name)!=identities.get('isolated',{}).get('modules',{}).get(name):
                holds.append('Core module import differs after isolation: '+name)
    for name in ['PIP_TARGET','PIP_PREFIX','PIP_USER']:
        if os.environ.get(name):holds.append(name+' is set; review pip destination configuration first')
    save(a.out/'assessment.json',dict(service_python=str(a.service_python),service_repo=str(a.service_repo),
        classifications=classifications,imported_sources=sources,hold_reasons=holds,
        clone_install_eligible=not holds,original_environment_changed=False,
        status='Read-only inspection complete; clone not created; no install executed'))
    print('Assessment:',a.out/'assessment.json')
    print('Clone additive-install eligible:',not holds)


if __name__=='__main__':main()
