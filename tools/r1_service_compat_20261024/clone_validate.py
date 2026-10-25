"""Optional user-run full service clone validation; never installs in source."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import traceback
import zipfile
from email.parser import Parser

from environment_probe import canonical
from inspect_service import ROOT,TARGET,classify,environment,load,probe_command,run,save,source_check


def must_run(command,out,name,cwd,env):
    rc,text=run(command,out,name,cwd,env)
    if rc:raise RuntimeError(name+' failed; stop, preserve logs, do not repair the service environment')
    return text


def validate_clone_identity(snapshot,prefix,python):
    lexical=Path(prefix).absolute();resolved=lexical.resolve()
    def require(ok,message):
        if not ok:raise ValueError('Clone identity: '+message)
    def inside(value,label):
        require(bool(value),'missing '+label)
        path=Path(value)
        require(path.resolve().is_relative_to(resolved),label+' is outside clone prefix')
        return path
    require(Path(snapshot['prefix']).resolve()==resolved,'sys.prefix mismatch')
    executable=inside(snapshot['executable'],'sys.executable')
    require(executable.absolute().is_relative_to(lexical),'sys.executable lexical path is outside clone')
    require(executable.resolve()==Path(python).resolve() and executable.is_file(),'unexpected executable')
    pip=snapshot.get('pip_identity',{})
    require(inside(pip.get('file'),'imported pip').is_file(),'pip module file is missing')
    selected=snapshot.get('selected',{}).get('pip',{})
    inside(selected.get('distribution_location'),'pip distribution')
    inside(selected.get('module_spec_origin'),'pip module spec')
    require(bool(pip.get('version')) and pip['version']==selected.get('version'),'pip metadata/import version mismatch')
    require(Path(pip.get('target_prefix','')).resolve()==resolved,'pip target prefix mismatch')
    scheme=pip.get('scheme',{})
    require(set(scheme)=={'purelib','platlib','scripts','data','headers'},'pip target scheme incomplete')
    for key,value in scheme.items():inside(value,'pip target '+key)
    env=snapshot.get('environment',{})
    require(not env.get('PIP_TARGET') and not env.get('PIP_PREFIX'),'implicit pip destination configured')
    require(env.get('PIP_USER')=='0','PIP_USER is not disabled')
    require(str(env.get('PIP_CONFIG_FILE','')).lower()==os.devnull.lower(),'pip config isolation missing')
    return dict(passed=True,sys_prefix=snapshot['prefix'],sys_executable=snapshot['executable'],
                pip=pip,expected_prefix=str(lexical))


def perform_validation(a,state):
    assessment=load(a.inspection/'assessment.json')
    state.update(service=Path(assessment['service_python']),repo=Path(assessment['service_repo']),env=environment(True))
    if not assessment['clone_install_eligible']:raise ValueError('Inspection has HOLD reasons; review assessment before any clone installation')
    source=load(a.inspection/'isolated_environment.json')
    state['baseline']=source
    state['baseline_origin']='inspection/isolated_environment.json'
    source_prefix=Path(source['prefix']).resolve();prefix=a.prefix.absolute()
    if prefix.exists() or prefix.is_symlink():raise ValueError('Clone prefix already exists; choose a new name, do not overwrite')
    if not prefix.parent.is_dir() or not prefix.resolve().is_relative_to(Path.home().resolve()):
        raise ValueError('Choose an existing writable parent inside your HOME')
    if prefix.resolve()==source_prefix:raise ValueError('Source and clone must differ')
    if a.out.absolute().is_relative_to(prefix) or prefix.is_relative_to(a.out.absolute()):
        raise ValueError('Keep report directory and large environment prefix separate')
    if not math.isfinite(a.max_copy_gib) or a.max_copy_gib<=0:
        raise ValueError('Provide a finite positive user-approved maximum clone size')
    env=environment(True);env['PIP_USER']='0';env.pop('PIP_TARGET',None);env.pop('PIP_PREFIX',None)
    env['PIP_CONFIG_FILE']=os.devnull  # clone-only: disable implicit global --target/--user destinations
    repo=Path(assessment['service_repo']);service=Path(assessment['service_python'])
    state.update(env=env,repo=repo,service=service)
    size=int(must_run(['du','-s','--apparent-size','-B1',source_prefix],a.out,'source_size',repo,env).split()[0])
    free=shutil.disk_usage(prefix.parent).free
    save(a.out/'copy_plan.json',dict(source_prefix=str(source_prefix),clone_prefix=str(prefix),
         estimated_apparent_bytes=size,free_bytes=free,user_max_gib=a.max_copy_gib,
         argv=sys.argv,original_service_install=False))
    if size>a.max_copy_gib*1024**3 or free<size*1.2+2*1024**3:
        raise ValueError('Clone exceeds declared size or free-space reserve; stop, no alternative environment substituted')
    must_run(probe_command(service,'environment_probe.py','--out',a.out/'source_before.json'),a.out,'source_before',repo,env)
    before=load(a.out/'source_before.json');state['baseline']=before;state['baseline_origin']='source_before.json'
    if before['active']!=source['active']:
        raise ValueError('Source packages changed after inspection; repeat read-only inspection first')
    source_report=assessment['imported_sources']['isolated']
    expected_sources=set(load(ROOT/'source_baseline.json')['sources'])
    if (source_report.get('schema_version')!=2 or set(source_report.get('checked',[]))!=expected_sources
            or set(source_report.get('sources',{}))!=expected_sources
            or source_report.get('skipped') or source_report.get('mismatched') or not source_report.get('passed')):
        raise ValueError('Repeat inspection with complete imported-vLLM and service-repo benchmark coverage')
    for record in source_report['sources'].values():
        path=Path(record['path']);digest=hashlib.sha256(path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
        if digest!=record['lf_sha256']:raise ValueError('Source code changed after audit: '+str(path))
    # Copy rather than hard-link source env package files. Never runs source pip install.
    must_run([a.conda,'create','--offline','--yes','--copy','--clone',source_prefix,'--prefix',prefix],
             a.out,'conda_clone',repo,env)
    python=prefix/'bin/python'
    must_run(probe_command(python,'environment_probe.py','--out',a.out/'clone_before.json','--repo',repo,
             '--pip-target-prefix',prefix),
             a.out,'clone_before',repo,env)
    clone_before=load(a.out/'clone_before.json')
    try:identity=validate_clone_identity(clone_before,prefix,python)
    except Exception as e:
        save(a.out/'clone_identity.json',dict(passed=False,error=repr(e)))
        raise
    save(a.out/'clone_identity.json',identity)
    if clone_before['active']!=source['active']:
        raise ValueError('Clone package versions differ before predictor installation; stop')
    must_run([python,'-m','pip','check'],a.out,'clone_pip_check_before',repo,env)
    wheels=a.out/'wheelhouse';wheels.mkdir()
    must_run([python,'-m','pip','download','--only-binary=:all:','--dest',wheels,
              '-r',ROOT/'predictor_requirements.txt','--disable-pip-version-check','--no-input'],
             a.out,'download_fixed_wheels',repo,env)
    wheel_info=[]
    audited_archives={canonical(x['metadata']['name']):x.get('download_info',{}).get('archive_info',{}).get('hashes',{}).get('sha256')
                      for x in load(a.inspection/'isolated_pip_plan.json').get('install',[])}
    for wheel in sorted(wheels.iterdir()):
        if wheel.suffix!='.whl':raise ValueError('Unexpected non-wheel download')
        with zipfile.ZipFile(wheel) as z:
            names=[n for n in z.namelist() if n.endswith('.dist-info/METADATA')]
            if len(names)!=1:raise ValueError('Invalid wheel metadata')
            meta=Parser().parsestr(z.read(names[0]).decode('utf-8'))
        name=canonical(meta['Name']);version=meta['Version']
        if TARGET.get(name)!=version:raise ValueError('Unexpected wheel package/version: '+name)
        digest=hashlib.sha256(wheel.read_bytes()).hexdigest()
        if audited_archives.get(name) and digest!=audited_archives[name]:
            raise ValueError('Downloaded wheel differs from the audited installation archive: '+name)
        wheel_info.append(dict(name=name,version=version,file=wheel.name,sha256=digest,
                               audited_plan_sha256=audited_archives.get(name)))
    if {x['name'] for x in wheel_info}!=set(TARGET):raise ValueError('Incomplete fixed wheel set')
    save(a.out/'wheel_sha256.json',wheel_info)
    plan=a.out/'clone_pip_plan.json'
    base=[python,'-m','pip','install','--no-user','--prefix',prefix,'--no-index','--find-links',wheels,
          '-r',ROOT/'predictor_requirements.txt','--disable-pip-version-check','--no-input']
    must_run([*base,'--dry-run','--report',plan],a.out,'clone_dry_run',repo,env)
    decision=classify(load(a.out/'clone_before.json'),load(plan))
    save(a.out/'clone_plan_assessment.json',decision)
    ordered=lambda changes:sorted(changes,key=lambda x:x['name'])
    if not decision['additive_plan_only'] or ordered(decision['changes'])!=ordered(assessment['classifications']['isolated']['changes']):
        raise ValueError('Clone plan differs from audited additive service plan; no install performed')
    installed=a.out/'clone_installed.json'
    must_run([*base,'--report',installed],a.out,'clone_install_fixed',repo,env)
    expected={(x['name'],x['after']) for x in decision['changes']}
    actual={(canonical(x['metadata']['name']),x['metadata']['version']) for x in load(installed).get('install',[])}
    if actual!=expected:raise ValueError('Installed package set differs; preserve clone for review')
    must_run([python,'-m','pip','check'],a.out,'clone_pip_check_after',repo,env)
    must_run(probe_command(python,'environment_probe.py','--out',a.out/'clone_after.json','--repo',repo),
             a.out,'clone_after',repo,env)
    after=load(a.out/'clone_after.json')['active']
    if any(after.get(k)!=v for k,v in source['active'].items()):raise ValueError('A pre-existing package was changed in clone')
    if {k for k in after if k not in source['active']}!={x['name'] for x in decision['changes']}:
        raise ValueError('Unexpected new package in clone')
    check_env=environment(True,probe=True)
    for order in ['vllm-first','sklearn-first']:
        must_run(probe_command(python,'cpu_checks.py','--mode','imports','--order',order,
            '--expected-prefix',prefix,'--out',a.out/(order+'.json')),a.out,order,repo,check_env)
        check=source_check(load(a.out/(order+'.json')),repo)
        save(a.out/(order+'_source.json'),check)
        if not load(a.out/(order+'.json')).get('passed') or not check['passed']:
            raise ValueError('Clone CPU imports or complete source verification did not pass')
    must_run(probe_command(python,'cpu_checks.py','--mode','predict','--expected-prefix',prefix,
             '--migration-dir',a.migration_dir,'--out',a.out/'model_validation.json'),
             a.out,'model_validation',repo,check_env)
    validation=load(a.out/'model_validation.json')
    if not validation.get('passed') or not validation.get('model_validation',{}).get('passed'):
        raise ValueError('Clone CPU model validation did not pass')


def record_source_after(a,state):
    status=dict(status='unknown',unchanged=None,attempted=False,
                package_versions_unchanged=None,
                comparison_fields=['active','prefix','executable','selected'],
                compared_to=state.get('baseline_origin'))
    try:
        if not all(k in state for k in ['service','repo','env']):
            raise RuntimeError('Source interpreter/repository unavailable; ending verification unverified')
        status['attempted']=True
        must_run(probe_command(state['service'],'environment_probe.py','--out',a.out/'source_after.json'),
                 a.out,'source_after',state['repo'],state['env'])
        after=load(a.out/'source_after.json');before=state.get('baseline')
        if before is None:raise RuntimeError('No valid before inventory; source_after recorded but comparison unverified')
        changed=[k for k in status['comparison_fields'] if after[k]!=before[k]]
        status.update(status='changed' if changed else 'unchanged',unchanged=not changed,changed_fields=changed,
                      package_versions_unchanged=after['active']==before['active'])
    except BaseException:
        status.update(status='unknown',unchanged=None,package_versions_unchanged=None,error=traceback.format_exc())
    try:save(a.out/'source_after_status.json',status)
    except Exception as e:
        status.update(status='unknown',unchanged=None,package_versions_unchanged=None,record_error=repr(e))
        print('Could not save source_after status: '+repr(e),file=sys.stderr)
    return status


def validate(a):
    # An existing result directory is never overwritten, even on failure.
    a.out.mkdir(parents=True,exist_ok=False)
    state={};primary=None;primary_tb=None;checks_complete=False
    try:
        perform_validation(a,state)
        checks_complete=True
    except BaseException as e:
        primary=e;primary_tb=e.__traceback__
        primary_traceback=traceback.format_exc()
    try:after=record_source_after(a,state)
    except BaseException:
        after=dict(status='unknown',unchanged=None,attempted=None,error=traceback.format_exc())
    if primary is None and after['status']!='unchanged':
        primary=RuntimeError('Ending source verification is '+after['status']+'; compatibility run is not complete')
        primary_traceback=str(primary)
    version_unchanged=after.get('package_versions_unchanged') if after['status']!='unknown' else None
    common=dict(clone_prefix=str(a.prefix),checks_complete=checks_complete,source_after=after,
                original_environment_snapshot_unchanged=after['unchanged'],
                original_package_versions_unchanged=version_unchanged)
    if primary is not None:
        for name,data in [
            ('failure.json',dict(common,complete=False,error=repr(primary),primary_traceback=primary_traceback,preserve_all_files=True)),
            ('completion.json',dict(common,complete=False))]:
            try:save(a.out/name,data)
            except Exception as e:print('Could not save '+name+': '+repr(e),file=sys.stderr)
        raise primary.with_traceback(primary_tb)
    save(a.out/'completion.json',dict(common,complete=True,full_service_clone=True,
         two_import_orders_passed=True,model_validation_n=1440,no_llm_engine_started=True,
         gpu_execution_verified=False,limitation='CPU imports/features only; not GPU execution compatibility'))
    print('Full service clone CPU checks passed:',a.out)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inspection',type=Path,required=True);p.add_argument('--conda',type=Path,required=True)
    p.add_argument('--prefix',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--migration-dir',type=Path,required=True);p.add_argument('--max-copy-gib',type=float,required=True)
    validate(p.parse_args())


if __name__=='__main__':main()
