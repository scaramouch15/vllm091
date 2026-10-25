"""Optional user-run full service clone validation; never installs in source."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import zipfile
from email.parser import Parser

from environment_probe import canonical
from inspect_service import ROOT,TARGET,classify,environment,load,probe_command,run,save,source_check


def must_run(command,out,name,cwd,env):
    rc,text=run(command,out,name,cwd,env)
    if rc:raise RuntimeError(name+' failed; stop, preserve logs, do not repair the service environment')
    return text


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inspection',type=Path,required=True);p.add_argument('--conda',type=Path,required=True)
    p.add_argument('--prefix',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--migration-dir',type=Path,required=True);p.add_argument('--max-copy-gib',type=float,required=True)
    a=p.parse_args()
    assessment=load(a.inspection/'assessment.json')
    if not assessment['clone_install_eligible']:raise ValueError('Inspection has HOLD reasons; review assessment before any clone installation')
    source=load(a.inspection/'isolated_environment.json')
    source_prefix=Path(source['prefix']).resolve();prefix=a.prefix.absolute()
    if prefix.exists() or prefix.is_symlink():raise ValueError('Clone prefix already exists; choose a new name, do not overwrite')
    if not prefix.parent.is_dir() or not prefix.resolve().is_relative_to(Path.home().resolve()):
        raise ValueError('Choose an existing writable parent inside your HOME')
    if prefix.resolve()==source_prefix:raise ValueError('Source and clone must differ')
    if a.out.absolute().is_relative_to(prefix) or prefix.is_relative_to(a.out.absolute()):
        raise ValueError('Keep report directory and large environment prefix separate')
    if not math.isfinite(a.max_copy_gib) or a.max_copy_gib<=0:
        raise ValueError('Provide a finite positive user-approved maximum clone size')
    a.out.mkdir(parents=True,exist_ok=False)
    env=environment(True);env['PIP_USER']='0';env.pop('PIP_TARGET',None);env.pop('PIP_PREFIX',None)
    env['PIP_CONFIG_FILE']=os.devnull  # clone-only: disable implicit global --target/--user destinations
    repo=Path(assessment['service_repo']);service=Path(assessment['service_python'])
    try:
        size=int(must_run(['du','-s','--apparent-size','-B1',source_prefix],a.out,'source_size',repo,env).split()[0])
        free=shutil.disk_usage(prefix.parent).free
        save(a.out/'copy_plan.json',dict(source_prefix=str(source_prefix),clone_prefix=str(prefix),
             estimated_apparent_bytes=size,free_bytes=free,user_max_gib=a.max_copy_gib,
             argv=sys.argv,original_service_install=False))
        if size>a.max_copy_gib*1024**3 or free<size*1.2+2*1024**3:
            raise ValueError('Clone exceeds declared size or free-space reserve; stop, no alternative environment substituted')
        must_run(probe_command(service,'environment_probe.py','--out',a.out/'source_before.json'),a.out,'source_before',repo,env)
        if load(a.out/'source_before.json')['active']!=source['active']:
            raise ValueError('Source packages changed after inspection; repeat read-only inspection first')
        for record in assessment['imported_sources']['isolated']['sources'].values():
            path=Path(record['path']);digest=hashlib.sha256(path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
            if digest!=record['lf_sha256']:raise ValueError('Source code changed after audit: '+str(path))
        # Copy rather than hard-link source env package files. Never runs source pip install.
        must_run([a.conda,'create','--offline','--yes','--copy','--clone',source_prefix,'--prefix',prefix],
                 a.out,'conda_clone',repo,env)
        python=prefix/'bin/python'
        must_run(probe_command(python,'environment_probe.py','--out',a.out/'clone_before.json','--repo',repo),
                 a.out,'clone_before',repo,env)
        if load(a.out/'clone_before.json')['active']!=source['active']:
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
            check=source_check(load(a.out/(order+'.json')))
            save(a.out/(order+'_source.json'),check)
            if not check['passed']:raise ValueError('Clone imported a different vLLM source/version')
        must_run(probe_command(python,'cpu_checks.py','--mode','predict','--expected-prefix',prefix,
                 '--migration-dir',a.migration_dir,'--out',a.out/'model_validation.json'),
                 a.out,'model_validation',repo,check_env)
        must_run(probe_command(service,'environment_probe.py','--out',a.out/'source_after.json'),a.out,'source_after',repo,env)
        if load(a.out/'source_after.json')['active']!=source['active']:raise ValueError('Original package inventory changed during validation')
        save(a.out/'completion.json',dict(complete=True,clone_prefix=str(prefix),full_service_clone=True,
             two_import_orders_passed=True,model_validation_n=1440,original_package_versions_unchanged=True,
             no_llm_engine_started=True,gpu_execution_verified=False,
             limitation='CPU import/model-feature validation only; not proof of GPU execution compatibility'))
    except BaseException as e:
        save(a.out/'failure.json',dict(complete=False,error=repr(e),clone_prefix=str(prefix),preserve_all_files=True))
        raise
    print('Full service clone CPU checks passed:',a.out)


if __name__=='__main__':main()
