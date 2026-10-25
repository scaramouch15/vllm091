"""Read-only metadata snapshot. No torch/vLLM/numerical imports."""
import argparse
import importlib.metadata as md
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import site
import subprocess
import sys
from urllib.parse import urlsplit, urlunsplit

MODULES={'vllm':'vllm','torch':'torch','numpy':'numpy','scipy':'scipy',
         'scikit-learn':'sklearn','joblib':'joblib','threadpoolctl':'threadpoolctl','pip':'pip'}


def canonical(name):
    return re.sub(r'[-_.]+','-',name).lower()


def clean(value):
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,list):return [clean(v) for v in value]
    if not isinstance(value,str):return value
    def url(match):
        try:
            p=urlsplit(match.group());host=p.hostname or ''
            if p.port:host+=':'+str(p.port)
            return urlunsplit((p.scheme,host,p.path,'',''))
        except ValueError:return '[redacted-url]'
    return re.sub(r'https?://[^\s\"\'<>]+',url,value)


def snapshot(repo=None):
    packages=[];active={}
    for dist in md.distributions():
        name=dist.metadata.get('Name','')
        try:direct=json.loads(dist.read_text('direct_url.json') or 'null')
        except (ValueError,OSError):direct='unreadable'
        packages.append(dict(name=name,version=dist.version,location=str(dist.locate_file('')),
            metadata_path=str(getattr(dist,'_path','')),installer=dist.read_text('INSTALLER'),
            direct_url=clean(direct),requires=dist.requires or []))
        active.setdefault(canonical(name),dist.version)
    selected={}
    for name,module in MODULES.items():
        try:
            d=md.distribution(name);version=d.version;location=str(d.locate_file(''))
        except md.PackageNotFoundError:version=location=None
        try:s=importlib.util.find_spec(module);origin=s.origin if s else None
        except Exception as e:origin='ERROR: '+repr(e)
        selected[name]=dict(version=version,distribution_location=location,module_spec_origin=origin)
    keys=['PATH','PYTHONPATH','PYTHONHOME','PYTHONNOUSERSITE','LD_LIBRARY_PATH','LD_PRELOAD',
          'CUDA_VISIBLE_DEVICES','VLLM_USE_V1','VLLM_WORKER_MULTIPROC_METHOD',
          'OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS',
          'PIP_INDEX_URL','PIP_EXTRA_INDEX_URL','PIP_CONFIG_FILE','PIP_USER','PIP_PREFIX','PIP_TARGET','CONDA_PREFIX']
    result=dict(executable=sys.executable,prefix=sys.prefix,base_prefix=sys.base_prefix,
        python=platform.python_version(),hostname=platform.node(),cwd=os.getcwd(),sys_path=sys.path,
        user_site=site.getusersitepackages(),enable_user_site=site.ENABLE_USER_SITE,
        environment=clean({k:os.environ.get(k) for k in keys}),selected=selected,active=active,
        packages=sorted(packages,key=lambda x:(x['name'].lower(),x['location'])))
    conda=[]
    for record in sorted((Path(sys.prefix)/'conda-meta').glob('*.json')):
        try:
            data=json.loads(record.read_text())
            conda.append(clean({k:data.get(k) for k in ['name','version','build','channel','url','sha256','md5']}))
        except (OSError,ValueError):conda.append({'unreadable':str(record)})
    result['conda_packages']=conda
    if repo:
        result['repo']={}
        for key,args in [('head',['rev-parse','HEAD']),('branch',['branch','--show-current']),
                         ('status',['status','--short']),('diff_stat',['diff','HEAD','--stat'])]:
            p=subprocess.run(['git','-C',repo,*args],capture_output=True,text=True)
            result['repo'][key]=dict(exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True,type=Path);p.add_argument('--repo')
    a=p.parse_args()
    with a.out.open('x',encoding='utf-8') as f:json.dump(snapshot(a.repo),f,ensure_ascii=False,indent=2,allow_nan=False)
