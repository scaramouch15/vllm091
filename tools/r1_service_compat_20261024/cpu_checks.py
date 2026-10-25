"""Disposable CPU import/coexistence checks. Never starts an engine/model."""
import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
import traceback


class CheckError(RuntimeError):
    """A validation gate failed (also enforced under Python -O)."""


def require(ok,message):
    if not ok:raise CheckError(message)


def validate_interpreter(prefix):
    lexical_prefix=Path(prefix).absolute();prefix=lexical_prefix.resolve()
    require(Path(sys.prefix).resolve()==prefix,'Wrong clone sys.prefix')
    executable=Path(sys.executable).absolute()
    require(executable.is_relative_to(lexical_prefix) and executable.resolve().is_relative_to(prefix),
            'Clone sys.executable is outside expected prefix')


def validate_samples(rows,previous,ids):
    require(len(rows)==1440 and len(previous)==1440,'Expected 1440 validation/reference records')
    actual,reference=ids(rows),ids(previous)
    require(len(actual)==1440 and len(reference)==1440 and actual==reference,'Prediction identity/order mismatch')
    require(len(set(actual))==1440,'Duplicate validation request identity')


def validate_predictions(checks):
    require(set(checks)=={'length','time','length_raw','time_raw'},'Missing prediction comparison stage')
    for name,check in checks.items():
        require(check['failed_elements']==0 and check['failed_records']==0,'Strict prediction comparison failed: '+name)
        for field in ['max_absolute_error','max_relative_error']:
            value=check[field]
            require(isinstance(value,(int,float)) and math.isfinite(value) and value>=0,
                    'Nonfinite/invalid comparison error: '+name+'/'+field)


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def check(a,result):
    if a.mode!='identity':require(a.expected_prefix is not None,'--expected-prefix required for clone validation')
    if a.expected_prefix:validate_interpreter(a.expected_prefix)
    names=['torch','vllm','numpy','scipy','sklearn','joblib','threadpoolctl']
    if a.order=='sklearn-first':names=['numpy','scipy','sklearn','joblib','threadpoolctl','torch','vllm']
    result['modules']={};result['optional_import_errors']={}
    for name in names:
        try:module=importlib.import_module(name)
        except Exception as e:
            if a.mode=='identity' and name in ['sklearn','joblib','threadpoolctl']:
                result['optional_import_errors'][name]=repr(e);continue
            raise
        result['modules'][name]=dict(file=getattr(module,'__file__',None),version=getattr(module,'__version__',None))
        if name=='torch':
            result['cuda_initialized']=bool(module.cuda.is_initialized())
            require(not result['cuda_initialized'],'CUDA initialized while importing torch')
            # Fail before lazy CUDA initialization, including indirect calls by vLLM.
            def forbid_cuda_init(*args,**kwargs):
                raise RuntimeError('CUDA initialization is forbidden in this CPU compatibility check')
            module.cuda._lazy_init=forbid_cuda_init
            for entry in ['_cuda_init','_cuda_getDeviceCount']:
                if hasattr(module._C,entry):setattr(module._C,entry,forbid_cuda_init)
    import torch
    result['cuda_initialized']=bool(torch.cuda.is_initialized())
    require(not result['cuda_initialized'],'CUDA initialized during imports')
    result['cuda_initialized']=False
    result['lazy_cuda_initialization_guard']=True
    result['cuda_driver_fallback_guard']=True
    result['torch_cpu_threads']=dict(intraop=torch.get_num_threads(),interop=torch.get_num_interop_threads())
    try:
        import threadpoolctl
        result['threadpools_before']=threadpoolctl.threadpool_info()
    except ImportError:pass
    if a.expected_prefix:
        prefix=a.expected_prefix.resolve()
        for name,info in result['modules'].items():
            if name!='vllm':
                require(bool(info['file']) and Path(info['file']).resolve().is_relative_to(prefix),name+' outside clone')
    if a.mode!='identity':
        expected={'numpy':'2.2.6','scipy':'1.15.3','sklearn':'1.7.2','joblib':'1.5.3','threadpoolctl':'3.6.0'}
        import platform
        require(platform.python_version()=='3.10.22','Python version is not 3.10.22')
        for name,version in expected.items():require(result['modules'][name]['version']==version,'Wrong package version: '+name)
    if a.mode=='predict':
        require(a.migration_dir is not None,'--migration-dir required')
        root=a.migration_dir.resolve()
        manifest=json.loads((root/'files_sha256.json').read_text())
        require(isinstance(manifest,dict) and bool(manifest),'Empty/invalid migration manifest')
        for name,digest in manifest.items():
            path=(root/name).resolve();require(path.is_relative_to(root),'Manifest path outside migration root: '+name)
            require(sha(path)==digest,'File checksum mismatch: '+name)
        sys.path.insert(0,str(root))
        import numpy as np
        import joblib
        from common import ids,read_rows,reference_modules,check_split
        from train_cpu import stages,comparison
        data=root/'data/validation.jsonl';source=root/'config_B_new310/results';model=source/'models/tfidf.joblib'
        require(sha(model)=='23aff057712e7bec05db91d470af39a1b34eabddbf5852914710e1b8e4b96e1f','Wrong B model SHA256')
        require(sha(data)=='1b7fe1e7e8036bc7f7b943866d1a2a28ace2dc7bd6e5875e44f5ef6e2e6f5d9f','Wrong validation SHA256')
        rows=read_rows(data);previous=read_rows(source/'tfidf_validation.jsonl')
        validate_samples(rows,previous,ids)
        check_split(rows,'validation',1440,360)
        base,ref=reference_modules()
        # Only this disposable verification process uses a process-wide limit.
        with threadpoolctl.threadpool_limits(limits=1):
            result['threadpools_during_validation']=threadpoolctl.threadpool_info()
            require(result['threadpools_during_validation'] and all(x['num_threads']==1 for x in result['threadpools_during_validation']),
                    'Actual validation threadpools are not single-threaded')
            bundle=joblib.load(model)
            actual=stages(bundle,rows,base,ref)
            columns={'length':'predicted_length_quantiles','time':'predicted_time_quantiles',
                     'length_raw':'raw_length_quantiles','time_raw':'raw_time_quantiles'}
            checks={k:comparison(actual[k],np.array([x[v] for x in previous])) for k,v in columns.items()}
            result['model_validation']=dict(n=1440,identity_order_verified=True,rtol=1e-10,atol=1e-10,
                model_sha256=sha(model),data_sha256=sha(data),stages=checks,
                passed=False)
            validate_predictions(checks)
        result['threadpools_after_validation']=threadpoolctl.threadpool_info()
        result['cuda_initialized']=bool(torch.cuda.is_initialized())
        require(not result['cuda_initialized'],'CUDA initialized during CPU validation')
        result['model_validation']['passed']=True


def execute(a):
    result=dict(executable=sys.executable,prefix=sys.prefix,mode=a.mode,order=a.order,passed=False)
    try:
        check(a,result)
    except BaseException:
        result['passed']=False;result['error']=traceback.format_exc()
        if 'model_validation' in result:result['model_validation']['passed']=False
        try:
            with a.out.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
        except Exception as e:
            print('Could not save failed CPU check: '+repr(e),file=sys.stderr)
        raise
    result['passed']=True
    with a.out.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print('CPU check passed:',a.out)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['identity','imports','predict'],required=True)
    p.add_argument('--order',choices=['vllm-first','sklearn-first'],default='vllm-first')
    p.add_argument('--out',type=Path,required=True);p.add_argument('--migration-dir',type=Path)
    p.add_argument('--expected-prefix',type=Path)
    execute(p.parse_args())


if __name__=='__main__':main()
