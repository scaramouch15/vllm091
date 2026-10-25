"""Disposable CPU import/coexistence checks. Never starts an engine/model."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import traceback


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def check(a,result):
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
            assert not module.cuda.is_initialized(),'CUDA initialized while importing torch'
            # Fail before lazy CUDA initialization, including indirect calls by vLLM.
            def forbid_cuda_init(*args,**kwargs):
                raise RuntimeError('CUDA initialization is forbidden in this CPU compatibility check')
            module.cuda._lazy_init=forbid_cuda_init
            for entry in ['_cuda_init','_cuda_getDeviceCount']:
                if hasattr(module._C,entry):setattr(module._C,entry,forbid_cuda_init)
    import torch
    assert not torch.cuda.is_initialized(),'CUDA initialized during imports'
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
        assert Path(sys.prefix).resolve()==prefix,'Wrong clone interpreter'
        for name,info in result['modules'].items():
            if name!='vllm':assert Path(info['file']).resolve().is_relative_to(prefix),name+' outside clone'
    if a.mode!='identity':
        expected={'numpy':'2.2.6','scipy':'1.15.3','sklearn':'1.7.2','joblib':'1.5.3','threadpoolctl':'3.6.0'}
        import platform
        assert platform.python_version()=='3.10.22'
        for name,version in expected.items():assert result['modules'][name]['version']==version,name
    if a.mode=='predict':
        assert a.migration_dir,'--migration-dir required'
        root=a.migration_dir.resolve()
        for name,digest in json.loads((root/'files_sha256.json').read_text()).items():
            path=(root/name).resolve();assert path.is_relative_to(root)
            assert sha(path)==digest,name
        sys.path.insert(0,str(root))
        import numpy as np
        import joblib
        from common import ids,read_rows,reference_modules,check_split
        from train_cpu import stages,comparison
        data=root/'data/validation.jsonl';source=root/'config_B_new310/results';model=source/'models/tfidf.joblib'
        assert sha(model)=='23aff057712e7bec05db91d470af39a1b34eabddbf5852914710e1b8e4b96e1f'
        assert sha(data)=='1b7fe1e7e8036bc7f7b943866d1a2a28ace2dc7bd6e5875e44f5ef6e2e6f5d9f'
        rows=read_rows(data);previous=read_rows(source/'tfidf_validation.jsonl')
        check_split(rows,'validation',1440,360)
        assert len(previous)==1440 and ids(rows)==ids(previous)
        base,ref=reference_modules()
        # Only this disposable verification process uses a process-wide limit.
        with threadpoolctl.threadpool_limits(limits=1):
            result['threadpools_during_validation']=threadpoolctl.threadpool_info()
            assert result['threadpools_during_validation'] and all(x['num_threads']==1 for x in result['threadpools_during_validation'])
            bundle=joblib.load(model)
            actual=stages(bundle,rows,base,ref)
            columns={'length':'predicted_length_quantiles','time':'predicted_time_quantiles',
                     'length_raw':'raw_length_quantiles','time_raw':'raw_time_quantiles'}
            checks={k:comparison(actual[k],np.array([x[v] for x in previous])) for k,v in columns.items()}
            result['model_validation']=dict(n=1440,identity_order_verified=True,rtol=1e-10,atol=1e-10,
                model_sha256=sha(model),data_sha256=sha(data),stages=checks,
                passed=all(x['failed_elements']==0 for x in checks.values()))
        result['threadpools_after_validation']=threadpoolctl.threadpool_info()
        assert result['model_validation']['passed'],'Strict validation failed; tolerance unchanged'
        assert not torch.cuda.is_initialized(),'CUDA initialized during CPU validation'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['identity','imports','predict'],required=True)
    p.add_argument('--order',choices=['vllm-first','sklearn-first'],default='vllm-first')
    p.add_argument('--out',type=Path,required=True);p.add_argument('--migration-dir',type=Path)
    p.add_argument('--expected-prefix',type=Path)
    a=p.parse_args();result=dict(executable=sys.executable,prefix=sys.prefix,mode=a.mode,order=a.order)
    try:check(a,result);result['passed']=True
    except BaseException:
        result['passed']=False;result['error']=traceback.format_exc()
        raise
    finally:
        with a.out.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print('CPU check passed:',a.out)


if __name__=='__main__':main()
