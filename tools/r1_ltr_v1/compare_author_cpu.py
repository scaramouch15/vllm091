"""Compare pinned author score() with the candidate CPU text path, offline."""
import argparse, hashlib, importlib.metadata, json, math, sys
from pathlib import Path

AUTHOR_SHA='b4e73077bab5d70b7f20dd586278868e6b6577c218e2f300a59e2e1e7039b9fb'
MODEL_SHA='e7739d26217ee23eb14b6282ee5088f9c79bdc477f456c027b7ebcffacde61ab'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(); p.add_argument('--author',type=Path,required=True); p.add_argument('--model',type=Path,required=True); p.add_argument('--tokenizer',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    if a.output.exists(): raise ValueError('output exists')
    if sha(a.author)!=AUTHOR_SHA or sha(a.model/'model.safetensors')!=MODEL_SHA: raise ValueError('source or model hash mismatch')
    import torch, transformers, tokenizers
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    torch.set_num_threads(1)
    # Original score() hardcodes cuda:0. Replace only that device for the CPU experiment.
    source=a.author.read_text(encoding='utf-8').replace('"cuda:0"','"cpu"')
    ns={}; exec(compile(source,str(a.author),'exec'),ns)
    tok=AutoTokenizer.from_pretrained(str(a.tokenizer),local_files_only=True,trust_remote_code=False)
    model=AutoModelForSequenceClassification.from_pretrained(str(a.model),local_files_only=True,trust_remote_code=False,use_safetensors=True).to('cpu').eval()
    author=ns['PredModel'].__new__(ns['PredModel']); torch.nn.Module.__init__(author)
    author.model=model; author.tokenizer=tok; author.max_length=2048; author.max_batch_size=1000; author.mtype='rank'; author.activation=torch.nn.Identity()
    texts=['Say hello.','Explain the sky in three sentences.','Say hello.',' '.join(['example']*2200)]
    class Capture:
        def __init__(self,delegate): self.delegate=delegate; self.calls=[]
        def __call__(self,texts,**kw):
            out=self.delegate(texts,**kw); self.calls.append((list(texts),kw,{k:v.clone() for k,v in out.items()})); return out
    author.tokenizer=Capture(tok)
    scores=[]; lengths=[]; differences=[]
    for batch in (texts[:3], texts[3:]):
        with torch.inference_mode():
            av=torch.cat(author.score(batch),dim=0).reshape(-1)
            enc=tok(batch,max_length=2048,padding=True,truncation=True,return_tensors='pt')
            bv=model(**enc).logits.reshape(-1)
        actual,kw,author_enc=author.tokenizer.calls[-1]
        assert actual==batch and kw==dict(max_length=2048,padding=True,truncation=True,return_tensors='pt')
        assert torch.equal(author_enc['input_ids'],enc['input_ids']) and torch.equal(author_enc['attention_mask'],enc['attention_mask'])
        assert torch.allclose(av,bv,atol=1e-6,rtol=1e-6)
        scores.extend(av.tolist()); lengths.extend(enc['attention_mask'].sum(1).tolist())
        differences.append(float((av-bv).abs().max()))
    assert lengths[-1]==2048
    x=scores; order=sorted(range(len(x)),key=lambda i:(-x[i],i)); assert x[0]==x[2] and order.index(0)<order.index(2)
    import tokenizers
    result={'author_sha256':AUTHOR_SHA,'checkpoint_sha256':MODEL_SHA,'samples':texts[:3]+['example repeated 2200 times'],'token_ids_equal':True,'attention_mask_equal':True,'batch_shapes':[[3,max(lengths[:3])],[1,2048]],'attention_lengths':lengths,'max_abs_logit_difference':max(differences),'atol':1e-6,'rtol':1e-6,'scores':x,'descending_stable_order':order,'device':'cpu','dtype':str(next(model.parameters()).dtype),'torch':torch.__version__,'torch_path':torch.__file__,'transformers':transformers.__version__,'transformers_path':transformers.__file__,'tokenizers':tokenizers.__version__,'tokenizers_path':tokenizers.__file__,'python':sys.version,'limit':'CPU substitution for author cuda:0; interface only, not accuracy'}
    with a.output.open('x',encoding='utf-8') as f: json.dump(result,f,indent=2); f.write('\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__': main()
