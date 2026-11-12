"""CPU-check AUXLLM encode/slice and OPT final-token head against HF logits.

Does not execute author vLLM GPU kernels; that remains a separate check.
"""
import argparse, hashlib, json, sys
from pathlib import Path

EXPECTED='e7739d26217ee23eb14b6282ee5088f9c79bdc477f456c027b7ebcffacde61ab'
def main():
 p=argparse.ArgumentParser(); p.add_argument('--model',type=Path,required=True); p.add_argument('--tokenizer',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
 if a.output.exists(): raise ValueError('output already exists')
 if hashlib.sha256((a.model/'model.safetensors').read_bytes()).hexdigest()!=EXPECTED: raise ValueError('model SHA mismatch')
 import torch, transformers, tokenizers
 from transformers import AutoTokenizer,AutoModelForSequenceClassification
 from safetensors import safe_open
 torch.set_num_threads(1)
 t=AutoTokenizer.from_pretrained(str(a.tokenizer),local_files_only=True,trust_remote_code=False)
 m=AutoModelForSequenceClassification.from_pretrained(str(a.model),local_files_only=True,use_safetensors=True,trust_remote_code=False).to('cpu').eval()
 with safe_open(str(a.model/'model.safetensors'),framework='pt',device='cpu') as f:
  weight=f.get_tensor('score.weight').float()
 assert torch.equal(weight,m.score.weight.detach().float())
 texts=['Say hello.','Explain the sky in three sentences.','Say hello.',' '.join(['example']*2200)]
 result=[]
 for batch in (texts[:3],texts[3:]):
  enc=t(batch,max_length=2048,padding=True,truncation=True,return_tensors='pt')
  separate=[t.encode(s)[:2048] for s in batch]
  for i,ids in enumerate(separate): assert ids==enc['input_ids'][i][enc['attention_mask'][i].bool()].tolist()
  with torch.inference_mode():
   hidden=m.model(**enc).last_hidden_state
   positions=enc['attention_mask'].sum(-1)-1
   manual=m.score(hidden[torch.arange(len(batch)),positions]).reshape(-1)
   output=m(**enc).logits.reshape(-1)
  assert torch.allclose(manual,output,atol=1e-6,rtol=1e-6)
  for i,ids in enumerate(separate): result.append({'effective_tokens':len(ids),'selected_position':int(positions[i]),'score':float(output[i]),'abs_difference':float(abs(manual[i]-output[i]))})
 scores=[x['score'] for x in result]; order=sorted(range(len(scores)),key=lambda i:(-scores[i],i)); assert scores[0]==scores[2] and order.index(0)<order.index(2)
 doc={'checkpoint_sha256':EXPECTED,'samples':['Say hello.','Explain the sky in three sentences.','Say hello.','example repeated 2200 times'],'cases':result,'stable_descending_order':order,'atol':1e-6,'rtol':1e-6,'device':'cpu','dtype':str(next(m.parameters()).dtype),'python':sys.version,'torch':torch.__version__,'torch_path':torch.__file__,'transformers':transformers.__version__,'transformers_path':transformers.__file__,'tokenizers':tokenizers.__version__,'tokenizers_path':tokenizers.__file__,'limitation':'CPU reconstruction of encode/slice and HF head only; author AUXLLM vLLM GPU execution not tested'}
 with a.output.open('x',encoding='utf-8') as f: json.dump(doc,f,indent=2); f.write('\n')
 print(json.dumps(doc,indent=2))
if __name__=='__main__': main()
