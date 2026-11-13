"""Offline real-checkpoint scoring comparison; result excludes input text/IDs."""
import argparse, hashlib, importlib.util, json, os, platform, statistics, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('ltr_compare_impl',ROOT/'vllm/v1/metrics/r1_ltr_predictor.py')
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
def main():
 p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--tokenizer',type=Path,required=True);p.add_argument('--client-jsonl',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--threads',type=int,default=2);a=p.parse_args()
 if a.output.exists():raise ValueError('output exists')
 import torch,transformers,tokenizers,psutil
 torch.set_num_threads(a.threads)
 rows=[json.loads(line) for line in a.client_jsonl.read_text(encoding='utf8').splitlines() if line.strip()]
 main=[r for r in rows if r.get('record_type')=='attempt' and r.get('phase')=='main']
 if len(main)!=30:raise ValueError('expected 30 main attempts')
 main.sort(key=lambda r:r['sample']['prompt_len'])
 picked=[main[i] for i in (0,4,8,12,16,20,24,29)]
 texts=[r['sample']['prompt'] for r in picked]; texts.append(texts[0]);texts.append(' '.join(['example']*2200))
 scorer=m.PaperLTRScorer(a.model,a.tokenizer);tok=scorer.tokenizer;model=scorer.model;model.eval()
 sizes=[len(tok.encode(x)) for x in texts]; proc=psutil.Process(); rss0=proc.memory_info().rss; timings=[]; scores_by_size={}
 with torch.inference_mode():model(**tok('warmup',return_tensors='pt'))
 for size in (1,2,4):
  scores=[];times=[]; token_times=[];infer_times=[]; cpu0=proc.cpu_times();start=time.perf_counter_ns()
  for offset in range(0,len(texts),size):
   b=texts[offset:offset+size]; t=time.perf_counter_ns(); enc=tok(b,max_length=2048,padding=True,truncation=True,return_tensors='pt'); token_times.append(time.perf_counter_ns()-t)
   t=time.perf_counter_ns()
   with torch.inference_mode(): logits=model(**enc).logits[:,0]
   infer_times.append(time.perf_counter_ns()-t); scores.extend(float(x) for x in logits.tolist());times.append(len(b))
  elapsed=time.perf_counter_ns()-start;cpu1=proc.cpu_times()
  scores_by_size[size]=scores
  timings.append({'batch_size':size,'requests':len(texts),'batches':len(times),'wall_ns':elapsed,'throughput_per_second':len(texts)*1e9/elapsed,'tokenizer_batch_ns':token_times,'inference_batch_ns':infer_times,'cpu_seconds':(cpu1.user+cpu1.system)-(cpu0.user+cpu0.system),'rss_after':proc.memory_info().rss,'scores':scores,'stable_order':sorted(range(len(scores)),key=lambda i:(-scores[i],i))})
 ref=scores_by_size[1]; comparisons=[]
 for size in (2,4):
  got=scores_by_size[size];delta=max(abs(x-y) for x,y in zip(ref,got)); order=sorted(range(len(got)),key=lambda i:(-got[i],i)); ref_order=sorted(range(len(ref)),key=lambda i:(-ref[i],i))
  comparisons.append({'batch_size':size,'max_abs_diff':delta,'tolerance_atol':1e-5,'tolerance_rtol':1e-5,'within_tolerance':all(abs(x-y)<=1e-5+1e-5*abs(x) for x,y in zip(ref,got)),'same_stable_order':order==ref_order,'duplicate_diff':abs(got[0]-got[8])})
 out={'source_file_sha256':hashlib.sha256(a.client_jsonl.read_bytes()).hexdigest(),'model_sha256':m.EXPECTED_MODEL_SHA256,'platform':platform.platform(),'cpu':platform.processor(),'cpu_logical':os.cpu_count(),'cpu_affinity':len(proc.cpu_affinity()) if hasattr(proc,'cpu_affinity') else None,'torch_threads':torch.get_num_threads(),'torch':torch.__version__,'transformers':transformers.__version__,'tokenizers':tokenizers.__version__,'rss_before':rss0,'prompt_token_lengths_before_truncation':sizes,'samples':'8 distinct recorded main prompts, one duplicate and one synthetic overlength; no prompt text saved','timings':timings,'comparisons':comparisons,'scope':'saturated sequential batches, not queueing or server throughput'}
 with a.output.open('x',encoding='utf8') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(out,indent=2))
if __name__=='__main__':main()
