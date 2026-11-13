"""Bounded real-checkpoint CPU diagnostic; synthetic texts, no service."""
import argparse, asyncio, importlib.util, json, logging, os, platform, re, statistics, sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('ltr_diagnostic_scorer',ROOT/'vllm/v1/metrics/r1_ltr_predictor.py')
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
class Capture(logging.Handler):
 def __init__(self):super().__init__();self.events=[]
 def emit(self,record):
  text=record.getMessage()
  if 'diagnostic_v1' in text:self.events.append(text)
def summary(v):
 v=sorted(v)
 if not v:return None
 return {'n':len(v),'min':v[0],'p50':statistics.median(v),'p95_sample_nearest':v[round((len(v)-1)*.95)],'max':v[-1]}
async def run(scorer,cap,timeout,count,texts,token_counts):
 handler=Capture();m.logger.addHandler(handler);m.logger.setLevel(logging.INFO)
 adapter=m.BoundedAsyncLTRScorer(scorer,cap)
 async def one(i):
  # Fixed texts: three input sizes; no user data and no true labels.
  text=texts[i]; start=time.perf_counter_ns()
  try:
   x=await adapter.score(str(i),text,timeout);return {'id':i,'status':'success','tokens':token_counts[i],'elapsed_ns':time.perf_counter_ns()-start,'score':x.score,'tokenizer_ns':x.tokenization_ns,'inference_ns':x.inference_ns}
  except Exception as e:return {'id':i,'status':str(e) if isinstance(e,m.ScoringUnavailable) else type(e).__name__,'tokens':token_counts[i],'elapsed_ns':time.perf_counter_ns()-start}
 try: rows=await asyncio.gather(*(one(i) for i in range(count)))
 finally: await adapter.close();m.logger.removeHandler(handler)
 return {'max_inflight':cap,'timeout_seconds':timeout,'requests':count,'records':rows,'events':handler.events,'queue_wait_ns':summary([int(match.group(1)) for e in handler.events if (match:=re.search(r'queue_wait_ns=(\d+)',e))]),'foreground_ns':summary([x['elapsed_ns'] for x in rows]),'foreground_success_ns':summary([x['elapsed_ns'] for x in rows if x['status']=='success']),'tokenizer_ns':summary([x['tokenizer_ns'] for x in rows if 'tokenizer_ns' in x]),'inference_ns':summary([x['inference_ns'] for x in rows if 'inference_ns' in x]),'status_counts':{s:sum(x['status']==s for x in rows) for s in sorted(set(x['status'] for x in rows))}}
def main():
 p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--tokenizer',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--threads',type=int,default=2);p.add_argument('--count',type=int,default=12);a=p.parse_args()
 if a.output.exists():raise ValueError('output already exists')
 import torch,transformers,tokenizers
 torch.set_num_threads(a.threads)
 s=m.PaperLTRScorer(a.model,a.tokenizer)
 texts=[' '.join(['example']*([16,128,512][i%3])) for i in range(a.count)]
 token_counts=[len(s.tokenizer.encode(text)) for text in texts]
 s.score('warmup','warm up')
 async def both():
  return [await run(s,2,5.,a.count,texts,token_counts),await run(s,8,5.,a.count,texts,token_counts),await run(s,8,10.,a.count,texts,token_counts)]
 import psutil
 proc=psutil.Process(); before=proc.memory_info().rss; cpu0=proc.cpu_times(); runs=asyncio.run(both());cpu1=proc.cpu_times();after=proc.memory_info().rss
 out={'machine':platform.platform(),'processor':platform.processor(),'logical_cpus':os.cpu_count(),'python':sys.version,'torch':torch.__version__,'transformers':transformers.__version__,'tokenizers':tokenizers.__version__,'threads':a.threads,'checkpoint_sha256':m.EXPECTED_MODEL_SHA256,'rss_before':before,'rss_after':after,'cpu_seconds':(cpu1.user+cpu1.system)-(cpu0.user+cpu0.system),'workload':'simultaneous synthetic requests; one CPU worker; not a server benchmark','runs':runs}
 with a.output.open('x',encoding='utf-8') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps({k:v for k,v in out.items() if k!='runs'},indent=2));print(json.dumps([{k:v for k,v in x.items() if k not in ('records','events')} for x in runs],indent=2))
if __name__=='__main__':main()
