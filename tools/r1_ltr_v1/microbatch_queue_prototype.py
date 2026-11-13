"""Offline bounded microbatch CPU experiment; not wired into vLLM."""
import argparse, asyncio, concurrent.futures, hashlib, importlib.util, json, math, os, platform, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('ltr_proto',ROOT/'vllm/v1/metrics/r1_ltr_predictor.py');m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
async def experiment(s, texts, reference, token_counts, max_batch, delay_s, rate, cap, timeout):
 import torch, psutil
 loop=asyncio.get_running_loop(); q=asyncio.Queue(maxsize=cap); inflight=set(); rows=[{'id':i,'status':'not_started'} for i in range(len(texts))]; executor=concurrent.futures.ThreadPoolExecutor(max_workers=1); batch_records=[]
 def infer(batch):
  t=time.perf_counter_ns(); enc=s.tokenizer([x['text'] for x in batch],max_length=2048,padding=True,truncation=True,return_tensors='pt'); tok=time.perf_counter_ns()-t;t=time.perf_counter_ns()
  with torch.inference_mode():scores=s.model(**enc).logits[:,0].tolist()
  return scores,tok,time.perf_counter_ns()-t
 async def worker():
  while True:
   first=await q.get()
   if first is None: break
   if first['future'].cancelled(): inflight.discard(first['id']);continue
   first['picked']=time.perf_counter_ns(); batch=[first]; deadline=loop.time()+delay_s if max_batch>1 else loop.time()
   while len(batch)<max_batch:
    remaining=deadline-loop.time()
    if remaining<=0:break
    try:x=await asyncio.wait_for(q.get(),remaining)
    except asyncio.TimeoutError:break
    if x is None:
     await q.put(None)
     break
    if x['future'].cancelled():inflight.discard(x['id']);continue
    x['picked']=time.perf_counter_ns();batch.append(x)
   dispatch=time.perf_counter_ns()
   try:
    vals,tok,inf=await loop.run_in_executor(executor,infer,batch)
    if len(vals)!=len(batch) or any(not math.isfinite(float(v)) for v in vals):
     raise ValueError('invalid batch scores')
    batch_records.append({'size':len(batch),'tokenizer_ns':tok,'inference_ns':inf})
    for x,val in zip(batch,vals):
     row=rows[x['id']];row.update(queue_wait_ns=x['picked']-x['submitted'],aggregation_wait_ns=dispatch-x['picked'],batch_size=len(batch),tokenizer_batch_ns=tok,inference_batch_ns=inf)
     if not x['future'].done():x['future'].set_result(float(val))
     else:row['late_completion']=True
     inflight.discard(x['id'])
   except Exception as e:
    for x in batch:
     if not x['future'].done():x['future'].set_exception(e)
     inflight.discard(x['id'])
 async def one(i,origin):
  await asyncio.sleep(max(0., origin+i/rate-loop.time()));now=time.perf_counter_ns();row=rows[i];row['input_token_count']=token_counts[i]
  if len(inflight)>=cap or q.full():row.update(status='queue_full',foreground_ns=time.perf_counter_ns()-now);return
  inflight.add(i);f=loop.create_future();await q.put({'id':i,'text':texts[i],'future':f,'submitted':now})
  try:
   value=await asyncio.wait_for(f,timeout);row.update(status='success',score=value,foreground_ns=time.perf_counter_ns()-now,abs_difference=abs(value-reference[i]))
  except asyncio.TimeoutError:row.update(status='timeout',foreground_ns=time.perf_counter_ns()-now)
  except asyncio.CancelledError:
   row.update(status='cancelled',foreground_ns=time.perf_counter_ns()-now)
   raise
  except Exception as e:row.update(status='model_exception',error_type=type(e).__name__,foreground_ns=time.perf_counter_ns()-now)
 proc=psutil.Process();before=proc.memory_info().rss;cpu0=proc.cpu_times();t=time.perf_counter_ns();task=asyncio.create_task(worker());origin=loop.time()
 try:
  await asyncio.gather(*(one(i,origin) for i in range(len(texts))))
 finally:
  await q.put(None)
  try: await task
  finally: executor.shutdown(wait=True)
 elapsed=time.perf_counter_ns()-t;cpu1=proc.cpu_times()
 return {'max_batch':max_batch,'max_aggregation_wait_ms':delay_s*1000,'rate':rate,'max_inflight':cap,'timeout_s':timeout,'requests':len(texts),'successful':sum(x['status']=='success' for x in rows),'elapsed_ns_including_drain':elapsed,'throughput_per_s_including_drain':sum(x['status']=='success' for x in rows)*1e9/elapsed,'cpu_seconds':(cpu1.user+cpu1.system)-(cpu0.user+cpu0.system),'rss_before':before,'rss_after':proc.memory_info().rss,'rows':rows,'batches':batch_records}
def main():
 p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--tokenizer',type=Path,required=True);p.add_argument('--client-jsonl',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise ValueError('output exists')
 import torch,psutil
 torch.set_num_threads(2)
 entries=[json.loads(l) for l in a.client_jsonl.read_text(encoding='utf8').splitlines()]; data=[x for x in entries if x.get('record_type')=='attempt' and x.get('phase')=='main'];data.sort(key=lambda x:x['sample']['prompt_len']);texts=[data[i]['sample']['prompt'] for i in (0,4,8,12,16,20,24,29)];texts.append(texts[0]);texts.append(' '.join(['example']*2200))
 reference=json.loads(a.reference.read_text());assert hashlib.sha256(a.client_jsonl.read_bytes()).hexdigest()==reference['source_file_sha256'];ref=reference['timings'][0]['scores'];token_counts=reference['prompt_token_lengths_before_truncation'];s=m.PaperLTRScorer(a.model,a.tokenizer);s.score('warmup','warmup')
 async def run():return [await experiment(s,texts,ref,token_counts,1,0.,8,8,5),await experiment(s,texts,ref,token_counts,2,.01,8,8,5)]
 result={'platform':platform.platform(),'cpu':platform.processor(),'logical_cpus':os.cpu_count(),'process_affinity':len(psutil.Process().cpu_affinity()),'torch_threads':torch.get_num_threads(),'model_sha256':m.EXPECTED_MODEL_SHA256,'limitations':'experimental prototype; no cancellations beyond deadlines; not service or production scheduler','runs':asyncio.run(run())}
 with a.output.open('x',encoding='utf8') as f:json.dump(result,f,indent=2);f.write('\n')
 for r in result['runs']: print({k:v for k,v in r.items() if k not in ('rows','batches')},[(x['id'],x['status']) for x in r['rows']])
if __name__=='__main__':main()
