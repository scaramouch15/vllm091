"""Join first-admission shadow predictions to native prefill statistics offline."""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path

SPLIT_SHA256 = '7590e6f446b1cb6014bcbfe4bca65e51f3f7aafcd3cdc3086bd105714087bb5e'

def analyze(shadow, merged, split_path, min_calls=1, min_matches=1, min_sweeps=1):
    if any(type(x) is not int or x < 1 for x in (min_calls,min_matches,min_sweeps)):
        raise ValueError('minimum sample counts must be positive integers')
    records=[json.loads(line) for line in Path(shadow).read_text(encoding='utf-8').splitlines() if line.strip()]
    if not records or records[0].get('record_type')!='header' or records[-1].get('record_type')!='footer':
        raise ValueError('shadow header/footer missing')
    header,footer=records[0],records[-1]
    metrics=footer.get('metrics') or {}
    if (header.get('role')!='shadow-core' or not footer.get('clean_shutdown') or
            not footer.get('records_complete') or metrics.get('dropped_records',0) or metrics.get('writer_errors',0)):
        raise ValueError('shadow writer incomplete or failed')
    seq=[r.get('sequence') for r in records[1:-1]]
    if (any(type(x) is not int for x in seq) or len(set(seq))!=len(seq) or
            seq != sorted(seq) or metrics.get('written_records')!=len(seq) or
            footer.get('last_sequence') != (seq[-1] if seq else 0)):
        raise ValueError('shadow count/sequence mismatch')
    data=json.loads(Path(merged).read_text(encoding='utf-8'))
    if (data.get('collection_complete') is not True or data.get('recording_status')!='complete'
            or not data.get('client_batch_counts_complete') or data.get('unassociated_records')):
        raise ValueError('collector recording incomplete or unassociated records')
    run=header.get('run_id'); counts=collections.Counter(); predictions={};admissions={};candidate_sizes=[];b_candidate_sizes=[]
    for r in records[1:-1]:
        if r.get('run_id')!=run:raise ValueError('shadow run mismatch')
        if r.get('record_type')=='shadow_prefill_v1':
            counts['prediction_batches']+=1
            if len(r.get('candidates',[]))!=r.get('candidate_count'):raise ValueError('candidate count mismatch')
            candidate_sizes.append(r['candidate_count'])
            for c in r['candidates']:
                key=(r.get('decision_batch_id'),c.get('engine_request_id'))
                q=c.get('quantiles_s')
                if (not all(key) or key in predictions or not isinstance(q,list) or len(q)!=3 or
                    any(type(x) not in (int,float) or not math.isfinite(x) or x<0 for x in q) or q!=sorted(q)):
                    raise ValueError('duplicate identity or invalid prediction')
                predictions[key]=q;counts['candidate_predictions']+=1
        elif r.get('record_type')=='shadow_prefill_error_v1':counts['prediction_errors']+=1
        elif r.get('record_type')=='shadow_candidates':
            count=r.get('candidate_count')
            if type(count) is not int or count<1 or len(r.get('candidates',[]))!=count:
                raise ValueError('invalid B candidate batch')
            b_candidate_sizes.append(count)
        elif r.get('record_type')=='shadow_first_admission':
            rid=r.get('engine_request_id');snap=r.get('snapshot') or {}; batch=snap.get('decision_batch_id')
            if not rid or rid in admissions:raise ValueError('duplicate/invalid first admission')
            admissions[rid]=(batch,r.get('predicted_in_this_batch'),snap.get('scheduled_timestamp'))
    split_bytes=Path(split_path).read_bytes()
    split_sha=hashlib.sha256(split_bytes).hexdigest()
    if split_sha!=SPLIT_SHA256:
        raise ValueError('fixed input split SHA256 mismatch')
    split_rows=[json.loads(x) for x in split_bytes.splitlines() if x.strip()]
    split={r.get('input_hash'):r for r in split_rows}
    if (len(split)!=len(split_rows) or None in split or len(split)!=1809 or
            collections.Counter(r.get('split') for r in split_rows)!={'train':1447,'validation':362}):
        raise ValueError('invalid or duplicate fixed split identity')
    rows={}
    for r in data.get('requests',[]):
        if r.get('run_id')!=run:raise ValueError('collector and shadow run_id mismatch')
        rid=r.get('engine_request_id')
        if not rid or rid in rows:raise ValueError('duplicate collector request ID')
        rows[rid]=r
    group_by_rid={}; group_totals=collections.Counter();seen_inputs=set()
    for rid,row in rows.items():
        attempts=row.get('attempts',[])
        if len(attempts)!=1:raise ValueError('attempt missing or duplicate')
        attempt=attempts[0]
        if attempt.get('phase')=='probe':continue
        sample=attempt.get('sample') or {};ih=sample.get('input_hash');ph=sample.get('prompt_hash')
        if attempt.get('phase')!='main' or ih not in split or ph not in split[ih].get('prompt_hashes',[]):
            raise ValueError('main request identity missing or not in fixed split')
        if ih in seen_inputs:raise ValueError('duplicate main input_hash')
        seen_inputs.add(ih);group_by_rid[rid]=split[ih]['split'];group_totals[group_by_rid[rid]]+=1
    matched=[]; matched_groups={'train':[],'validation':[]}
    for rid,(batch,predicted,decision_time) in admissions.items():
        key=(batch,rid)
        if key not in predictions:
            counts['admitted_without_same_decision_prefill']+=1;continue
        counts['admitted_same_decision_prediction']+=1
        row=rows.get(rid)
        if row is None:counts['collector_request_missing']+=1;continue
        attempts=row.get('attempts',[])
        if len(attempts)!=1 or attempts[0].get('phase') not in ('main','probe'):
            counts['attempt_missing_or_ambiguous']+=1;continue
        if attempts[0]['phase']=='probe':
            counts['probe_excluded']+=1;continue
        if row.get('status')!=['complete'] or len(row.get('finished',[]))!=1:
            counts['finished_missing_or_ambiguous']+=1;continue
        f=row['finished'][0];ts=f.get('timestamps') or {};native=f.get('native_stats') or {}
        s,t,v=ts.get('scheduled_ts'),ts.get('first_token_ts'),native.get('prefill_time')
        if (f.get('stats_status')!='valid' or any(type(x) not in (int,float) or not math.isfinite(x) for x in (s,t,v))
                or t<s or v<0 or not math.isclose(v,t-s,rel_tol=1e-9,abs_tol=1e-9)):
            counts['invalid_native_prefill']+=1;continue
        if (type(decision_time) not in (int,float) or not math.isfinite(decision_time)
                or decision_time != s):
            counts['scheduled_timestamp_mismatch']+=1;continue
        counts['matched_first_admission_finished']+=1;matched.append((v,predictions[key]))
        matched_groups[group_by_rid[rid]].append((v,predictions[key]))
    counts['candidate_predictions_without_same_decision_admission']=sum(
        1 for batch,rid in predictions if admissions.get(rid,(None,None))[0]!=batch)
    summary={}
    for i,q in enumerate((.5,.9,.95)):
        if matched:
            residual=[y-p[i] for y,p in matched]
            summary[str(q)]={'n':len(matched),'coverage':sum(x<=0 for x in residual)/len(matched),
                'pinball_s':sum(max(q*x,(q-1)*x) for x in residual)/len(matched),
                'mean_prediction_error_s':sum(-x for x in residual)/len(matched)}
        else:summary[str(q)]=None
    group_quality={}
    for name, group in matched_groups.items():
        qmetrics={}
        for i,q in enumerate((.5,.9,.95)):
            residual=[y-p[i] for y,p in group]
            qmetrics[str(q)]=({'n':len(group),'coverage':sum(x<=0 for x in residual)/len(group),
                'pinball_s':sum(max(q*x,(q-1)*x) for x in residual)/len(group)} if group else None)
        group_quality[name]={'matched':len(group),'missing':group_totals[name]-len(group),
            'formal_requests':group_totals[name], 'quality':qmetrics,
            'interpretation':('training_inputs_not_out_of_sample' if name=='train' else 'internal_validation_not_final_test')}
    summaries=[r for r in records if r.get('record_type')=='shadow_summary']
    if len(summaries)!=1:raise ValueError('shadow summary missing or duplicate')
    shadow_summary=summaries[0];sc=shadow_summary.get('counts') or {};metadata=header.get('metadata') or {}
    enabled=metadata.get('prefill_enabled')
    if type(enabled) is not bool:raise ValueError('missing prefill enabled flag')
    if sc.get('prefill_success',0)!=counts['prediction_batches'] or sc.get('prefill_errors',0)!=counts['prediction_errors']:
        raise ValueError('prefill summary and event counts mismatch')
    if sc.get('dynamic_success',0)!=len(b_candidate_sizes):
        raise ValueError('B summary and candidate batch counts mismatch')
    if any(sc.get(k,0) for k in ('dynamic_errors','prefill_errors','prefill_instrumentation_errors','log_errors','log_dropped')):
        raise ValueError('prefill prediction or shadow logging error: '+json.dumps(
            {k:sc.get(k,0) for k in ('dynamic_errors','prefill_calls','prefill_errors','prefill_instrumentation_errors','log_errors','log_dropped')}))
    if enabled and sc.get('prefill_calls',0)!=counts['prediction_batches']:
        raise ValueError('prefill call count mismatch')
    if enabled and sc.get('prefill_calls',0)!=sc.get('dynamic_success',0):
        raise ValueError('prefill calls and B successful batches mismatch')
    if not enabled and (sc.get('prefill_calls',0) or counts['prediction_batches']):
        raise ValueError('control unexpectedly contains prefill calls')
    identity=[]
    for rid,row in rows.items():
        attempts=row.get('attempts',[])
        if len(attempts)!=1:raise ValueError('attempt missing or duplicate')
        a=attempts[0]
        if a.get('phase')=='probe':continue
        sample=a.get('sample') or {}
        item=(sample.get('request_index'),sample.get('input_hash'),sample.get('prompt_hash'),sample.get('task_type'))
        if (a.get('phase')!='main' or type(item[0]) is not int or
                any(not isinstance(x,str) or not x for x in item[1:])):raise ValueError('invalid main identity')
        identity.append(item)
    if len({item[0] for item in identity})!=len(identity):raise ValueError('duplicate main request index')
    identity.sort()
    fingerprint=hashlib.sha256(json.dumps(identity,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
    clients=[f.get('header',{}).get('metadata',{}) for f in data.get('files',[]) if f.get('header',{}).get('role')=='client']
    if len(clients)!=1:raise ValueError('expected one client shard per run')
    client=clients[0]; distributions=(shadow_summary.get('foreground_distributions') or {}).get('overall') or {}
    if any(k not in client for k in ('seed','model','burstiness','sampling_params','ignore_eos','request_rate','max_concurrency')):
        raise ValueError('client configuration metadata missing')
    if 'schedule_wall_v1' not in distributions:raise ValueError('schedule timing is unavailable')
    groups=(shadow_summary.get('foreground_distributions') or {}).get('groups') or []
    if any(g.get('stage')=='schedule_wall_v1' and g.get('outcome')!='success' and g.get('n',0)>0 for g in groups):
        raise ValueError('failed schedule calls recorded')
    if sc.get('waiting_opportunities',0)<min_sweeps:status='insufficient_samples'
    elif enabled and (counts['prediction_batches']<min_calls or counts['matched_first_admission_finished']<min_matches):
        status='insufficient_samples'
    else:status='exploratory_eligible'
    return {'run_id':run,'status':status,'prefill_enabled':enabled,'min_calls':min_calls,'min_matches':min_matches,'min_sweeps':min_sweeps,
            'main_request_count':len(identity),'main_identity_sha256':fingerprint,
            'request_rate':client.get('request_rate'),'max_concurrency':client.get('max_concurrency'),
            'client_config':{k:client.get(k) for k in ('seed','model','burstiness','sampling_params','ignore_eos')},
            'sweep_opportunities':sc.get('waiting_opportunities',0),
            'prefill_calls':sc.get('prefill_calls',0),'candidate_count_distribution':dict(collections.Counter(candidate_sizes)),
            'b_candidate_batches':len(b_candidate_sizes),
            'b_candidate_count_distribution':dict(collections.Counter(b_candidate_sizes)),
            'fixed_split_sha256':split_sha,'grouped_quality':group_quality,
            'counts':dict(counts),'quality':summary if enabled and status=='exploratory_eligible' else None,
            'shadow_summary':summaries[0], 'collector_recording_status':data['recording_status'],
            'collector_collection_complete':data.get('collection_complete'),
            'limit':'Only same-decision first admissions with complete native finished statistics; not all waiting candidates.'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--shadow',required=True);p.add_argument('--merged',required=True);p.add_argument('--split',required=True);p.add_argument('--output',required=True);p.add_argument('--min-calls',type=int,required=True);p.add_argument('--min-matches',type=int,required=True);p.add_argument('--min-sweeps',type=int,required=True);a=p.parse_args()
    try: result=analyze(a.shadow,a.merged,a.split,a.min_calls,a.min_matches,a.min_sweeps)
    except (ValueError,KeyError,TypeError,OSError,json.JSONDecodeError) as exc:
        result={'status':'invalid','error_type':type(exc).__name__,'error':str(exc)}
    with open(a.output,'x',encoding='utf-8',newline='\n') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    if result['status']!='exploratory_eligible':raise SystemExit(2)
if __name__=='__main__':main()
