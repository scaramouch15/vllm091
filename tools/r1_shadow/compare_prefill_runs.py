"""Compare two independent shadow runs; no per-schedule pairing is assumed."""
import argparse
import json


def compare(control, enabled):
    errors=[]
    if control.get('run_id')==enabled.get('run_id'):errors.append('run_id_not_distinct')
    if control.get('prefill_enabled') is not False or enabled.get('prefill_enabled') is not True:
        errors.append('invalid_control_or_enabled_mode')
    for label,row in (('control',control),('enabled',enabled)):
        if row.get('status')!='exploratory_eligible':errors.append(label+'_not_eligible:'+str(row.get('status')))
    for key in ('main_identity_sha256','main_request_count','request_rate','max_concurrency','client_config','fixed_split_sha256'):
        if control.get(key) is None or control.get(key)!=enabled.get(key):errors.append('mismatch:'+key)
    def one(row):
        summ=row.get('shadow_summary') or {};dist=(summ.get('foreground_distributions') or {}).get('overall') or {}
        return {'run_id':row.get('run_id'),'status':row.get('status'),
                'sweep_opportunities':row.get('sweep_opportunities'),
                'prefill_calls':row.get('prefill_calls'),
                'candidate_count_distribution':row.get('candidate_count_distribution'),
                'b_candidate_batches':row.get('b_candidate_batches'),
                'b_candidate_count_distribution':row.get('b_candidate_count_distribution'),
                'matched':(row.get('counts') or {}).get('matched_first_admission_finished',0),
                'schedule_wall_v1':dist.get('schedule_wall_v1'),
                'prefill_foreground_v1':dist.get('prefill_foreground_v1')}
    a,b=one(control),one(enabled)
    if not a['schedule_wall_v1'] or not b['schedule_wall_v1']:errors.append('missing_schedule_timing')
    if not b['prefill_foreground_v1']:errors.append('missing_prefill_timing')
    return {'status':'not_eligible' if errors else 'exploratory_comparison',
            'errors':errors,'control':a,'enabled':b,
            'identity_sha256':control.get('main_identity_sha256'),
            'note':'Separate independent runs: no per-schedule pairing; histogram quantiles are bounds, not paired deltas.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--control',required=True);p.add_argument('--enabled',required=True);p.add_argument('--output',required=True)
    a=p.parse_args()
    with open(a.control,encoding='utf8') as f:control=json.load(f)
    with open(a.enabled,encoding='utf8') as f:enabled=json.load(f)
    result=compare(control,enabled)
    with open(a.output,'x',encoding='utf8',newline='\n') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    if result['status']!='exploratory_comparison':raise SystemExit(2)
if __name__=='__main__':main()
