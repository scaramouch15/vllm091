"""Run real benchmark CLI against a deterministic local HTTP test double."""
import argparse, json, os, subprocess, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
class Handler(BaseHTTPRequestHandler):
    calls = []
    fail_formal = False
    def log_message(self, *args): pass
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.__class__.calls.append(body)
        i = len(self.__class__.calls)
        prompt = body['request_info']['prompt']
        def rec(text, fin, count):
            return (json.dumps({'text':[prompt+text], 'request_id': f'engine-{i}',
                    'finished':fin, 'output_tokens':count,
                    'finish_reason':'stop' if fin else None})+'\0').encode()
        empty=rec('', False, 0)
        first=rec('hello', False, 4)
        last=rec('hello world', True, 11)
        content=empty+first if self.fail_formal and i==2 else empty+first+last
        self.send_response(200)
        self.send_header('Content-Type','application/octet-stream')
        self.send_header('Content-Length',str(len(content)+(50 if self.fail_formal and i==2 else 0)))
        self.end_headers()
        self.wfile.write(content[:7]); self.wfile.flush()
        self.wfile.write(content[7:len(empty)]); self.wfile.flush()
        time.sleep(.12)  # empty record must not start TTFT
        self.wfile.write(content[len(empty):]); self.wfile.flush()
        self.close_connection=True

def run(mode, tokenizer, outdir):
    Handler.calls=[]; Handler.fail_formal=(mode=='interrupted')
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
    result=outdir / (mode+'.json'); raw=outdir / 'code.jsonl'
    raw.write_text(json.dumps({'query':'Write a Python function to add two integers.', 'lang':'python','resource':'local-validation'})+'\n',encoding='utf8')
    env=os.environ.copy(); env['PYTHONPATH']=str(ROOT/'tools/jitserve_client_validation/shim')
    env['HF_HUB_OFFLINE']='1'; env['TRANSFORMERS_OFFLINE']='1'
    cmd=[sys.executable, str(ROOT/'benchmarks/benchmark_serving.py'), '--backend','jitserve', '--endpoint','/generate', '--host','127.0.0.1', '--port',str(server.server_port), '--model',str(tokenizer), '--dataset-name','mixed', '--mixed-code-path',str(raw), '--mixed-code-count','1', '--mixed-output-len','2048', '--temperature','0.9', '--num-prompts','1', '--request-rate','8', '--max-concurrency','32', '--disable-tqdm', '--goodput-by-type','chat.ttft:2000','chat.tpot:30','code.e2el:30000','summary.e2el:20000', '--goodput-by-type-output',str(result)]
    try:
        proc=subprocess.run(cmd,cwd=ROOT,env=env,capture_output=True,text=True,encoding='utf8',errors='replace',timeout=180)
    finally:
        server.shutdown(); server.server_close(); thread.join()
    (outdir/(mode+'.stdout.txt')).write_text(proc.stdout,encoding='utf8')
    (outdir/(mode+'.stderr.txt')).write_text(proc.stderr,encoding='utf8')
    (outdir/(mode+'.requests.json')).write_text(json.dumps(Handler.calls,indent=2,ensure_ascii=False),encoding='utf8')
    if proc.returncode: raise RuntimeError(f'{mode}: exit {proc.returncode}; see {outdir/mode}.stderr.txt')
    data=json.loads(result.read_text(encoding='utf8'))
    assert len(Handler.calls)==2, len(Handler.calls)
    a,b=Handler.calls
    assert a['request_info']['request_id'] != b['request_info']['request_id']
    for payload in Handler.calls:
        assert payload['request_info']['request_type']==1
        assert payload['slo_constraint']==[0.0,0.0,30.0]
        assert payload['sampling_params']['max_tokens']==2048
        assert payload['sampling_params']['temperature']==0.9
        assert 'output_len' not in payload['request_info']
        assert 'target_output_length' not in payload and 'target_output_length' not in payload['sampling_params']
    summary=data['summary']['types']['code']
    assert summary['attempted']==1 and summary['attained']==(0 if Handler.fail_formal else 1), summary
    formal=next(iter(data['results'].values()))
    assert formal['client_request_id']==b['request_info']['request_id']
    if not Handler.fail_formal:
        assert formal['output_tokens']==11 and formal['engine_request_id']=='engine-2'
        assert formal['ttft_s'] >= .08, formal['ttft_s']
    else: assert formal['success'] is False
    assert 'Type goodput code:' in proc.stdout
    print(f'{mode}: probe={a["request_info"]["request_id"]} formal={b["request_info"]["request_id"]} attempted=1 attained={summary["attained"]}')

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--tokenizer',type=Path,required=True); ap.add_argument('--output-dir',type=Path,required=True); args=ap.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    for mode in ('complete','interrupted'): run(mode,args.tokenizer,args.output_dir)
