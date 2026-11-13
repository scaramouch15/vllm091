import contextlib, io, json, sys, tempfile, types, unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'benchmarks'))
import benchmark_serving as bench
from backend_request_func import RequestFuncOutput

class Integration(unittest.IsolatedAsyncioTestCase):
    async def test_probe_and_formal_failure(self):
        calls=[]
        async def send(request_func_input, pbar=None):
            inp=request_func_input
            calls.append((inp.task_type, inp.r1_type_attempt is None, inp.request_id))
            if inp.r1_type_attempt: inp.r1_type_attempt()
            return RequestFuncOutput(success=len(calls)==1, latency=0.5, ttft=0.1, output_tokens=1, generated_text='x', error='' if len(calls)==1 else 'failed')
        req=types.SimpleNamespace(prompt='p', prompt_len=1, expected_output_len=2048, multi_modal_data=None, metadata={'request_index': 7, 'task_type':'code'})
        config={'chat.ttft':2.,'chat.tpot':.03,'code.e2el':30.,'summary.e2el':20.}
        with tempfile.TemporaryDirectory() as d, patch.dict(bench.ASYNC_REQUEST_FUNCS, {'jitserve':send}), contextlib.redirect_stdout(io.StringIO()):
            path=str(Path(d)/'score.json')
            await bench.benchmark('jitserve','http://x/generate','http://x','m',None,None,[req],None,float('inf'),1,True,False,[],[],False,{},1,None,None,type_slos=config,type_output=path)
            data=json.loads(Path(path).read_text())
        self.assertEqual(len(calls),2)
        self.assertEqual(calls[0][:2],('code',True))
        self.assertEqual(calls[1][:2],('code',False))
        self.assertTrue(calls[1][2])
        self.assertEqual(data['summary']['overall']['attempted'],1)
        self.assertEqual(data['summary']['overall']['attained'],0)
        self.assertIn('7', data['results'])
if __name__ == '__main__': unittest.main()
