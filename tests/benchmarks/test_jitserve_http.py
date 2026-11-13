import asyncio, json, sys, unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'benchmarks'))
import backend_request_func as backend

class Content:
    def __init__(self, chunks, error=False): self.chunks, self.error = chunks, error
    async def iter_any(self):
        for chunk in self.chunks: yield chunk
        if self.error: raise ConnectionError('interrupted')
class Response:
    def __init__(self, status, chunks=(), error=False):
        self.status, self.reason, self.content = status, 'bad', Content(chunks, error)
    async def __aenter__(self): return self
    async def __aexit__(self, *args): pass
class Session:
    response = None
    payload = None
    def __init__(self, **kwargs): pass
    async def __aenter__(self): return self
    async def __aexit__(self, *args): pass
    def post(self, url, json):
        Session.payload = json
        return Session.response

def rec(text, finished=False, tokens=1):
    return json.dumps({'text': ['p'+text], 'finished': finished, 'request_id': 'engine', 'output_tokens': tokens, 'finish_reason': 'stop' if finished else None}).encode()+b'\0'
class BackendTest(unittest.IsolatedAsyncioTestCase):
    async def run_req(self, response, kind='code', attempted=None):
        Session.response = response
        req = backend.RequestFuncInput(prompt='p', api_url='http://example/generate', prompt_len=1, output_len=2048, model='m', task_type=kind, request_id='client', r1_type_attempt=attempted)
        with patch.object(backend.aiohttp, 'ClientSession', Session):
            return await backend.async_request_jitserve(req)
    async def test_complete_and_mapping(self):
        raw = rec('a') + rec('ab', True, 3)
        calls=[]
        out = await self.run_req(Response(200, [raw[:4], raw[4:]]), attempted=lambda: calls.append(1))
        self.assertTrue(out.success, out.error)
        self.assertEqual((out.generated_text, out.output_tokens, out.engine_request_id), ('ab', 3, 'engine'))
        self.assertEqual(calls, [1])
        self.assertEqual(Session.payload['slo_constraint'], [0.,0.,30.])
        self.assertEqual(Session.payload['sampling_params']['max_tokens'], 2048)
        self.assertNotIn('output_len', Session.payload['request_info'])
        self.assertNotIn('target_output_length', Session.payload)
        self.assertEqual(Session.payload['request_info']['request_id'], 'client')
    async def test_failures(self):
        for response in (Response(503), Response(200,[rec('a')]), Response(200,[rec('a')], True), Response(200,[rec('a')[:-1]])):
            out = await self.run_req(response)
            self.assertFalse(out.success)
            self.assertTrue(out.error)
    async def test_all_types(self):
        for kind, expected in [('chat', [2.,.03,0.]), ('summary',[0.,0.,20.])]:
            out = await self.run_req(Response(200,[rec('x',True)]), kind)
            self.assertTrue(out.success, out.error)
            self.assertEqual(Session.payload['slo_constraint'], expected)

if __name__ == '__main__': unittest.main()
