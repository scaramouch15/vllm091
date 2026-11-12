"""EngineCoreRequest declaration round-trip through real msgspec bytes.

The project class is AST-extracted with annotation stubs, not a full engine.
"""
import ast
from pathlib import Path
import subprocess
import sys
import typing
import unittest
import msgspec

class SamplingParams(msgspec.Struct):
    extra_args: typing.Any = None

class LTRWireTests(unittest.TestCase):
    def test_actual_struct_score_roundtrip_and_subprocess_pipe(self):
        p = Path(__file__).resolve().parents[2] / 'vllm/v1/engine/__init__.py'
        node = next(n for n in ast.parse(p.read_text(encoding='utf-8')).body
                    if isinstance(n, ast.ClassDef) and n.name == 'EngineCoreRequest')
        ns = {'msgspec': msgspec, 'Optional': typing.Optional, 'Sequence': typing.Sequence,
              'MultiModalKwargs': typing.Any, 'PlaceholderRange': typing.Any,
              'LoRARequest': typing.Any, 'SamplingParams': SamplingParams}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(p), 'exec'), ns)
        cls = ns['EngineCoreRequest']
        for score in (-2.5, None, 3.0):
            obj = cls(request_id='request-1', prompt_token_ids=[1], mm_inputs=None,
                      mm_hashes=None, mm_placeholders=None, sampling_params=SamplingParams(),
                      eos_token_id=None, arrival_time=1., lora_request=None, cache_salt=None,
                      data_parallel_rank=None, r1_ltr_score=score)
            wire=msgspec.msgpack.encode(obj)
            self.assertEqual(msgspec.msgpack.decode(wire,type=cls).r1_ltr_score, score)
            child=subprocess.run([sys.executable,'-c','import sys;sys.stdout.buffer.write(sys.stdin.buffer.read())'],
                                 input=wire,stdout=subprocess.PIPE,check=True)
            decoded=msgspec.msgpack.decode(child.stdout,type=cls)
            self.assertEqual((decoded.request_id,decoded.r1_ltr_score),('request-1',score))
