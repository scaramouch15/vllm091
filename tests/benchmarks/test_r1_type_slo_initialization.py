import ast
import asyncio
import contextlib
import io
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]

class InitializationTests(unittest.TestCase):
    def test_init_failures_never_send_probe(self):
        tree = ast.parse((ROOT / "benchmarks/benchmark_serving.py").read_text(encoding="utf-8"))
        node = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "benchmark")
        node.returns = None
        for arg in node.args.args + node.args.kwonlyargs:
            arg.annotation = None
        calls = []
        async def send(**kwargs):
            calls.append(kwargs)
            raise AssertionError("probe must not be sent")
        class Response:
            status = 503
            async def __aenter__(self): return self
            async def __aexit__(self, *args): pass
            async def json(self): return {}
        class Session:
            def __init__(self, **kwargs): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *args): pass
            def post(self, *args, **kwargs): return Response()
        fake_aiohttp = types.SimpleNamespace(ClientSession=Session, ClientTimeout=lambda **kw: None)
        import os
        namespace = {"ASYNC_REQUEST_FUNCS": {"vllm": send}, "os": os,
                     "config_digest": lambda config: "digest"}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), "benchmark_serving.py", "exec"), namespace)
        sample = types.SimpleNamespace(metadata={"request_index": 0, "task_type": "chat"})
        args = dict(backend="vllm", api_url="", base_url="", model_id="m", model_name="m", tokenizer=None,
                    input_requests=[sample], logprobs=None, request_rate=1, burstiness=1, disable_tqdm=True,
                    profile=False, selected_percentile_metrics=[], selected_percentiles=[], ignore_eos=False,
                    goodput_config_dict={}, max_concurrency=None, lora_modules=None, extra_body=None,
                    type_slos={"x": 1}, type_output="not-a-real-file", r1_type_slo=True)
        for status in (503, 200):
            Response.status = status
            with self.subTest(status=status), patch.dict(sys.modules, {"aiohttp": fake_aiohttp}), self.assertRaises(ValueError):
                asyncio.run(namespace["benchmark"](**args))
        self.assertEqual(calls, [])

if __name__ == "__main__":
    unittest.main()
