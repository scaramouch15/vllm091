import ast
import asyncio
import os
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]

class FrontendGuardTests(unittest.TestCase):
    def test_guard_before_add_and_disabled_compatibility(self):
        tree = ast.parse((ROOT / "vllm/v1/engine/async_llm.py").read_text(encoding="utf-8"))
        cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == "AsyncLLM")
        node = next(x for x in cls.body if isinstance(x, ast.AsyncFunctionDef) and x.name == "add_request")
        node.returns = None
        for arg in node.args.args + node.args.kwonlyargs: arg.annotation = None
        class Params:
            n = 1
            output_kind = None
            def __init__(self, value): self.extra_args = {"_vllm_r1_request_type": value}
        class Core:
            initialized = False
            async def call_utility_async(self, method, value):
                if not self.initialized or value not in ("chat", "code", "summary"):
                    raise RuntimeError("R1 type invalid or uninitialized")
                return {"run_id": "x", "config_sha256": "y"}
        class Processor:
            def process_inputs(self, *args): return "prompt", types.SimpleNamespace()
        class Fake:
            errored = False
            collection_config = None
            engine_core = Core()
            processor = Processor()
            added = False
            async def _add_request(self, *args): self.added = True
        namespace = {"SamplingParams": Params, "RequestOutputCollector": lambda **kwargs: object()}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), "async_llm.py", "exec"), namespace)
        run = namespace["add_request"]
        async def exercise():
            fake = Fake()
            async def unexpected_utility(*args):
                raise AssertionError("per-request utility must not be called")
            fake.engine_core.call_utility_async = unexpected_utility
            with patch.dict(os.environ, {"VLLM_R1_TYPE_SLO": "1"}):
                await run(fake, "id", "text", Params("code"))
                self.assertTrue(fake.added)
            fake.added = False
            with patch.dict(os.environ, {"VLLM_R1_TYPE_SLO": "0"}):
                await run(fake, "id", "text", Params(None))
                self.assertTrue(fake.added)
        asyncio.run(exercise())

if __name__ == "__main__": unittest.main()
