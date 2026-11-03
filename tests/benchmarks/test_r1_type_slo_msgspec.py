import ast
import typing
import unittest
from pathlib import Path
import msgspec

class SamplingParams(msgspec.Struct):
    extra_args: typing.Any = None

class EngineCoreRequestCodecTests(unittest.TestCase):
    def test_actual_struct_declaration_msgpack_roundtrip(self):
        path = Path(__file__).resolve().parents[2] / "vllm/v1/engine/__init__.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        node = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == "EngineCoreRequest")
        namespace = {"msgspec": msgspec, "Optional": typing.Optional, "Sequence": typing.Sequence,
                     "MultiModalKwargs": typing.Any, "PlaceholderRange": typing.Any,
                     "LoRARequest": typing.Any, "SamplingParams": SamplingParams}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), "exec"), namespace)
        cls = namespace["EngineCoreRequest"]
        for task in ("chat", "code", "summary"):
            item = cls(request_id="id", prompt_token_ids=[1], mm_inputs=None, mm_hashes=None,
                       mm_placeholders=None, sampling_params=SamplingParams(), eos_token_id=None,
                       arrival_time=1.0, lora_request=None, cache_salt=None, data_parallel_rank=None,
                       r1_request_type=task, r1_type_slo_instance="instance-1")
            wire = msgspec.msgpack.encode(item)
            result = msgspec.msgpack.decode(wire, type=cls)
            self.assertEqual(result.r1_request_type, task)
            self.assertEqual(result.r1_type_slo_instance, "instance-1")
            # Exercise a real child-process byte pipe. The child only echoes the
            # bytes; it is not an EngineCore or a full vLLM installation.
            import subprocess
            import sys
            child = subprocess.run([sys.executable, "-c",
                "import sys; sys.stdout.buffer.write(sys.stdin.buffer.read())"],
                input=wire, stdout=subprocess.PIPE, check=True)
            decoded = msgspec.msgpack.decode(child.stdout, type=cls)
            self.assertEqual(decoded.r1_request_type, task)
            self.assertEqual(decoded.r1_type_slo_instance, "instance-1")

if __name__ == "__main__":
    unittest.main()
