import ast
import importlib.util
from pathlib import Path
import types
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("r1state", ROOT / "vllm/v1/core/sched/r1_type_slo_state.py")
state_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(state_mod)

class CoreAddTests(unittest.TestCase):
    def test_invalid_or_stale_returns_per_request_error(self):
        tree = ast.parse((ROOT / "vllm/v1/engine/core.py").read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "EngineCoreProc")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_handle_client_request")
        method.returns = None
        for arg in method.args.args: arg.annotation = None
        kinds = types.SimpleNamespace(ADD=1, ABORT=2, UTILITY=3, EXECUTOR_FAILED=4)
        namespace = {"EngineCoreRequestType": kinds, "EngineCoreOutputs": lambda **kw: types.SimpleNamespace(**kw),
                     "EngineCoreOutput": lambda **kw: types.SimpleNamespace(**kw)}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), "core.py", "exec"), namespace)
        vals = {"chat.ttft": 2., "chat.tpot": .03, "code.e2el": 30., "summary.e2el": 20.}
        state = state_mod.ExperimentState()
        acknowledgement = state.initialize("run", vals, state_mod.slo_digest(vals))
        received, queued = [], []
        core = types.SimpleNamespace(scheduler=types.SimpleNamespace(r1_type_slo=state, r1_type_slo_enabled=True),
            output_queue=types.SimpleNamespace(put_nowait=queued.append), add_request=received.append)
        def send(kind, token):
            request = types.SimpleNamespace(r1_request_type=kind, r1_type_slo_instance=token, client_index=0, request_id="id")
            namespace["_handle_client_request"](core, kinds.ADD, request)
        send("chat", acknowledgement["instance_id"])
        self.assertEqual(len(received), 1)
        for kind, token in ((None, acknowledgement["instance_id"]), ("other", acknowledgement["instance_id"]), ("chat", "stale")):
            send(kind, token)
        self.assertEqual(len(received), 1)
        self.assertEqual(len(queued), 3)
        self.assertTrue(all(item[1].outputs[0].r1_error for item in queued))
        core.scheduler.r1_type_slo = state_mod.ExperimentState()
        send("chat", acknowledgement["instance_id"])
        self.assertEqual(len(queued), 4)
        core.scheduler.r1_type_slo = None
        send("chat", acknowledgement["instance_id"])
        self.assertEqual(len(queued), 5)
        core.scheduler.r1_type_slo_enabled = False
        send(None, None)
        self.assertEqual(len(received), 2)


    def test_core_error_is_request_scoped(self):
        tree = ast.parse((ROOT / "vllm/v1/engine/output_processor.py").read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "OutputProcessor")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "process_outputs")
        method.returns = None
        for arg in method.args.args: arg.annotation = None
        namespace = {"OutputProcessorOutput": lambda **kw: types.SimpleNamespace(**kw)}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), "output_processor.py", "exec"), namespace)
        delivered, aborted = [], []
        request = types.SimpleNamespace(queue=types.SimpleNamespace(put=delivered.append))
        proc = types.SimpleNamespace(request_states={"id": request},
             abort_requests=lambda ids: aborted.extend(ids),
             lora_states=types.SimpleNamespace(update_iteration_stats=lambda *args: None))
        output = types.SimpleNamespace(request_id="id", r1_error="stale configuration")
        result = namespace["process_outputs"](proc, [output])
        self.assertEqual(aborted, ["id"])
        self.assertEqual(len(delivered), 1)
        self.assertIsInstance(delivered[0], ValueError)
        self.assertEqual(str(delivered[0]), "stale configuration")
        self.assertEqual(result.request_outputs, [])

if __name__ == "__main__": unittest.main()
