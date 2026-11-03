import importlib.util
import unittest
from pathlib import Path

root = Path(__file__).resolve().parents[2]
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, root / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

state_module = load("r1_type_slo_state", "vllm/v1/core/sched/r1_type_slo_state.py")
client = load("r1_goodput_by_type", "benchmarks/r1_goodput_by_type.py")

class StateTests(unittest.TestCase):
    def setUp(self):
        self.values = client.parse_goodput_by_type(["chat.ttft:2000", "chat.tpot:30", "code.e2el:30000", "summary.e2el:20000"])
        self.digest = client.config_digest(self.values)
        self.state = state_module.ExperimentState()

    def test_digest_and_idempotence(self):
        self.assertEqual(state_module.slo_digest(state_module.normalize_slos(self.values)), self.digest)
        expected = {"run_id": "run-a", "config_sha256": self.digest}
        self.assertEqual(self.state.initialize("run-a", self.values, self.digest), expected)
        self.assertEqual(self.state.initialize("run-a", self.values, self.digest), expected)
        self.assertEqual(self.state.snapshot()["config_seconds"], self.values)
        self.state.snapshot()["config_seconds"]["chat.ttft"] = -1
        self.assertEqual(self.state.snapshot()["config_seconds"]["chat.ttft"], 2)

    def test_conflict_and_type_validation(self):
        with self.assertRaises(ValueError): self.state.validate_request_type("chat")
        with self.assertRaises(ValueError): self.state.initialize("run-a", self.values, "wrong")
        self.state.initialize("run-a", self.values, self.digest)
        for typ in ("chat", "code", "summary"):
            self.assertEqual(self.state.validate_request_type(typ), typ)
        for typ in (None, "other", ""):
            with self.assertRaises(ValueError): self.state.validate_request_type(typ)
        with self.assertRaises(ValueError): self.state.initialize("run-b", self.values, self.digest)
        other = dict(self.values, **{"chat.ttft": 3.0})
        with self.assertRaises(ValueError): self.state.initialize("run-a", other, client.config_digest(other))


class CoreStateBindingTests(unittest.TestCase):
    def test_core_utility_operates_on_scheduler_state(self):
        import ast
        import types
        path = root / "vllm/v1/engine/core.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "EngineCore")
        methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ("r1_initialize_type_slos", "r1_validate_type_request")]
        for method in methods:
            method.returns = None
            for arg in method.args.args:
                arg.annotation = None
        logger = types.SimpleNamespace(info=lambda *args: None)
        namespace = {"logger": logger}
        exec(compile(ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[])), str(path), "exec"), namespace)
        state = state_module.ExperimentState()
        core = types.SimpleNamespace(scheduler=types.SimpleNamespace(r1_type_slo=state))
        values = client.parse_goodput_by_type(["chat.ttft:2000", "chat.tpot:30", "code.e2el:30000", "summary.e2el:20000"])
        digest = client.config_digest(values)
        with self.assertRaises(ValueError): namespace["r1_validate_type_request"](core, "chat")
        self.assertEqual(namespace["r1_initialize_type_slos"](core, "run", values, digest), {"run_id": "run", "config_sha256": digest})
        self.assertEqual(state.snapshot()["config_seconds"], values)
        for kind in ("chat", "code", "summary"):
            self.assertEqual(namespace["r1_validate_type_request"](core, kind)["config_sha256"], digest)
        for kind in (None, "unknown"):
            with self.assertRaises(ValueError): namespace["r1_validate_type_request"](core, kind)
        with self.assertRaises(ValueError): namespace["r1_initialize_type_slos"](core, "another", values, digest)

if __name__ == "__main__":
    unittest.main()
