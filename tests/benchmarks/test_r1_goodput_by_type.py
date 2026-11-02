import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("r1_goodput_by_type", Path(__file__).resolve().parents[2] / "benchmarks" / "r1_goodput_by_type.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class ConfigTests(unittest.TestCase):
    def test_normalization_and_digest(self):
        entries = ["chat.ttft:2000", "chat.tpot:30", "code.e2el:30000", "summary.e2el:20000"]
        result = m.parse_goodput_by_type(entries)
        self.assertEqual(result, {"chat.ttft": 2.0, "chat.tpot": .03, "code.e2el": 30.0, "summary.e2el": 20.0})
        self.assertEqual(m.config_digest(result), m.config_digest(m.parse_goodput_by_type(reversed(entries))))

    def test_reject_invalid_values(self):
        valid = ["chat.ttft:2000", "chat.tpot:30", "code.e2el:30000", "summary.e2el:20000"]
        for entries in (valid[:-1], valid + [valid[0]], valid + ["other.ttft:1"], valid + ["chat.e2el:1"], ["chat.ttft:nan"] + valid[1:], ["chat.ttft:inf"] + valid[1:], ["chat.ttft:0"] + valid[1:], ["chat.ttft:-1"] + valid[1:]):
            with self.subTest(entries=entries), self.assertRaises(ValueError):
                m.parse_goodput_by_type(entries)


class EvaluationTests(unittest.TestCase):
    def test_formal_denominators_and_short_output(self):
        config = m.parse_goodput_by_type(["chat.ttft:2000", "chat.tpot:30", "code.e2el:30000", "summary.e2el:20000"])
        attempts = [{"request_id": "c", "task_type": "chat", "phase": "formal"}, {"request_id": "d", "task_type": "code", "phase": "formal"}, {"request_id": "s", "task_type": "summary", "phase": "formal"}, {"request_id": "w", "task_type": "chat", "phase": "warmup"}]
        results = {"c": {"success": True, "ttft_s": 2.0, "latency_s": 2.0, "output_tokens": 1}, "d": {"success": True, "latency_s": 30.0}, "w": {"success": True}}
        result = m.evaluate(config, attempts, results, 10.0)
        self.assertEqual(result["overall"]["attained"], 2)
        self.assertEqual(result["overall"]["attempted"], 3)
        self.assertEqual(result["types"]["summary"]["attainment_rate"], 0)
        self.assertAlmostEqual(sum(g["request_goodput"] for g in result["types"].values()), result["overall"]["request_goodput"])


class IntegrationContractTests(unittest.TestCase):
    def test_offline_round_trip(self):
        import tempfile
        config = m.parse_goodput_by_type(["chat.ttft:2000", "chat.tpot:30", "code.e2el:30000", "summary.e2el:20000"])
        attempts = [{"request_id": "1", "task_type": "chat", "phase": "formal"}, {"request_id": "2", "task_type": "code", "phase": "formal"}]
        results = {"1": {"success": True, "ttft_s": 1, "latency_s": 2, "output_tokens": 101}}
        summary = m.evaluate(config, attempts, results, 5)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "records.json")
            m.write_evaluation(path, config, attempts, results, 5, summary)
            self.assertEqual(summary, m.load_and_evaluate(path))
            with self.assertRaises(FileExistsError):
                m.write_evaluation(path, config, attempts, results, 5, summary)

    def test_chat_both_conditions_and_invalid_values(self):
        config = m.parse_goodput_by_type(["chat.ttft:2000", "chat.tpot:30", "code.e2el:30000", "summary.e2el:20000"])
        attempts = [{"request_id": str(i), "task_type": "chat", "phase": "formal"} for i in range(4)]
        results = {"0": {"success": True, "ttft_s": 2, "latency_s": 2.03, "output_tokens": 2},
                   "1": {"success": True, "ttft_s": 2.1, "latency_s": 2.1, "output_tokens": 1},
                   "2": {"success": True, "ttft_s": 1, "latency_s": 1.1, "output_tokens": 2},
                   "3": {"success": True, "ttft_s": float("nan"), "latency_s": 1, "output_tokens": 1}}
        # Use exact arithmetic at the boundary in a separate zero-duration TPOT case.
        results["0"] = {"success": True, "ttft_s": 2, "latency_s": 2, "output_tokens": 1}
        self.assertEqual(m.evaluate(config, attempts, results, 1)["types"]["chat"]["attained"], 1)

    def test_cli_contract_and_native_conflict(self):
        values = ["chat.ttft:2", "chat.tpot:3", "code.e2el:4", "summary.e2el:5"]
        self.assertIsNone(m.check_type_args(["ttft:2"], None, "random", None))
        self.assertEqual(m.check_type_args(None, values, "mixed", "record.json")["chat.ttft"], .002)
        for args in ((["ttft:2"], values, "mixed", "record.json"), (None, values, "random", "record.json"), (None, values, "mixed", None)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                m.check_type_args(*args)

    def test_duplicate_and_unknown_id_rejected(self):
        config = m.parse_goodput_by_type(["chat.ttft:1", "chat.tpot:1", "code.e2el:1", "summary.e2el:1"])
        attempt = {"request_id": "a", "task_type": "chat", "phase": "formal"}
        with self.assertRaises(ValueError):
            m.evaluate(config, [attempt, attempt], {}, 1)
        with self.assertRaises(ValueError):
            m.evaluate(config, [attempt], {"unknown": {}}, 1)

    def test_benchmark_function_and_offline_use_same_records(self):
        import ast
        import asyncio
        import contextlib
        import io
        import tempfile
        import time
        from types import SimpleNamespace
        source = (Path(__file__).resolve().parents[2] / "benchmarks" / "benchmark_serving.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        node = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "benchmark")
        node.returns = None
        for arg in node.args.args + node.args.kwonlyargs:
            arg.annotation = None
        async def request_func(request_func_input, **kwargs):
            return SimpleNamespace(success=True, error="", latency=2.0, ttft=1.0, output_tokens=101, itl=[], prompt_len=1, generated_text="x")
        async def request_iter(requests, *unused):
            for request in requests:
                yield request
        class Input:
            def __init__(self, **kwargs): self.__dict__.update(kwargs)
        class Metrics:
            completed = total_input = total_output = 1
            request_throughput = request_goodput = output_throughput = total_token_throughput = 1.0
        namespace = {"asyncio": asyncio, "time": time, "ASYNC_REQUEST_FUNCS": {"fake": request_func},
                     "RequestFuncInput": Input, "get_request": request_iter,
                     "calculate_metrics": lambda **kwargs: (Metrics(), [101] * len(kwargs["outputs"])),
                     "evaluate": m.evaluate, "write_evaluation": m.write_evaluation, "os": __import__("os")}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), "benchmark_serving.py", "exec"), namespace)
        config = m.parse_goodput_by_type(["chat.ttft:2000", "chat.tpot:30", "code.e2el:30000", "summary.e2el:20000"])
        requests = [SimpleNamespace(prompt="same", prompt_len=1, expected_output_len=10, multi_modal_data=None,
                                    metadata={"request_index": i, "task_type": t}) for i, t in enumerate(("chat", "code", "summary"))]
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            path = str(Path(directory) / "records.json")
            result = asyncio.run(namespace["benchmark"](backend="fake", api_url="", base_url="", model_id="m", model_name="m", tokenizer=None,
                input_requests=requests, logprobs=None, request_rate=float("inf"), burstiness=1, disable_tqdm=True, profile=False,
                selected_percentile_metrics=[], selected_percentiles=[], ignore_eos=False, goodput_config_dict={}, max_concurrency=None,
                lora_modules=None, extra_body=None, type_slos=config, type_output=path))
            self.assertEqual(result["goodput_by_type"], m.load_and_evaluate(path))
            self.assertEqual(result["goodput_by_type"]["overall"]["attained"], 3)

    def test_native_goodput_functions_unchanged(self):
        import ast
        import argparse
        from types import SimpleNamespace
        source = (Path(__file__).resolve().parents[2] / "benchmarks" / "benchmark_serving.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("check_goodput_args", "parse_goodput")]
        namespace = {"argparse": argparse}
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), "benchmark_serving.py", "exec"), namespace)
        self.assertEqual(namespace["check_goodput_args"](SimpleNamespace(goodput=["ttft:100", "tpot:5"])), {"ttft": 100.0, "tpot": 5.0})
        self.assertEqual(namespace["check_goodput_args"](SimpleNamespace(goodput=None)), {})


if __name__ == "__main__":
    unittest.main()
