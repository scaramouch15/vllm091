# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU unit tests with isolated native definitions, not a server/import test.

Run with unittest to avoid the GPU-wide conftest. AST loading below isolates the
actual native interfaces/main from unavailable imports; production is untouched.
PyArrow and the optional HF test use real libraries and synthetic local records.
"""

import __future__
import argparse
import ast
import asyncio
import contextlib
import importlib.util
import io
import json
import logging
import os
import random
import sys
import tempfile
import types
import unittest
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

BENCHMARKS = Path(__file__).resolve().parents[2] / "benchmarks"


def compile_nodes(nodes, namespace, filename):
    exec(
        compile(
            ast.Module(body=nodes, type_ignores=[]),
            filename,
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        namespace,
    )


def load_isolated_dataset():
    native = types.ModuleType("benchmark_dataset")
    native.__dict__.update(
        ABC=ABC,
        abstractmethod=abstractmethod,
        dataclass=dataclass,
        random=random,
        json=json,
        logger=logging.getLogger(__name__),
    )
    tree = ast.parse(
        (BENCHMARKS / "benchmark_dataset.py").read_text(encoding="utf-8")
    )
    nodes = [
        n
        for n in tree.body
        if isinstance(n, (ast.ClassDef, ast.FunctionDef))
        and n.name
        in (
            "SampleRequest",
            "BenchmarkDataset",
            "ShareGPTDataset",
            "is_valid_sequence",
        )
    ]
    spec = importlib.util.spec_from_file_location(
        "mixed_workload_dataset",
        BENCHMARKS / "mixed_workload_dataset.py",
    )
    mixed = importlib.util.module_from_spec(spec)
    with patch.dict(
        sys.modules,
        {"benchmark_dataset": native, "mixed_workload_dataset": mixed},
    ):
        compile_nodes(nodes, native.__dict__, "isolated_native_dataset")
        spec.loader.exec_module(mixed)
    return native, mixed


NATIVE, MIXED = load_isolated_dataset()


class Tokenizer:
    name_or_path = "synthetic-character-tokenizer"
    model_max_length = 1  # Mixed must not turn this into a context filter.

    def __init__(self):
        self.template_calls = []
        self.token_calls = []

    def get_chat_template(self):
        return "<bos><user>content<end><assistant>"

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False and add_generation_prompt is True
        assert len(messages) == 1 and messages[0]["role"] == "user"
        self.template_calls.append(messages[0]["content"])
        return "<bos><user>\n" + messages[0]["content"] + "<end>\n<assistant>\n"

    def __call__(self, text, add_special_tokens=True):
        self.token_calls.append((text, add_special_tokens))
        return types.SimpleNamespace(
            input_ids=([1] if add_special_tokens else []) + list(text)
        )


def make_fixtures(root):
    chat = root / "lmsys" / "data"
    code = root / "code"
    summary = root / "cnn" / "3.0.0"
    for directory in (chat, code, summary):
        directory.mkdir(parents=True)

    def chat_row(
        sample_id, content, reference="CHAT_REFERENCE", language="English"
    ):
        return {
            "conversation_id": sample_id,
            "language": language,
            "conversation": [
                {"role": "user", "content": content},
                {"role": "assistant", "content": reference},
            ],
        }

    chat_rows = [
        chat_row(
            "c0",
            " \nExplain rain.\n\nKeep this line.\n ",
            "  CHAT_REFERENCE \n",
        ),
        chat_row("c1", "Tell me about tides."),
        chat_row("duplicate", "Explain rain.\n\nKeep this line."),
        chat_row("foreign", "Foreign request", language="Chinese"),
        {"conversation_id": "empty", "language": "English", "conversation": []},
        {
            "conversation_id": "no-reference",
            "language": "English",
            "conversation": [
                {"role": "user", "content": "No assistant reference"}
            ],
        },
        chat_row("blank-reference", "Blank assistant reference", " \n "),
    ]
    # Intentionally create shards out of lexical order.
    pq.write_table(
        pa.Table.from_pylist(chat_rows[3:]), chat / "train-00001.parquet"
    )
    pq.write_table(
        pa.Table.from_pylist(chat_rows[:3]), chat / "train-00000.parquet"
    )
    pq.write_table(
        pa.Table.from_pylist([chat_row("bad-language", "X", language=5)]),
        chat / "train-00002.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist([chat_row("bad-content", 5, reference=6)]),
        chat / "train-00003.parquet",
    )
    query = " \nFix this function:\n```python\ndef f():\n    return 1\n```\n "
    code_rows = [
        {
            "query": query,
            "resource": "fixture-a",
            "lang": " PyThOn ",
            "answer": "  CODE_REFERENCE \n",
        },
        {
            "query": "Implement a heap.",
            "resource": "fixture-b",
            "lang": "python",
            "answer": "CODE_REFERENCE",
        },
        {
            "query": query.strip(),
            "resource": "fixture-c",
            "lang": "python",
            "answer": "OTHER_REFERENCE",
        },
        {"query": "JavaScript", "resource": "fixture-a", "lang": "javascript"},
        {"query": " \n ", "resource": "fixture-a", "lang": "python"},
        {
            "query": "No answer available",
            "resource": "fixture-d",
            "lang": "python",
        },
        {
            "query": "Wrong answer type",
            "resource": "fixture-e",
            "lang": "python",
            "answer": 7,
        },
        {"query": 3, "resource": "fixture-f", "lang": "python"},
    ]
    with (code / "CodeFeedback-Filtered-Instruction.jsonl").open(
        "w", encoding="utf-8"
    ) as stream:
        for row in code_rows:
            stream.write(json.dumps(row) + "\n")
    summary_rows = [
        {
            "id": "s0",
            "article": " \nA town opened a library.\nIt has 500 books.\n ",
            "highlights": "  SUMMARY_REFERENCE \n",
        },
        {
            "id": "s1",
            "article": "The city restored a bridge.",
            "highlights": "SUMMARY_REFERENCE",
        },
        {
            "id": "duplicate",
            "article": "The city restored a bridge.",
            "highlights": "OTHER_REFERENCE",
        },
        {"id": "empty", "article": " \n ", "highlights": "REFERENCE"},
        {"id": "no-reference", "article": "No highlights available"},
        {
            "id": "null-reference",
            "article": "Null highlights",
            "highlights": None,
        },
    ]
    pq.write_table(
        pa.Table.from_pylist(summary_rows), summary / "validation-00000.parquet"
    )
    pq.write_table(
        pa.Table.from_pylist([{"id": "bad-article", "article": 3}]),
        summary / "validation-00001.parquet",
    )
    return {
        "chat_path": str(chat.parent),
        "code_path": str(code),
        "summary_path": str(summary.parent),
    }


class MixedDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.paths = make_fixtures(self.root)
        self.tokenizer = Tokenizer()

    def dataset(self, counts=(2, 2, 2), output_len=None, seed=42):
        return MIXED.MixedWorkloadDataset(
            **self.paths,
            **dict(zip((f"{t}_count" for t in MIXED.TASKS), counts)),
            output_len=output_len,
            random_seed=seed,
        )

    def test_mapping_reference_lengths_and_metadata(self):
        dataset = self.dataset()
        requests = dataset.sample(self.tokenizer, 6)
        by_id = {r.metadata["source_sample_id"]: r for r in requests}
        self.assertIn("Explain rain.\n\nKeep this line.", by_id["c0"].prompt)
        self.assertEqual(
            by_id["c0"].expected_output_len, len("  CHAT_REFERENCE \n") + 1
        )
        self.assertEqual(
            by_id["s0"].expected_output_len, len("  SUMMARY_REFERENCE \n") + 1
        )
        self.assertIn(
            MIXED.SUMMARY_PREFIX
            + "A town opened a library.\nIt has 500 books.",
            by_id["s0"].prompt,
        )
        code = next(
            r for r in requests if r.metadata.get("resource") == "fixture-a"
        )
        self.assertIn("def f():\n    return 1\n```", code.prompt)
        self.assertEqual(
            code.expected_output_len, len("  CODE_REFERENCE \n") + 1
        )
        self.assertEqual(code.metadata["lang"], " PyThOn ")
        self.assertEqual(code.metadata["source_row"], 1)
        self.assertEqual(len(self.tokenizer.template_calls), 6)
        for index, request in enumerate(requests):
            metadata = request.metadata
            self.assertNotIn("REFERENCE", request.prompt)
            self.assertIsInstance(request, NATIVE.SampleRequest)
            self.assertEqual(metadata["request_index"], index)
            self.assertEqual(metadata["prompt"], request.prompt)
            self.assertEqual(metadata["prompt_len"], len(request.prompt))
            self.assertEqual(
                metadata["max_tokens"], request.expected_output_len
            )
            self.assertEqual(
                metadata["prompt_hash"], MIXED._hash(request.prompt)
            )
            if metadata["task_type"] == "chat":
                self.assertEqual(
                    metadata["chat_review_status"], "unreviewed_candidate"
                )
        self.assertEqual(
            [dataset.stats[t]["read"] for t in MIXED.TASKS], [9, 8, 7]
        )
        self.assertEqual(
            [dataset.stats[t]["excluded"] for t in MIXED.TASKS], [6, 5, 4]
        )
        self.assertEqual(
            [dataset.stats[t]["duplicate"] for t in MIXED.TASKS], [1, 1, 1]
        )
        self.assertEqual(
            dataset.stats["chat"]["exclusion_reasons"]["invalid_type:language"],
            1,
        )
        self.assertIn("Chinese", dataset.stats["chat"]["language_values"])
        self.assertEqual(
            dataset.report()["output_length_mode"], "reference_tokens"
        )

    def test_explicit_override_accepts_missing_and_invalid_references(self):
        dataset = self.dataset(counts=(4, 4, 4), output_len=37)
        requests = dataset.sample(self.tokenizer, 12)
        self.assertTrue(all(r.expected_output_len == 37 for r in requests))
        self.assertEqual(len(self.tokenizer.token_calls), 12)
        self.assertTrue(
            all(not special for _, special in self.tokenizer.token_calls)
        )
        self.assertEqual(dataset.report()["output_length_mode"], "fixed")
        self.assertEqual(
            [dataset.stats[t]["excluded"] for t in MIXED.TASKS], [4, 3, 2]
        )

    def test_no_context_filter_or_truncation(self):
        path = (
            Path(self.paths["code_path"])
            / "CodeFeedback-Filtered-Instruction.jsonl"
        )
        text = "def f():\n    return '" + "x" * 3000 + "'"
        path.write_text(
            json.dumps({"query": text, "lang": "python", "resource": "x"})
            + "\n",
            encoding="utf-8",
        )
        request = self.dataset(counts=(0, 1, 0), output_len=5000).sample(
            self.tokenizer, 1
        )[0]
        self.assertIn(text, request.prompt)
        self.assertGreater(request.prompt_len, 2048)
        self.assertEqual(request.expected_output_len, 5000)

    def test_seed_locality_reservoir_sampling_and_shuffle(self):
        random.seed(827)
        np.random.seed(827)
        state, np_state = random.getstate(), np.random.get_state()
        first = self.dataset(counts=(1, 1, 1)).sample(self.tokenizer, 3)
        self.assertEqual(state, random.getstate())
        current = np.random.get_state()
        self.assertEqual(np_state[0], current[0])
        np.testing.assert_array_equal(np_state[1], current[1])
        self.assertEqual(np_state[2:], current[2:])
        self.assertEqual(
            first, self.dataset(counts=(1, 1, 1)).sample(self.tokenizer, 3)
        )
        alternatives = [
            self.dataset(counts=(1, 1, 1), seed=i).sample(self.tokenizer, 3)
            for i in range(5)
        ]
        self.assertTrue(any(first != other for other in alternatives))
        self.assertEqual(len({r.metadata["input_hash"] for r in first}), 3)

    def test_counts_zero_sources_and_shortage(self):
        dataset = MIXED.MixedWorkloadDataset(
            code_path=self.paths["code_path"], code_count=1
        )
        self.assertEqual(len(dataset.sample(self.tokenizer, 1)), 1)
        self.assertEqual(dataset.files["chat"], [])
        for count in (0, -1, 5):
            with self.assertRaisesRegex(ValueError, "num-prompts"):
                self.dataset().sample(self.tokenizer, count)
        with self.assertRaisesRegex(
            ValueError, "insufficient.*requested 3, available 2"
        ):
            self.dataset(counts=(3, 0, 0)).sample(self.tokenizer, 3)
        for value in (0, -1):
            with self.assertRaisesRegex(ValueError, "positive integer"):
                self.dataset(output_len=value)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            self.dataset(counts=(-1, 1, 1))

    def test_source_errors_and_type_conflicts(self):
        path = (
            Path(self.paths["code_path"])
            / "CodeFeedback-Filtered-Instruction.jsonl"
        )
        path.write_text('{"lang":"python","resource":"x"}\n', encoding="utf-8")
        with self.assertRaisesRegex(
            ValueError, "jsonl:1: missing required.*query"
        ):
            self.dataset(counts=(0, 1, 0)).sample(self.tokenizer, 1)
        path.write_text("not-json\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "jsonl:1: invalid JSON"):
            self.dataset(counts=(0, 1, 0)).sample(self.tokenizer, 1)
        path.write_text(
            json.dumps(
                {
                    "query": "Tell me about tides.",
                    "lang": "python",
                    "resource": "x",
                    "answer": "REF",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            ValueError, "input/type conflict: chat.*code"
        ):
            self.dataset(counts=(1, 1, 0)).sample(self.tokenizer, 2)
        summary = Path(self.paths["summary_path"]) / "3.0.0"
        wrong = summary.rename(summary.with_name("2.0.0"))
        dataset = self.dataset(counts=(0, 0, 1))
        dataset.paths["summary"] = str(wrong)
        with self.assertRaisesRegex(
            ValueError, "unsupported raw source layout"
        ):
            dataset.sample(self.tokenizer, 1)

    def test_conservative_whitespace_dedup_and_manifest(self):
        path = (
            Path(self.paths["code_path"])
            / "CodeFeedback-Filtered-Instruction.jsonl"
        )
        rows = [
            {
                "query": "def f():\n" + indent + "return 1",
                "lang": "python",
                "resource": "x",
            }
            for indent in ("    ", "        ")
        ]
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        dataset = self.dataset(counts=(0, 2, 0), output_len=10)
        requests = dataset.sample(self.tokenizer, 2)
        self.assertEqual(len(requests), 2)
        self.assertEqual(dataset.stats["code"]["duplicate"], 0)
        output = self.root / "manifest.json"
        dataset.export(
            str(output), tokenizer_id="synthetic", tokenizer_mode="auto"
        )
        payload = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(payload["requests"], [r.metadata for r in requests])
        self.assertEqual(payload["run"]["seed"], 42)
        self.assertEqual(payload["run"]["tokenizer_id"], "synthetic")
        with self.assertRaisesRegex(ValueError, "overwrite"):
            dataset.export(
                str(path), tokenizer_id="synthetic", tokenizer_mode="auto"
            )

    def test_empty_parquet_still_checks_required_columns(self):
        path = (
            Path(self.paths["summary_path"])
            / "3.0.0"
            / "validation-missing.parquet"
        )
        pq.write_table(pa.table({"id": pa.array([], type=pa.string())}), path)
        with self.assertRaisesRegex(
            ValueError, "validation-missing.parquet: missing required.*article"
        ):
            self.dataset(counts=(0, 0, 1)).sample(self.tokenizer, 1)

    def test_final_prompt_type_conflict_is_reported(self):
        path = Path(self.paths["chat_path"]) / "data" / "train-00004.parquet"
        pq.write_table(
            pa.Table.from_pylist(
                [
                    {
                        "conversation_id": "summary-shaped-chat",
                        "language": "English",
                        "conversation": [
                            {
                                "role": "user",
                                "content": MIXED.SUMMARY_PREFIX
                                + "The city restored a bridge.",
                            },
                            {"role": "assistant", "content": "REFERENCE"},
                        ],
                    }
                ]
            ),
            path,
        )
        with self.assertRaisesRegex(
            ValueError, "input/type conflict: chat.*summary"
        ):
            self.dataset(counts=(1, 0, 1)).sample(self.tokenizer, 2)

    def test_real_hf_tokenizer_template_and_bos(self):
        from tokenizers import Tokenizer as BackendTokenizer
        from tokenizers import models, pre_tokenizers, processors
        from transformers import PreTrainedTokenizerFast

        backend = BackendTokenizer(
            models.WordLevel(
                {
                    "<unk>": 0,
                    "<bos>": 1,
                    "<end>": 2,
                    "<user>": 3,
                    "<assistant>": 4,
                },
                unk_token="<unk>",
            )
        )
        backend.pre_tokenizer = pre_tokenizers.Whitespace()
        backend.post_processor = processors.TemplateProcessing(
            single="<bos> $A", special_tokens=[("<bos>", 1)]
        )
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend,
            unk_token="<unk>",
            bos_token="<bos>",
            eos_token="<end>",
            additional_special_tokens=["<user>", "<assistant>"],
        )
        tokenizer.chat_template = (
            "{{ bos_token }}<user>\n{{ messages[0]['content'] }}"
            "{{ eos_token }}\n"
            "{% if add_generation_prompt %}<assistant>\n{% endif %}"
        )
        with patch.object(
            tokenizer,
            "apply_chat_template",
            wraps=tokenizer.apply_chat_template,
        ) as render:
            requests = self.dataset().sample(tokenizer, 6)
            self.assertEqual(render.call_count, 6)
        for request in requests:
            ids = tokenizer(request.prompt, add_special_tokens=False).input_ids
            self.assertEqual(ids.count(1), 1)
            self.assertEqual(tokenizer(request.prompt).input_ids.count(1), 2)
            self.assertEqual(len(ids), request.prompt_len)
        with (
            patch.object(
                tokenizer,
                "get_chat_template",
                side_effect=ValueError("missing"),
            ),
            self.assertRaisesRegex(ValueError, "chat template"),
        ):
            self.dataset().sample(tokenizer, 6)

    def load_native_main(self):
        tree = ast.parse(
            (BENCHMARKS / "benchmark_serving.py").read_text(encoding="utf-8")
        )
        main = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "main"
        )
        captures = []

        async def benchmark(**kwargs):
            captures.append(kwargs)
            return {}

        tokenizer_calls = []

        def get_tokenizer(*args, **kwargs):
            tokenizer_calls.append((args, kwargs))
            return self.tokenizer

        namespace = {
            "argparse": argparse,
            "asyncio": asyncio,
            "np": np,
            "os": os,
            "random": random,
            "gc": types.SimpleNamespace(
                collect=lambda: None, freeze=lambda: None
            ),
            "FlexibleArgumentParser": argparse.ArgumentParser,
            "ASYNC_REQUEST_FUNCS": {
                "vllm": None,
                "openai": None,
                "openai-chat": None,
            },
            "OPENAI_COMPATIBLE_BACKENDS": ("vllm", "openai", "openai-chat"),
            "get_tokenizer": get_tokenizer,
            "benchmark": benchmark,
            "mixed_dataset_from_args": MIXED.mixed_dataset_from_args,
            "add_mixed_dataset_args": MIXED.add_mixed_dataset_args,
            "check_goodput_args": lambda args: {},
            "ShareGPTDataset": NATIVE.ShareGPTDataset,
        }
        compile_nodes(
            [main] + tree.body[-1].body[:-2], namespace, "isolated_native_main"
        )
        return namespace, captures, tokenizer_calls

    def command(self):
        return [
            "--model",
            "synthetic-instruct",
            "--tokenizer",
            str(self.root),
            "--dataset-name",
            "mixed",
            "--num-prompts",
            "6",
            "--seed",
            "42",
            *[
                value
                for task in MIXED.TASKS
                for value in (
                    f"--mixed-{task}-path",
                    self.paths[f"{task}_path"],
                    f"--mixed-{task}-count",
                    "2",
                )
            ],
        ]

    def test_same_entry_dry_run_before_server_calls(self):
        namespace, captures, calls = self.load_native_main()
        output = self.root / "dry-run.json"
        command = self.command() + [
            "--mixed-dry-run",
            "--mixed-export-path",
            str(output),
        ]
        with (
            contextlib.redirect_stdout(io.StringIO()) as console,
            patch(
                "socket.socket.connect",
                side_effect=AssertionError("no network"),
            ),
        ):
            namespace["main"](namespace["parser"].parse_args(command))
        self.assertEqual(captures, [])
        self.assertTrue(calls[0][1]["local_files_only"])
        self.assertIn("--- code final prompt", console.getvalue())
        self.assertIn('"max_tokens"', console.getvalue())
        self.assertEqual(
            len(json.loads(output.read_text(encoding="utf-8"))["requests"]), 6
        )
        args = namespace["parser"].parse_args(self.command())
        self.assertIsNone(args.mixed_output_len)
        args.tokenizer = "nonexistent-tokenizer-id"
        args.mixed_dry_run = True
        with (
            contextlib.redirect_stdout(io.StringIO()),
            self.assertRaisesRegex(ValueError, "local --tokenizer"),
        ):
            namespace["main"](args)

    def test_native_main_backends_sampling_body_and_nonmixed(self):
        namespace, captures, _ = self.load_native_main()
        for backend in ("vllm", "openai"):
            command = self.command() + [
                "--backend",
                backend,
                "--mixed-output-len",
                "37",
                "--temperature",
                "0.7",
                "--top-p",
                "0.9",
                "--ignore-eos",
            ]
            with contextlib.redirect_stdout(io.StringIO()):
                namespace["main"](namespace["parser"].parse_args(command))
            self.assertEqual(
                captures[-1]["extra_body"],
                {"temperature": 0.7, "top_p": 0.9, "add_special_tokens": False},
            )
            self.assertTrue(captures[-1]["ignore_eos"])
            self.assertTrue(
                all(
                    r.expected_output_len == 37
                    for r in captures[-1]["input_requests"]
                )
            )
        path = self.root / "sharegpt.json"
        path.write_text(
            json.dumps(
                [
                    {
                        "conversations": [
                            {"value": "Native input"},
                            {"value": "Native reference"},
                        ]
                    }
                ]
            ),
            encoding="utf-8",
        )
        with contextlib.redirect_stdout(io.StringIO()):
            namespace["main"](
                namespace["parser"].parse_args(
                    [
                        "--model",
                        "native",
                        "--dataset-path",
                        str(path),
                        "--num-prompts",
                        "2",
                        "--backend",
                        "openai-chat",
                        "--endpoint",
                        "/v1/chat/completions",
                    ]
                )
            )
        self.assertEqual(captures[-1]["extra_body"], {"temperature": 0.0})
        self.assertEqual(
            [r.prompt for r in captures[-1]["input_requests"]],
            ["Native input"] * 2,
        )
        for key, value in (
            ("backend", "openai-chat"),
            ("endpoint", "/v1/chat/completions"),
        ):
            args = namespace["parser"].parse_args(self.command())
            setattr(args, key, value)
            with self.assertRaisesRegex(ValueError, "mixed supports"):
                MIXED.mixed_dataset_from_args(args)
        tree = ast.parse(
            (BENCHMARKS / "backend_request_func.py").read_text(encoding="utf-8")
        )
        mapping = next(
            n.value
            for n in tree.body
            if isinstance(n, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "ASYNC_REQUEST_FUNCS"
                for t in n.targets
            )
        )
        functions = {
            key.value: value.id
            for key, value in zip(mapping.keys, mapping.values)
        }
        self.assertEqual(functions["vllm"], "async_request_openai_completions")
        self.assertEqual(functions["openai"], functions["vllm"])

    def test_native_probe_and_formal_request_inputs(self):
        import time

        serving = ast.parse(
            (BENCHMARKS / "benchmark_serving.py").read_text(encoding="utf-8")
        )
        backend = ast.parse(
            (BENCHMARKS / "backend_request_func.py").read_text(encoding="utf-8")
        )
        request_type = next(
            n
            for n in backend.body
            if isinstance(n, ast.ClassDef) and n.name == "RequestFuncInput"
        )
        inputs = []

        async def fake_sender(request_func_input, pbar=None):
            inputs.append(request_func_input)
            return types.SimpleNamespace(
                success=True,
                prompt_len=request_func_input.prompt_len,
                ttft=0.01,
                itl=[],
                generated_text="synthetic",
                error="",
            )

        metrics = types.SimpleNamespace(
            completed=6,
            total_input=1,
            total_output=12,
            request_throughput=1,
            output_throughput=1,
            total_token_throughput=1,
        )
        namespace = {
            "__name__": __name__,
            "dataclass": dataclass,
            "asyncio": asyncio,
            "np": np,
            "random": random,
            "time": time,
            "ASYNC_REQUEST_FUNCS": {"vllm": fake_sender},
            "calculate_metrics": lambda **kwargs: (metrics, [2] * 6),
        }
        nodes = [
            n
            for n in serving.body
            if isinstance(n, ast.AsyncFunctionDef)
            and n.name in ("get_request", "benchmark")
        ]
        compile_nodes(
            [request_type] + nodes, namespace, "isolated_probe_and_batch"
        )
        requests = self.dataset(output_len=37).sample(self.tokenizer, 6)
        with contextlib.redirect_stdout(io.StringIO()):
            asyncio.run(
                namespace["benchmark"](
                    backend="vllm",
                    api_url="http://unused/v1/completions",
                    base_url="http://unused",
                    model_id="synthetic",
                    model_name=None,
                    tokenizer=self.tokenizer,
                    input_requests=requests,
                    logprobs=None,
                    request_rate=float("inf"),
                    burstiness=1,
                    disable_tqdm=True,
                    profile=False,
                    selected_percentile_metrics=[],
                    selected_percentiles=[99],
                    ignore_eos=False,
                    goodput_config_dict={},
                    max_concurrency=2,
                    lora_modules=None,
                    extra_body={
                        "temperature": 0.7,
                        "add_special_tokens": False,
                    },
                )
            )
        self.assertEqual(len(inputs), 7)  # Native probe, then the six requests.
        self.assertEqual(inputs[0].prompt, requests[0].prompt)
        self.assertEqual(
            [r.prompt for r in inputs[1:]], [r.prompt for r in requests]
        )
        self.assertTrue(all(r.output_len == 37 for r in inputs))
        self.assertTrue(
            all(
                r.extra_body
                == {
                    "temperature": 0.7,
                    "add_special_tokens": False,
                }
                for r in inputs
            )
        )
        sender = next(
            n
            for n in backend.body
            if isinstance(n, ast.AsyncFunctionDef)
            and n.name == "async_request_openai_completions"
        )
        payload = next(
            n.value
            for n in ast.walk(sender)
            if isinstance(n, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "payload" for t in n.targets
            )
        )
        max_tokens = next(
            v
            for k, v in zip(payload.keys, payload.values)
            if isinstance(k, ast.Constant) and k.value == "max_tokens"
        )
        self.assertEqual(
            ast.unparse(max_tokens), "request_func_input.output_len"
        )


if __name__ == "__main__":
    unittest.main()
