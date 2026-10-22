# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stdlib CPU tests of real source definitions, isolated from GPU imports.

No HTTP listener, model, tokenizer download or benchmark is started. AST loading
executes the actual native methods with resource/transport stubs; it is not a
full installed-vLLM, msgpack/ZMQ, or GPU integration test.
"""

import __future__
import ast
import asyncio
import contextlib
import enum
import importlib.util
import io
import json
import logging
import os
import random
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import types
import unittest
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "vllm/v1/metrics/request_collector.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


C = load_module("isolated_r1_collector", SOURCE)
NS = types.SimpleNamespace


class NoIOLoop(asyncio.SelectorEventLoop):
    """CPU task scheduling without sockets in restricted Windows."""

    def _make_self_pipe(self):
        pass

    def _close_self_pipe(self):
        pass


def run_async(coro):
    def factory():
        loop = NoIOLoop()
        loop._selector.select = lambda timeout: []
        return loop

    loop = factory()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    finally:
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(
                asyncio.gather(*pending, return_exceptions=True)
            )
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
        asyncio.set_event_loop(None)


def compile_nodes(nodes, namespace, filename="isolated_native"):
    exec(
        compile(
            ast.Module(body=nodes, type_ignores=[]),
            filename,
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        namespace,
    )


def load_defs(path, names, namespace=None, source=None):
    namespace = namespace if namespace is not None else {}
    namespace.setdefault("__name__", __name__)
    tree = ast.parse(
        source if source is not None else (ROOT / path).read_text("utf-8")
    )
    nodes = [
        n
        for n in tree.body
        if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and n.name in names
    ]
    compile_nodes(nodes, namespace, path)
    return namespace


N = {
    "enum": enum,
    "time": time,
    "dataclass": dataclass,
    "field": field,
    "FINISH_REASON_STRINGS": ("stop", "length", "abort"),
}
load_defs(
    "vllm/v1/engine/__init__.py", ["FinishReason", "EngineCoreEventType"], N
)
load_defs(
    "vllm/v1/metrics/stats.py",
    [
        "RequestStateStats",
        "FinishedRequestStats",
        "IterationStats",
        "LoRARequestStates",
    ],
    N,
)
ENGINE_EVENTS = types.ModuleType("vllm.v1.engine")
ENGINE_EVENTS.EngineCoreEventType = N["EngineCoreEventType"]
N.update(
    {
        "ConstantList": list,
        "is_list_of": lambda value, kind: True,
        "StructuredOutputRequest": lambda sampling_params: NS(),
        "EngineCoreEvent": NS(
            new_event=lambda kind, timestamp=None: NS(
                type=kind,
                timestamp=time.monotonic() if timestamp is None else timestamp,
            )
        ),
    }
)
load_defs("vllm/v1/request.py", ["Request", "RequestStatus"], N)


def params(**kwargs):
    values = dict(
        max_tokens=8,
        n=1,
        guided_decoding=None,
        extra_args=None,
        logprobs=None,
        prompt_logprobs=None,
    )
    values.update(kwargs)
    return NS(**values)


def request(request_id, count=3, status=None):
    obj = N["Request"](
        request_id, list(range(count)), None, None, None, params(), None
    )
    if status is not None:
        obj.status = status
    return obj


class FakeKV:
    def __init__(self):
        self.allocations = []
        self.cache_queries = []
        self.fail = set()
        self.usage_reads = 0
        self.allocated = 0

    @property
    def usage(self):
        self.usage_reads += 1
        return self.allocated / 100

    def allocate_slots(self, req, tokens, *args, **kwargs):
        self.allocations.append((req.request_id, tokens))
        if req.request_id in self.fail:
            return None
        self.allocated += tokens
        return NS(get_block_ids=lambda: ([1],))

    def get_computed_blocks(self, req):
        self.cache_queries.append(req.request_id)
        return NS(), 0

    def get_block_ids(self, request_id):
        return ([1],)

    def get_num_common_prefix_blocks(self, *args):
        return [0]

    def take_events(self):
        return []

    def free(self, req):
        pass

    def free_block_hashes(self, req):
        pass


def scheduler_namespace(source=None):
    namespace = dict(
        N,
        defaultdict=defaultdict,
        deque=deque,
        SchedulerInterface=object,
        MULTIMODAL_REGISTRY=None,
        logger=logging.getLogger(__name__),
        CollectionConfig=C.CollectionConfig,
        CoreCollector=C.CoreCollector,
        NewRequestData=NS(
            from_request=lambda req, blocks: NS(
                req_id=req.request_id,
                num_computed_tokens=req.num_computed_tokens,
            )
        ),
        CachedRequestData=NS(
            from_request=lambda req, resumed, tokens, blocks: NS(
                req_id=req.request_id, resumed_from_preemption=resumed
            )
        ),
        SchedulerOutput=lambda **kwargs: NS(**kwargs),
    )
    return load_defs(
        "vllm/v1/core/sched/scheduler.py", ["Scheduler"], namespace, source
    )


def scheduler(running=(), waiting=(), collector=None, cls=None):
    cls = cls or scheduler_namespace()["Scheduler"]
    obj = cls.__new__(cls)
    obj.running, obj.waiting = list(running), deque(waiting)
    obj.requests = {r.request_id: r for r in [*running, *waiting]}
    obj.request_collector = collector
    obj.log_stats = True
    obj.max_num_scheduled_tokens = 20
    obj.max_num_running_reqs = 8
    obj.max_model_len = 100
    obj.max_num_encoder_input_tokens = 0
    obj.scheduler_config = NS(long_prefill_token_threshold=0)
    obj.lora_config = obj.connector = obj.kv_event_publisher = None
    obj.num_lookahead_tokens = 0
    obj.kv_cache_manager = FakeKV()
    obj.kv_cache_config = NS(kv_cache_groups=[None])
    obj.structured_output_manager = NS(grammar_bitmask=lambda *args: None)
    obj.encoder_cache_manager = NS(
        get_freed_ids=lambda: [], free=lambda req: None
    )
    obj.finished_req_ids = set()
    obj.finished_req_ids_dict = None
    obj._cached_reqs_data = {}
    obj._connector_finished = lambda req: (False, None)
    return obj


def output_namespace():
    namespace = dict(
        N,
        asyncio=asyncio,
        ParentRequest=NS(observe_finished_request=lambda *args: None),
    )
    return load_defs(
        "vllm/v1/engine/output_processor.py",
        ["OutputProcessor", "OutputProcessorOutput"],
        namespace,
    )


def state(request_id, stop_string=None):
    return NS(
        request_id=request_id,
        stats=N["RequestStateStats"](arrival_time=10),
        prompt_token_ids=[1, 2, 3],
        prompt_len=3,
        max_tokens_param=8,
        is_prefilling=True,
        parent_req=None,
        lora_name=None,
        detokenizer=NS(update=lambda *args: stop_string),
        logprobs_processor=NS(update_from_output=lambda output: None),
        queue=None,
        make_request_output=lambda *args: None,
    )


def engine_output(request_id, reason=None, events=None):
    return NS(
        request_id=request_id,
        new_token_ids=[7],
        finish_reason=reason,
        finished=reason is not None,
        stop_reason=None,
        kv_transfer_params=None,
        num_cached_tokens=0,
        events=events,
    )


class CollectorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="r1-test-", dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = C.CollectionConfig("unit-run", self.root)
        self.addCleanup(patch.stopall)

    def writer(self, role="client", config=None, metadata=None):
        writer = C.RecordWriter(config or self.config, role, metadata)
        self.addCleanup(writer.close)
        return writer

    def records(self, writer):
        writer.close()
        return [
            json.loads(line)
            for line in writer.path.read_text("utf-8").splitlines()
        ]

    def test_config_off_and_invalid(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(C, "RecordWriter") as ctor,
        ):
            self.assertIsNone(C.CollectionConfig.from_env())
            ctor.assert_not_called()
        for env in (
            {"VLLM_R1_COLLECT": "bad"},
            {"VLLM_R1_COLLECT": "1"},
            {"VLLM_R1_COLLECT": "1", "VLLM_R1_RUN_ID": "../bad"},
        ):
            with (
                patch.dict(os.environ, env, clear=True),
                self.assertRaises(ValueError),
            ):
                C.CollectionConfig.from_env()
        with self.assertRaisesRegex(ValueError, "log_stats"):
            self.config.require_stats(False)

    def test_client_scope_and_unique_ids_without_global_rng(self):
        for changes in (
            {"backend": "openai"},
            {"endpoint": "/v1/chat/completions"},
            {"profile": True},
            {"dataset_name": "random"},
        ):
            kwargs = dict(
                backend="vllm",
                endpoint="/v1/completions",
                profile=False,
                dataset_name="mixed",
                lora_modules=None,
            )
            kwargs.update(changes)
            with self.assertRaises(ValueError):
                C.ClientCollector(self.config, **kwargs)
        writer = self.writer()
        sample = dict(
            prompt="<bos>\n  def f():\n    pass\n",
            request_index=0,
            source_sample_id="source-1",
            task_type="code",
        )
        before = random.getstate()
        ids = []
        for phase in ("probe", "main", "main"):
            attempt = C.ClientAttempt(writer, sample, phase)
            ident = attempt.start(
                {"prompt": sample["prompt"]}, "http://local/v1/completions"
            )
            ids.append(ident)
            self.config.validate_engine_request(f"cmpl-{ident}-0", params())
            with self.assertRaisesRegex(ValueError, "reused"):
                attempt.start(
                    {"prompt": sample["prompt"]}, "http://local/v1/completions"
                )
        self.assertEqual(before, random.getstate())
        self.assertEqual(len(set(ids)), 3)
        sample["task_type"] = "mutated"
        rows = [
            r for r in self.records(writer) if r["record_type"] == "attempt"
        ]
        self.assertEqual([r["phase"] for r in rows], ["probe", "main", "main"])
        self.assertTrue(all(r["sample"]["task_type"] == "code" for r in rows))
        self.assertEqual(rows[0]["engine_request_id"], f"cmpl-{ids[0]}-0")

    def test_reject_multi_prompt_beam_and_run_mismatch(self):
        writer = self.writer()
        for extra in (
            {"n": 2},
            {"use_beam_search": True},
            {"best_of": 2},
            {"prompt": ["p"]},
        ):
            ctx = C.ClientAttempt(writer, {"prompt": "p"}, "main")
            with self.assertRaises(ValueError):
                ctx.start(
                    dict({"prompt": "p"}, **extra),
                    "http://local/v1/completions",
                )
        for ident, p in (
            ("cmpl-r1.other." + "a" * 32 + "-0", params()),
            ("cmpl-r1.unit-run." + "a" * 32 + "-1", params()),
            ("cmpl-r1.unit-run." + "a" * 32 + "-0", params(n=2)),
        ):
            with self.assertRaises(ValueError):
                self.config.validate_engine_request(ident, p)

    def test_queue_full_nonblocking_and_normal_close(self):
        gate, entered = threading.Event(), threading.Event()
        original = C.RecordWriter._write

        def paused(writer, record):
            if record["record_type"] == "header":
                entered.set()
                gate.wait(2)
            original(writer, record)

        with patch.object(C.RecordWriter, "_write", paused):
            writer = self.writer(
                config=C.CollectionConfig("unit-run", self.root, queue_size=1)
            )
            self.assertTrue(entered.wait(1))
            self.assertTrue(
                writer.emit("diagnostic", {"engine_request_id": "x"})
            )
            with self.assertLogs(C.logger, logging.ERROR) as logs:
                self.assertFalse(
                    writer.emit("diagnostic", {"engine_request_id": "y"})
                )
                self.assertFalse(
                    writer.emit("diagnostic", {"engine_request_id": "z"})
                )
            self.assertEqual(len(logs.records), 1)
            gate.set()
            self.assertFalse(writer.close())
        footer = self.records(writer)[-1]
        self.assertEqual(footer["metrics"]["queue_full"], 2)
        self.assertTrue(footer["clean_shutdown"])
        self.assertFalse(footer["records_complete"])

    def test_writer_failure_visible_without_repeated_logs(self):
        gate, entered = threading.Event(), threading.Event()

        def fail(writer, record):
            entered.set()
            gate.wait(2)
            raise OSError("synthetic disk failure")

        with patch.object(C.RecordWriter, "_write", fail):
            writer = self.writer()
            self.assertTrue(entered.wait(1))
            with self.assertLogs(C.logger, logging.ERROR) as logs:
                gate.set()
                self.assertFalse(writer.close())
                writer.emit("diagnostic", {"engine_request_id": "x"})
                writer.emit("diagnostic", {"engine_request_id": "y"})
            self.assertEqual(
                sum("write)" in r.getMessage() for r in logs.records), 1
            )
        report = C.merge_records(self.root / "unit-run")
        self.assertFalse(report["collection_complete"])
        self.assertIn(
            "incomplete_or_unconfirmed_close", report["files"][0]["status"]
        )

    def test_serialization_failure_and_idempotent_close(self):
        writer = self.writer()
        with self.assertLogs(C.logger, logging.ERROR):
            writer.emit("bad", {"unsupported": object()})
            self.assertFalse(writer.close())
        self.assertFalse(writer.close())
        footer = self.records(writer)[-1]
        self.assertEqual(footer["metrics"]["serialization_errors"], 1)

    def test_footer_and_overhead_counters(self):
        writer = self.writer()
        writer.emit(
            "diagnostic", {"engine_request_id": "x"}, time.perf_counter_ns()
        )
        self.assertTrue(writer.close())
        self.assertTrue(writer.close())
        footer = self.records(writer)[-1]
        self.assertTrue(footer["records_complete"])
        for key in (
            "record_build_ns",
            "enqueue_ns",
            "background_serialize_ns",
            "background_write_ns",
            "background_flush_ns",
            "shutdown_drain_ns",
        ):
            self.assertGreaterEqual(footer["metrics"][key], 0)

    def test_scheduler_first_shared_snapshot_and_resume(self):
        core = C.CoreCollector(self.config)
        self.addCleanup(core.close)
        running = request("running", 6, N["RequestStatus"].RUNNING)
        running.num_computed_tokens = 2
        one, two = request("one"), request("two")
        obj = scheduler([running], [one, two], core)
        obj.schedule()
        self.assertEqual(obj.kv_cache_manager.usage_reads, 1)
        self.assertEqual(
            obj.kv_cache_manager.allocations,
            [("running", 4), ("one", 3), ("two", 3)],
        )
        self.assertEqual(obj.kv_cache_manager.cache_queries, ["one", "two"])
        obj.running.remove(one)
        one.status = N["RequestStatus"].PREEMPTED
        one.num_computed_tokens = 0
        obj.waiting.appendleft(one)
        obj.schedule()
        # Protect restored WAITING status after an async-KV transition, too.
        obj.running.remove(one)
        one.status = N["RequestStatus"].WAITING
        one.num_computed_tokens = 0
        obj.waiting.appendleft(one)
        obj.schedule()
        running.num_computed_tokens = 999
        core.close()
        rows = [
            r
            for r in self.records(core.writer)
            if r["record_type"] == "first_admission"
        ]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["snapshot"], rows[1]["snapshot"])
        snap = rows[0]["snapshot"]
        self.assertEqual(snap["running_count"], 1)
        self.assertEqual(snap["running_computed_tokens"], 2)
        self.assertEqual(snap["running_remaining_prompt_tokens"], 4)
        self.assertEqual(snap["token_budget"], 16)
        self.assertEqual(snap["kv_usage"], 0.04)
        self.assertNotIn("running_request_ids", snap)
        one.status = N["RequestStatus"].FINISHED_STOPPED
        obj._free_request(one)
        self.assertNotIn("one", core.first_admitted)

    def test_failed_allocation_not_admitted(self):
        core = C.CoreCollector(self.config)
        self.addCleanup(core.close)
        obj = scheduler(waiting=[request("failed")], collector=core)
        obj.kv_cache_manager.fail.add("failed")
        obj.schedule()
        self.assertFalse(core.first_admitted)
        self.assertEqual(obj.kv_cache_manager.allocations, [("failed", 3)])
        rows = self.records(core.writer)
        self.assertFalse(
            any(r["record_type"] == "first_admission" for r in rows)
        )

    def test_snapshot_gates_and_disabled_native_equivalence(self):
        core = C.CoreCollector(self.config)
        self.addCleanup(core.close)
        for budget, slots, waiting in (
            (0, 8, [request("w")]),
            (20, 0, [request("w")]),
            (20, 8, []),
        ):
            obj = scheduler(waiting=waiting, collector=core)
            obj.max_num_scheduled_tokens, obj.max_num_running_reqs = (
                budget,
                slots,
            )
            obj.schedule()
            self.assertEqual(obj.kv_cache_manager.usage_reads, 0)
        r = request("preempt", status=N["RequestStatus"].RUNNING)
        obj = scheduler([r], [request("w")], core)
        obj.kv_cache_manager.fail.add("preempt")
        obj.schedule()
        self.assertEqual(obj.kv_cache_manager.usage_reads, 0)
        baseline = subprocess.run(
            ["git", "show", "v0.9.1:vllm/v1/core/sched/scheduler.py"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        native_cls = scheduler_namespace(baseline)["Scheduler"]
        outputs = []
        for cls in (native_cls, scheduler_namespace()["Scheduler"]):
            obj = scheduler(
                [request("r", 5, N["RequestStatus"].RUNNING)],
                [request("a"), request("b")],
                cls=cls,
            )
            with (
                patch.object(
                    C.uuid, "uuid4", side_effect=AssertionError("disabled ID")
                ),
                patch.object(time, "monotonic", return_value=100),
            ):
                out = obj.schedule()
            self.assertEqual(obj.kv_cache_manager.usage_reads, 0)
            outputs.append(
                (
                    vars(out),
                    [r.request_id for r in obj.running],
                    [r.num_computed_tokens for r in obj.running],
                    obj.kv_cache_manager.allocations,
                    obj.kv_cache_manager.cache_queries,
                )
            )
        self.assertEqual(outputs[0], outputs[1])

    def process_finish(self, reason, stop_string=None, events=True):
        collector = C.FrontendCollector(self.config)
        self.addCleanup(collector.close)
        namespace = output_namespace()
        processor = namespace["OutputProcessor"](None, True, collector)
        req = state("engine-id", stop_string)
        processor.request_states[req.request_id] = req
        native_events = (
            [
                NS(type=N["EngineCoreEventType"].QUEUED, timestamp=11),
                NS(type=N["EngineCoreEventType"].SCHEDULED, timestamp=12),
            ]
            if events
            else None
        )
        stats = N["IterationStats"]()
        with patch.dict(sys.modules, {"vllm.v1.engine": ENGINE_EVENTS}):
            result = processor.process_outputs(
                [engine_output(req.request_id, reason, native_events)],
                15,
                stats,
            )
        return collector, processor, req, stats, result

    def test_finished_native_stats_stop_and_length(self):
        for reason in (N["FinishReason"].STOP, N["FinishReason"].LENGTH):
            collector, processor, req, stats, result = self.process_finish(
                reason
            )
            row = next(
                r
                for r in self.records(collector.writer)
                if r["record_type"] == "finished"
            )
            self.assertEqual(
                row["native_stats"]["inference_time"],
                stats.finished_requests[0].inference_time,
            )
            self.assertEqual(row["inference_time_label"], 3)
            self.assertEqual(row["native_stats"]["queued_time"], 1)
            self.assertEqual(row["native_stats"]["decode_time"], 0)
            self.assertEqual(row["native_stats"]["num_generation_tokens"], 1)
            self.assertEqual(row["timestamps"]["last_token_ts"], 15)
            self.assertEqual(
                row["length_capped"], reason == N["FinishReason"].LENGTH
            )
            self.assertFalse(processor.request_states)

    def test_stop_string_is_not_client_cancel(self):
        collector, processor, req, stats, result = self.process_finish(
            None, "STOP"
        )
        self.assertEqual(result.reqs_to_abort, ["engine-id"])
        self.assertEqual(processor.abort_requests(["engine-id"]), [])
        rows = self.records(collector.writer)
        finished = [r for r in rows if r["record_type"] == "finished"]
        self.assertEqual(finished[0]["terminal_origin"], "frontend_stop_string")
        self.assertFalse(any(r["record_type"] == "diagnostic" for r in rows))

    def test_missing_stats_and_cancellation_are_not_zero_labels(self):
        collector, processor, req, stats, result = self.process_finish(
            N["FinishReason"].STOP, events=False
        )
        row = next(
            r
            for r in self.records(collector.writer)
            if r["record_type"] == "finished"
        )
        self.assertIsNone(row["inference_time_label"])
        self.assertIsNone(row["timestamps"]["scheduled_ts"])
        second = C.FrontendCollector(self.config)
        self.addCleanup(second.close)
        second.finished(state("missing"), None, "engine")
        processor = output_namespace()["OutputProcessor"](None, True, second)
        req = state("cancelled")
        processor.request_states[req.request_id] = req
        self.assertEqual(
            processor.abort_requests([req.request_id]), [req.request_id]
        )
        rows = self.records(second.writer)
        self.assertTrue(
            all(
                r["inference_time_label"] is None
                for r in rows
                if r["record_type"] in ("finished", "diagnostic")
            )
        )
        self.assertFalse(
            any(
                r["record_type"] == "finished"
                and r["engine_request_id"] == "cancelled"
                for r in rows
            )
        )

    def test_frontend_and_core_explicit_shutdown_paths(self):
        namespace = dict(
            N,
            EngineClient=object,
            UsageContext=NS(ENGINE_CONTEXT=0),
            MULTIMODAL_REGISTRY=None,
            shutdown_prometheus=lambda: None,
        )
        load_defs("vllm/v1/engine/async_llm.py", ["AsyncLLM"], namespace)
        cls = namespace["AsyncLLM"]
        obj = cls.__new__(cls)
        front = C.FrontendCollector(self.config)
        self.addCleanup(front.close)
        obj.output_processor = NS(
            request_collector=front,
            request_states={"pending": state("pending")},
        )
        obj.engine_core = NS(shutdown=Mock())
        obj.output_handler = NS(cancel=Mock())
        obj.shutdown()
        obj.shutdown()
        self.assertFalse(front.writer._thread.is_alive())
        self.assertTrue(self.records(front.writer)[-1]["clean_shutdown"])
        core_namespace = dict(N, MULTIMODAL_REGISTRY=None)
        load_defs("vllm/v1/engine/core.py", ["EngineCore"], core_namespace)
        core = core_namespace["EngineCore"].__new__(
            core_namespace["EngineCore"]
        )
        order = []
        core.scheduler = NS(
            request_collector=NS(close=lambda: order.append("collector")),
            shutdown=lambda: order.append("scheduler"),
        )
        core.model_executor = NS(shutdown=lambda: order.append("model"))
        core.structured_output_manager = NS(
            clear_backend=lambda: order.append("structured")
        )
        core.shutdown()
        self.assertEqual(
            order, ["collector", "structured", "model", "scheduler"]
        )

    def test_merge_outer_join_duplicates_missing_and_unclean(self):
        writers = [self.writer(role) for role in ("client", "core", "frontend")]
        client, core, front = writers
        client.emit("attempt", {"engine_request_id": "ok", "phase": "main"})
        client.emit(
            "attempt_result", {"engine_request_id": "ok", "success": True}
        )
        core.emit("first_admission", {"engine_request_id": "ok"})
        front.emit(
            "finished",
            {
                "engine_request_id": "ok",
                "stats_status": "valid",
                "inference_time_label": 3,
            },
        )
        client.emit("attempt", {"engine_request_id": "duplicate"})
        client.emit("attempt", {"engine_request_id": "duplicate"})
        core.emit("first_admission", {"engine_request_id": "orphan"})
        for writer in writers:
            writer.close()
        report = C.merge_records(self.root / "unit-run")
        rows = {r["engine_request_id"]: r for r in report["requests"]}
        self.assertEqual(rows["ok"]["inference_time_label"], 3)
        self.assertIn("duplicate_attempts", rows["duplicate"]["status"])
        self.assertEqual(len(rows["duplicate"]["attempts"]), 2)
        self.assertIn("missing_attempts", rows["orphan"]["status"])
        self.assertIsNone(rows["orphan"]["inference_time_label"])
        self.assertFalse(report["collection_complete"])
        path = core.path
        lines = path.read_text("utf-8").splitlines()
        path.write_bytes(
            "\n".join(lines[:-1]).encode("utf-8") + b"\n{partial\n\xff"
        )
        report = C.merge_records(self.root / "unit-run")
        self.assertEqual(len(report["requests"]), 3)
        broken = next(f for f in report["files"] if f["path"] == str(path))
        self.assertIn("incomplete_or_unconfirmed_close", broken["status"])
        self.assertEqual(len(broken["issues"]), 2)

    def merge_batch_fixture(self, specs, name="batches"):
        config = C.CollectionConfig("unit-run", self.root / name)
        core = self.writer("core", config)
        front = self.writer("frontend", config)
        clients = []
        for metadata, phases, duplicate in specs:
            client = self.writer("client", config, metadata)
            clients.append(client)
            for index, phase in enumerate(phases):
                attempt = C.ClientAttempt(client, {"prompt": "p"}, phase)
                ident = attempt.start(
                    {"prompt": "p"}, "http://local/v1/completions"
                )
                engine_id = f"cmpl-{ident}-0"
                attempt.finish(
                    NS(
                        success=True,
                        error="",
                        output_tokens=1,
                        latency=3,
                        ttft=1,
                    )
                )
                core.emit("first_admission", {"engine_request_id": engine_id})
                front.emit(
                    "finished",
                    {
                        "engine_request_id": engine_id,
                        "stats_status": "valid",
                        "inference_time_label": 3,
                    },
                )
                if duplicate and index == 0:
                    client.emit(
                        "attempt",
                        {"engine_request_id": engine_id, "phase": phase},
                    )
        for writer in [*clients, core, front]:
            self.assertTrue(writer.close())
        return C.merge_records(config.output_dir / config.run_id), clients

    def test_merge_expected_121_with_zero_requests_is_incomplete(self):
        report, _ = self.merge_batch_fixture(
            [
                (
                    {
                        "expected_main_requests": 120,
                        "expected_probe_requests": 1,
                    },
                    [],
                    False,
                )
            ]
        )
        self.assertEqual(report["recording_status"], "complete")
        self.assertEqual(report["requests"], [])
        self.assertFalse(report["collection_complete"])
        self.assertFalse(report["client_batch_counts_complete"])
        batch = report["client_batches"][0]
        self.assertEqual(
            batch["actual_unique_attempts"], {"main": 0, "probe": 0}
        )
        self.assertEqual(
            batch["status"],
            ["main_attempt_shortfall", "probe_attempt_shortfall"],
        )

    def test_merge_expected_120_main_with_one_main_is_incomplete(self):
        report, _ = self.merge_batch_fixture(
            [
                (
                    {
                        "expected_main_requests": 120,
                        "expected_probe_requests": 1,
                    },
                    ["probe", "main"],
                    False,
                )
            ]
        )
        self.assertTrue(
            all(row["status"] == ["complete"] for row in report["requests"])
        )
        self.assertFalse(report["collection_complete"])
        batch = report["client_batches"][0]
        self.assertEqual(
            batch["actual_unique_attempts"], {"main": 1, "probe": 1}
        )
        self.assertEqual(batch["status"], ["main_attempt_shortfall"])

    def test_merge_probe_count_checked_independently(self):
        for count in (0, 2):
            with self.subTest(probes=count):
                report, _ = self.merge_batch_fixture(
                    [
                        (
                            {
                                "expected_main_requests": 1,
                                "expected_probe_requests": 1,
                            },
                            ["main"] + ["probe"] * count,
                            False,
                        )
                    ],
                    name=f"probes-{count}",
                )
                batch = report["client_batches"][0]
                direction = "shortfall" if count == 0 else "excess"
                self.assertEqual(
                    batch["status"], [f"probe_attempt_{direction}"]
                )
                self.assertFalse(report["collection_complete"])

    def test_merge_two_benchmarks_same_run_checked_separately(self):
        for counts, valid in (((120, 120), True), ((119, 121), False)):
            with self.subTest(counts=counts):
                report, clients = self.merge_batch_fixture(
                    [
                        (
                            {
                                "expected_main_requests": 120,
                                "expected_probe_requests": 1,
                            },
                            ["probe"] + ["main"] * count,
                            False,
                        )
                        for count in counts
                    ],
                    name=str(counts),
                )
                batches = {
                    b["instance_id"]: b for b in report["client_batches"]
                }
                self.assertEqual(len(batches), 2)
                self.assertEqual(report["collection_complete"], valid)
                self.assertEqual(report["client_batch_counts_complete"], valid)
                self.assertEqual(len(report["requests"]), 242)
                for client, count in zip(clients, counts):
                    batch = batches[client.instance_id]
                    self.assertEqual(batch["source_file"], str(client.path))
                    self.assertEqual(batch["source_line"], 1)
                    self.assertEqual(batch["run_id"], "unit-run")
                    self.assertEqual(
                        batch["expected_requests"], {"main": 120, "probe": 1}
                    )
                    self.assertEqual(
                        batch["actual_unique_attempts"],
                        {"main": count, "probe": 1},
                    )
                    expected_status = (
                        ["complete"]
                        if valid
                        else [
                            "main_attempt_shortfall"
                            if count < 120
                            else "main_attempt_excess"
                        ]
                    )
                    self.assertEqual(batch["status"], expected_status)

    def test_merge_batch_duplicate_attempt_not_counted_twice(self):
        report, _ = self.merge_batch_fixture(
            [
                (
                    {"expected_main_requests": 1, "expected_probe_requests": 1},
                    ["main", "probe"],
                    True,
                )
            ]
        )
        batch = report["client_batches"][0]
        self.assertEqual(batch["attempt_records"], 3)
        self.assertEqual(
            batch["actual_unique_attempts"], {"main": 1, "probe": 1}
        )
        self.assertEqual(batch["status"], ["duplicate_attempts"])
        self.assertGreater(batch["issues"][0]["source_line"], 1)
        self.assertFalse(report["collection_complete"])
        self.assertTrue(
            any(
                "duplicate_attempts" in row["status"]
                for row in report["requests"]
            )
        )

    def test_merge_batch_invalid_phase_is_explicit(self):
        for phase in (None, "warmup", ["main"]):
            with self.subTest(phase=phase):
                report, _ = self.merge_batch_fixture(
                    [
                        (
                            {
                                "expected_main_requests": 1,
                                "expected_probe_requests": 1,
                            },
                            [phase, "probe"],
                            False,
                        )
                    ],
                    name=str(phase),
                )
                batch = report["client_batches"][0]
                self.assertIn("invalid_phase", batch["status"])
                self.assertEqual(batch["actual_unique_attempts"]["main"], 0)
                self.assertFalse(report["collection_complete"])

    def test_merge_unverifiable_expected_counts_are_explicit(self):
        for phase in ("main", "probe"):
            for index, value in enumerate((None, -1, True, "1", 1.0)):
                with self.subTest(phase=phase, value=value):
                    metadata = {
                        "expected_main_requests": 1,
                        "expected_probe_requests": 1,
                    }
                    key = f"expected_{phase}_requests"
                    if value is None:
                        del metadata[key]
                    else:
                        metadata[key] = value
                    report, _ = self.merge_batch_fixture(
                        [(metadata, ["main", "probe"], False)],
                        name=f"{phase}-{index}",
                    )
                    self.assertFalse(report["collection_complete"])
                    self.assertIn(
                        f"unverifiable_{key}",
                        report["client_batches"][0]["status"],
                    )
        report, _ = self.merge_batch_fixture(
            [
                (
                    {"expected_main_requests": 0, "expected_probe_requests": 0},
                    [],
                    False,
                )
            ],
            name="zero-declared-total",
        )
        self.assertIn(
            "unverifiable_expected_total", report["client_batches"][0]["status"]
        )
        self.assertFalse(report["collection_complete"])

    def test_merge_client_instance_identity_must_be_verifiable(self):
        for value in (None, "", ["bad"]):
            with self.subTest(instance_id=value):
                report, clients = self.merge_batch_fixture(
                    [
                        (
                            {
                                "expected_main_requests": 1,
                                "expected_probe_requests": 1,
                            },
                            ["main", "probe"],
                            False,
                        )
                    ],
                    name=str(value),
                )
                client = clients[0]
                rows = self.records(client)
                rows[0]["instance_id"] = value
                client.path.write_text(
                    "\n".join(json.dumps(row) for row in rows) + "\n", "utf-8"
                )
                report = C.merge_records(client.path.parent)
                self.assertFalse(report["collection_complete"])
                self.assertIn(
                    "unverifiable_client_instance_id",
                    report["client_batches"][0]["status"],
                )

    def test_merge_ambiguous_client_header_cannot_verify_batch(self):
        _, clients = self.merge_batch_fixture(
            [
                (
                    {"expected_main_requests": 1, "expected_probe_requests": 1},
                    ["main", "probe"],
                    False,
                )
            ]
        )
        rows = self.records(clients[0])
        rows.insert(
            1,
            dict(
                rows[0],
                metadata={
                    "expected_main_requests": 120,
                    "expected_probe_requests": 1,
                },
            ),
        )
        clients[0].path.write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n", "utf-8"
        )
        report = C.merge_records(clients[0].path.parent)
        self.assertFalse(report["collection_complete"])
        self.assertIn(
            "unverifiable_client_header", report["client_batches"][0]["status"]
        )

    def test_merge_duplicate_client_instance_not_silently_combined(self):
        report, clients = self.merge_batch_fixture(
            [
                (
                    {"expected_main_requests": 1, "expected_probe_requests": 1},
                    ["main", "probe"],
                    False,
                )
                for _ in range(2)
            ]
        )
        rows = self.records(clients[1])
        rows[0]["instance_id"] = clients[0].instance_id
        clients[1].path.write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n", "utf-8"
        )
        report = C.merge_records(clients[1].path.parent)
        self.assertEqual(len(report["client_batches"]), 2)
        self.assertTrue(
            all(
                b["status"] == ["duplicate_client_instance_id"]
                for b in report["client_batches"]
            )
        )
        self.assertFalse(report["collection_complete"])

    def test_real_benchmark_main_header_metadata_with_infinite_rate(self):
        # Execute the actual main() AST, including its metadata construction
        # and real ClientCollector/writer. Stub the dataset/tokenizer and
        # benchmark coroutine; no HTTP, model or Benchmark loop is run.
        for index, (rate, enabled) in enumerate(
            ((float("inf"), "1"), (4.0, "1"), (float("inf"), "0"))
        ):
            with self.subTest(rate=rate, enabled=enabled):
                args = NS(
                    seed=0,
                    backend="vllm",
                    model="local-model",
                    served_model_name=None,
                    tokenizer=None,
                    tokenizer_mode="auto",
                    trust_remote_code=False,
                    base_url=None,
                    host="127.0.0.1",
                    port=8011,
                    endpoint="/v1/completions",
                    dataset_name="mixed",
                    num_prompts=2,
                    mixed_dry_run=False,
                    mixed_export_path=None,
                    top_p=None,
                    top_k=None,
                    min_p=None,
                    temperature=0.9,
                    profile=False,
                    lora_modules=None,
                    request_rate=rate,
                    burstiness=1,
                    max_concurrency=16,
                    logprobs=None,
                    disable_tqdm=True,
                    percentile_metrics="ttft",
                    metric_percentiles="99",
                    ignore_eos=False,
                    save_result=False,
                    append_result=False,
                )
                received = []

                async def no_benchmark(received=received, **kwargs):
                    received.append(kwargs)
                    return {}

                namespace = dict(
                    __file__=str(ROOT / "benchmarks/benchmark_serving.py"),
                    os=os,
                    random=NS(seed=Mock()),
                    np=NS(random=NS(seed=Mock())),
                    gc=NS(collect=Mock(), freeze=Mock()),
                    asyncio=NS(run=run_async),
                    benchmark=no_benchmark,
                    mixed_dataset_from_args=lambda _: NS(
                        sample=lambda *args: [NS(), NS()],
                        report=lambda: {"seed": 0},
                    ),
                    get_tokenizer=lambda *args, **kwargs: NS(),
                    check_goodput_args=lambda _: {},
                    OPENAI_COMPATIBLE_BACKENDS={"vllm"},
                )
                load_defs(
                    "benchmarks/benchmark_serving.py", ["main"], namespace
                )
                output_dir = self.root / f"header-{index}"
                with (
                    patch.dict(
                        os.environ,
                        {
                            "VLLM_R1_COLLECT": enabled,
                            "VLLM_R1_RUN_ID": "unit-run",
                            "VLLM_R1_COLLECT_DIR": str(output_dir),
                        },
                    ),
                    contextlib.redirect_stdout(io.StringIO()),
                    self.assertNoLogs(level=logging.ERROR),
                ):
                    namespace["main"](args)
                self.assertEqual(args.request_rate, rate)
                self.assertEqual(received[0]["request_rate"], rate)
                if enabled == "0":
                    self.assertIsNone(received[0]["request_collector"])
                    self.assertFalse(output_dir.exists())
                    continue
                paths = list((output_dir / "unit-run").glob("client-*.jsonl"))
                self.assertEqual(len(paths), 1)

                def reject_constant(value):
                    self.fail(f"Non-standard JSON constant: {value}")

                rows = [
                    json.loads(line, parse_constant=reject_constant)
                    for line in paths[0].read_text("utf-8").splitlines()
                ]
                self.assertEqual(len(rows), 2)
                metadata = rows[0]["metadata"]
                self.assertEqual(
                    metadata["request_rate"],
                    "inf" if rate == float("inf") else rate,
                )
                self.assertEqual(metadata["expected_main_requests"], 2)
                self.assertEqual(metadata["expected_probe_requests"], 1)
                self.assertTrue(rows[-1]["clean_shutdown"])
                self.assertTrue(rows[-1]["records_complete"])
                json.dumps(rows, allow_nan=False)

    def test_native_sender_and_probe_main_identity_chain(self):
        sent = []
        body = {"temperature": 0.7, "add_special_tokens": False}
        client = C.ClientCollector(
            self.config,
            backend="vllm",
            endpoint="/v1/completions",
            dataset_name="mixed",
            profile=False,
            lora_modules=None,
        )
        self.addCleanup(client.close)

        class FakeResponse:
            status, reason = 200, None

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            @property
            def content(self):
                async def chunks():
                    yield (
                        b'data: {"choices":[{"text":"token"}],'
                        b'"usage":{"completion_tokens":1}}'
                    )
                    yield b"data: [DONE]"

                return chunks()

        class FakeSession:
            def __init__(self, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            def post(self, *, url, json, headers):
                sent.append((dict(json), dict(headers)))
                return FakeResponse()

        namespace = {
            "__name__": __name__,
            "dataclass": dataclass,
            "field": field,
            "time": time,
            "os": os,
            "sys": sys,
            "traceback": traceback,
            "json": json,
            "asyncio": asyncio,
            "aiohttp": NS(ClientSession=FakeSession),
            "AIOHTTP_TIMEOUT": None,
        }
        load_defs(
            "benchmarks/backend_request_func.py",
            [
                "RequestFuncInput",
                "RequestFuncOutput",
                "async_request_openai_completions",
            ],
            namespace,
        )
        metrics = NS(
            completed=2,
            total_input=6,
            total_output=2,
            request_throughput=1,
            output_throughput=1,
            total_token_throughput=1,
        )
        namespace.update(
            ASYNC_REQUEST_FUNCS={
                "vllm": namespace["async_request_openai_completions"]
            },
            calculate_metrics=lambda **kwargs: (metrics, [1, 1]),
        )
        load_defs(
            "benchmarks/benchmark_serving.py",
            ["get_request", "benchmark"],
            namespace,
        )
        samples = [
            NS(
                prompt=f"<bos>prompt {i}",
                prompt_len=3,
                expected_output_len=8,
                multi_modal_data=None,
                metadata={
                    "prompt": f"<bos>prompt {i}",
                    "request_index": i,
                    "source_sample_id": f"sample-{i}",
                    "task_type": "chat",
                },
            )
            for i in range(2)
        ]
        with contextlib.redirect_stdout(io.StringIO()):
            run_async(
                namespace["benchmark"](
                    backend="vllm",
                    api_url="http://local/v1/completions",
                    base_url="http://local",
                    model_id="local-model",
                    model_name=None,
                    tokenizer=None,
                    input_requests=samples,
                    logprobs=None,
                    request_rate=float("inf"),
                    burstiness=1,
                    disable_tqdm=True,
                    profile=False,
                    selected_percentile_metrics=[],
                    selected_percentiles=[99],
                    ignore_eos=False,
                    goodput_config_dict={},
                    max_concurrency=1,
                    lora_modules=None,
                    extra_body=body,
                    request_collector=client,
                )
            )
        self.assertEqual(len(sent), 3)
        for payload, headers in sent:
            self.assertEqual(set(headers), {"Authorization", "X-Request-Id"})
            self.assertNotIn("task_type", payload)
            self.assertFalse(payload["add_special_tokens"])
            self.assertEqual(payload["temperature"], 0.7)
        rows = [
            r
            for r in self.records(client.writer)
            if r["record_type"] == "attempt"
        ]
        self.assertEqual([r["phase"] for r in rows], ["probe", "main", "main"])
        self.assertEqual(
            [r["sample"]["source_sample_id"] for r in rows],
            ["sample-0", "sample-0", "sample-1"],
        )
        # Execute native API ID assignments and native Request conversion.
        serving = ast.parse(
            (ROOT / "vllm/entrypoints/openai/serving_engine.py").read_text(
                "utf-8"
            )
        )
        base = next(
            n
            for n in ast.walk(serving)
            if isinstance(n, ast.FunctionDef) and n.name == "_base_request_id"
        )
        base.decorator_list = []
        ids = {"random_uuid": lambda: "unused"}
        compile_nodes([base], ids)
        completions = ast.parse(
            (ROOT / "vllm/entrypoints/openai/serving_completion.py").read_text(
                "utf-8"
            )
        )
        assignments = {
            target.id: n
            for n in ast.walk(completions)
            if isinstance(n, ast.Assign)
            for target in n.targets
            if isinstance(target, ast.Name)
            and target.id in ("request_id", "request_id_item")
        }
        for row, (_, headers) in zip(rows, sent):
            ids.update(
                self=NS(_base_request_id=ids["_base_request_id"]),
                raw_request=NS(headers=headers),
                i=0,
            )
            compile_nodes(
                [assignments["request_id"], assignments["request_id_item"]], ids
            )
            self.assertEqual(row["api_request_id"], ids["request_id"])
            engine_request = NS(
                request_id=ids["request_id_item"],
                client_index=0,
                prompt_token_ids=[1],
                mm_inputs=None,
                mm_hashes=None,
                mm_placeholders=None,
                sampling_params=params(),
                eos_token_id=None,
                lora_request=None,
                cache_salt=None,
            )
            native = N["Request"].from_engine_core_request(engine_request)
            self.assertEqual(native.request_id, row["engine_request_id"])
            self.config.validate_engine_request(
                native.request_id, engine_request.sampling_params
            )
        sent.clear()
        # Disabled sender preserves the payload and creates no collection ID.
        with patch.object(
            C.uuid, "uuid4", side_effect=AssertionError("disabled ID")
        ):
            run_async(
                namespace["async_request_openai_completions"](
                    namespace["RequestFuncInput"](
                        "p", "http://local/v1/completions", 1, 8, "m"
                    )
                )
            )
        self.assertEqual(set(sent[0][1]), {"Authorization"})

    def test_native_async_n1_and_error_diagnostics(self):
        namespace = dict(
            N,
            EngineClient=object,
            UsageContext=NS(ENGINE_CONTEXT=0),
            MULTIMODAL_REGISTRY=None,
            SamplingParams=types.SimpleNamespace,
            EngineDeadError=type("EngineDeadError", (Exception,), {}),
            RequestOutputCollector=lambda **kwargs: NS(),
            shutdown_prometheus=lambda: None,
        )
        load_defs("vllm/v1/engine/async_llm.py", ["AsyncLLM"], namespace)
        cls = namespace["AsyncLLM"]
        obj = cls.__new__(cls)
        collector = C.FrontendCollector(self.config)
        self.addCleanup(collector.close)
        ident = "cmpl-r1.unit-run." + "a" * 32 + "-0"
        added = []

        async def add_core(req):
            added.append(req.request_id)

        obj.collection_config = self.config
        obj.engine_core = NS(
            resources=NS(engine_dead=False),
            add_request_async=add_core,
            shutdown=lambda: None,
        )
        obj.output_handler = None
        obj.output_processor = NS(
            request_collector=collector, request_states={}, add_request=Mock()
        )
        obj.log_requests = False
        obj.processor = NS(
            process_inputs=lambda request_id, *args: (
                "p",
                NS(request_id=request_id),
            )
        )
        run_async(obj.add_request(ident, "p", params(output_kind=0)))
        self.assertEqual(added, [ident])
        self.assertEqual(
            obj.output_processor.add_request.call_args.args[0].request_id, ident
        )
        obj._record_collection_error(ident, "client_cancelled")
        obj.output_processor.request_states[ident] = state(ident)
        obj._record_collection_error(ident, "client_cancelled")
        obj._record_collection_error(ident, "generate_error", "SyntheticError")
        obj.shutdown()
        rows = [
            r
            for r in self.records(collector.writer)
            if r["record_type"] == "diagnostic"
        ]
        self.assertEqual(
            [r["reason"] for r in rows],
            ["client_cancelled", "generate_error", "shutdown_unfinished"],
        )

    def test_close_timeout_is_visible_and_later_drain_unconfirmed(self):
        gate, entered = threading.Event(), threading.Event()
        original = C.RecordWriter._write

        def paused(writer, record):
            if record["record_type"] == "header":
                entered.set()
                gate.wait(2)
            original(writer, record)

        with patch.object(C.RecordWriter, "_write", paused):
            writer = self.writer(
                config=C.CollectionConfig(
                    "unit-run", self.root, close_timeout=0.01
                )
            )
            self.assertTrue(entered.wait(1))
            with self.assertLogs(C.logger, logging.ERROR):
                self.assertFalse(writer.close())
            gate.set()
            writer._thread.join(1)
        self.assertFalse(self.records(writer)[-1]["clean_shutdown"])

    def test_code_only_mixed_unchanged_none_and_fixed_caps(self):
        native = types.ModuleType("benchmark_dataset")
        native.__dict__.update(
            ABC=ABC,
            abstractmethod=abstractmethod,
            dataclass=dataclass,
            random=random,
        )
        with patch.dict(sys.modules, {"benchmark_dataset": native}):
            load_defs(
                "benchmarks/benchmark_dataset.py",
                ["BenchmarkDataset", "SampleRequest"],
                native.__dict__,
            )
            mixed = load_module(
                "isolated_mixed_r1_regression",
                ROOT / "benchmarks/mixed_workload_dataset.py",
            )
        path = self.root / "CodeFeedback-Filtered-Instruction.jsonl"
        path.write_text(
            "\n".join(
                json.dumps(
                    {
                        "query": f"def f{i}():\n    return {i}",
                        "answer": "REFERENCE",
                        "resource": "local",
                        "lang": "python",
                    }
                )
                for i in range(4)
            ),
            "utf-8",
        )

        class Tokenizer:
            name_or_path = "local-stub"

            def get_chat_template(self):
                return "<bos>user/body/assistant"

            def apply_chat_template(self, messages, **kwargs):
                return "<bos>" + messages[0]["content"] + "<assistant>"

            def __call__(self, text, **kwargs):
                return NS(input_ids=list(text))

        before = random.getstate()
        for cap in (None, 17):
            results = []
            for enabled in ("0", "1"):
                with (
                    patch.dict(os.environ, {"VLLM_R1_COLLECT": enabled}),
                    patch.object(tempfile, "tempdir", str(self.root)),
                ):
                    dataset = mixed.MixedWorkloadDataset(
                        code_path=str(path),
                        code_count=2,
                        output_len=cap,
                        random_seed=42,
                    )
                    requests = dataset.sample(Tokenizer(), 2)
                results.append([r.metadata for r in requests])
                self.assertTrue(
                    all(
                        r.expected_output_len
                        == (17 if cap else len("REFERENCE"))
                        for r in requests
                    )
                )
                self.assertTrue(
                    all("\n    return" in r.prompt for r in requests)
                )
                self.assertTrue(
                    all("REFERENCE" not in r.prompt for r in requests)
                )
            self.assertEqual(results[0], results[1])
        self.assertEqual(before, random.getstate())

    def test_disabled_scheduler_constructor_creates_no_collector(self):
        namespace = scheduler_namespace()
        namespace.update(
            EventPublisherFactory=NS(create=lambda *args: None),
            compute_encoder_budget=lambda **kwargs: (0, 0),
            EncoderCacheManager=lambda **kwargs: NS(),
            KVCacheManager=lambda **kwargs: FakeKV(),
        )
        config = NS(
            scheduler_config=NS(
                max_num_seqs=8, max_num_batched_tokens=20, max_model_len=100
            ),
            cache_config=NS(
                num_gpu_blocks=100,
                block_size=16,
                enable_prefix_caching=False,
                prefix_caching_hash_algo="builtin",
            ),
            lora_config=None,
            kv_events_config=None,
            kv_transfer_config=None,
            speculative_config=None,
            model_config=NS(),
            parallel_config=NS(data_parallel_rank=0),
        )
        with (
            patch.dict(os.environ, {"VLLM_R1_COLLECT": "0"}),
            patch.object(
                C.uuid, "uuid4", side_effect=AssertionError("disabled ID")
            ),
            patch.object(
                C, "RecordWriter", side_effect=AssertionError("disabled writer")
            ),
        ):
            obj = namespace["Scheduler"](config, NS(), NS())
        self.assertIsNone(obj.request_collector)
        self.assertFalse(hasattr(obj, "first_admitted"))

    def test_native_resume_events_preserve_first_scheduled_label(self):
        collector = C.FrontendCollector(self.config)
        self.addCleanup(collector.close)
        processor = output_namespace()["OutputProcessor"](None, True, collector)
        req = state("resumed")
        processor.request_states[req.request_id] = req
        event = N["EngineCoreEventType"]
        with patch.dict(sys.modules, {"vllm.v1.engine": ENGINE_EVENTS}):
            processor.process_outputs(
                [
                    engine_output(
                        "resumed",
                        events=[
                            NS(type=event.QUEUED, timestamp=11),
                            NS(type=event.SCHEDULED, timestamp=12),
                        ],
                    )
                ],
                15,
                N["IterationStats"](),
            )
            stats = N["IterationStats"]()
            processor.process_outputs(
                [
                    engine_output(
                        "resumed",
                        N["FinishReason"].LENGTH,
                        [
                            NS(type=event.PREEMPTED, timestamp=16),
                            NS(type=event.SCHEDULED, timestamp=18),
                        ],
                    )
                ],
                22,
                stats,
            )
        row = next(
            r
            for r in self.records(collector.writer)
            if r["record_type"] == "finished"
        )
        self.assertEqual(row["timestamps"]["scheduled_ts"], 12)
        self.assertEqual(row["inference_time_label"], 10)
        self.assertEqual(row["native_stats"]["num_generation_tokens"], 2)

    def test_merge_cli_preserves_unassociated_bad_identity(self):
        writer = self.writer()
        writer.emit("diagnostic", {"engine_request_id": ["invalid"]})
        writer.close()
        output = self.root / "merged.json"
        result = subprocess.run(
            [
                sys.executable,
                "-B",
                str(SOURCE),
                "merge",
                "--input-dir",
                str(self.root / "unit-run"),
                "--output",
                str(output),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(output.read_text("utf-8"))
        self.assertEqual(len(report["unassociated_records"]), 1)
        self.assertFalse(report["collection_complete"])
        self.assertEqual(report["missing_roles"], ["core", "frontend"])

    def test_native_arrivals_metrics_and_ipc_definitions_unchanged(self):
        for path, names in (
            (
                "benchmarks/benchmark_serving.py",
                {"get_request", "calculate_metrics"},
            ),
            ("vllm/v1/metrics/stats.py", None),
            ("vllm/v1/engine/__init__.py", None),
        ):
            original = subprocess.run(
                ["git", "show", f"v0.9.1:{path}"],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            current = (ROOT / path).read_text("utf-8")
            if names is None:
                self.assertEqual(current, original)
            else:

                def definitions(text, names=names):
                    return [
                        ast.dump(n)
                        for n in ast.parse(text).body
                        if isinstance(
                            n, (ast.FunctionDef, ast.AsyncFunctionDef)
                        )
                        and n.name in names
                    ]

                self.assertEqual(definitions(current), definitions(original))


if __name__ == "__main__":
    unittest.main()
