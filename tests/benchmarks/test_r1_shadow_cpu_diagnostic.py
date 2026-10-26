# SPDX-License-Identifier: Apache-2.0
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools/r1_shadow/diagnose_cpu_overhead.py"
SPEC = importlib.util.spec_from_file_location("r1_shadow_cpu_diagnostic_test", SCRIPT)
DIAG = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = DIAG
SPEC.loader.exec_module(DIAG)


def source_candidate(index):
    return {
        "engine_request_id": f"source-{index}",
        "prompt_tokens": 100 + index,
        "max_tokens": 2048,
        "length_quantiles": [10.0, 20.0, 30.0],
        "time_quantiles": [0.1, 0.2, 0.3],
    }


class FakePredictor:
    def assemble(self, candidates, running_count, kv_usage):
        return [[row.num_prompt_tokens, running_count, kv_usage,
                 *row.r1_shadow_lengths] for row in candidates]

    def dynamic(self, matrix):
        return [[float(row[0]) / 1000.0, float(row[0]) / 500.0,
                 float(row[0]) / 250.0] for row in matrix]


def run_test_block(name, size, block_index, successful_start, successful_count,
                   args, modules, predictor, batch, snapshot, block_dir):
    return DIAG.run_condition_block(
        name, size, block_index, successful_start, successful_count,
        args, modules, predictor, batch, snapshot, block_dir,
    )


class CpuDiagnosticTest(unittest.TestCase):
    def test_selects_intact_source_event_with_captured_state(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            path = Path(tmp) / "core.jsonl"
            record = {
                "record_type": "shadow_candidates",
                "run_id": "run-one",
                "candidate_count": 8,
                "snapshot": {"running_count": 3, "kv_usage": 0.5,
                             "token_budget": 8192, "scheduled_timestamp": 123.0},
                "candidates": [source_candidate(i) for i in range(8)],
            }
            header = {"record_type": "header", "role": "shadow-core",
                      "run_id": "run-one"}
            path.write_text(json.dumps(header) + "\n" + json.dumps(record) + "\n",
                            encoding="utf-8")
            selected, provenance = DIAG.load_source_event([path])
            self.assertEqual(selected, record)
            self.assertEqual(provenance["file_sha256"], DIAG.sha256(path))
            self.assertEqual(provenance["line"], 2)

    def test_requires_real_eight_candidate_event(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            path = Path(tmp) / "core.jsonl"
            record = {"record_type": "shadow_candidates", "run_id": "run-one",
                      "candidate_count": 1,
                      "snapshot": {"running_count": 0, "kv_usage": 0.0,
                                   "token_budget": 1},
                      "candidates": [source_candidate(0)]}
            header = {"record_type": "header", "role": "shadow-core",
                      "run_id": "run-one"}
            path.write_text(json.dumps(header) + "\n" + json.dumps(record) + "\n",
                            encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "at least 8"):
                DIAG.load_source_event([path])

    def test_sizes_mark_repeated_rows_as_synthetic(self):
        candidates = [source_candidate(i) for i in range(8)]
        batch, provenance = DIAG.make_batch(candidates, 128)
        self.assertEqual(len(batch), 128)
        self.assertEqual(len({row["request_id"] for row in batch}), 128)
        self.assertEqual(batch[0]["num_prompt_tokens"], batch[8]["num_prompt_tokens"])
        self.assertFalse(provenance[7]["synthetic_copy"])
        self.assertTrue(provenance[8]["synthetic_copy"])
        self.assertEqual(provenance[8]["source_engine_request_id"], "source-0")

    def test_nearest_rank_quantiles_and_empty_groups(self):
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.assertEqual(DIAG.nearest_rank(values, 50), 5)
        self.assertEqual(DIAG.nearest_rank(values, 95), 10)
        self.assertIsNone(DIAG.nearest_rank([], 50))
        report = DIAG.summarize([
            {"phase": "measured", "outcome": "success", "candidate_count": 8,
             "elapsed_ns": value} for value in values
        ])
        self.assertEqual(report["groups"][0]["p50_ns"], 5)
        self.assertEqual(report["groups"][0]["max_ns"], 10)

    def test_alternating_block_order(self):
        self.assertEqual([DIAG.paired_order(i) for i in range(4)],
                         [("A", "B"), ("B", "A"),
                          ("A", "B"), ("B", "A")])

    def test_memory_sink_is_bounded_and_returns_success(self):
        sink = DIAG.MemorySink("test", capture_limit=1)
        row = {"snapshot": {"decision_batch_id": "diag-test"},
               "candidate_count": 1, "candidates": [source_candidate(0)]}
        self.assertTrue(sink.emit("shadow_candidates", row))
        self.assertTrue(sink.emit("shadow_dynamic_error", {}))
        with self.assertRaisesRegex(RuntimeError, "capture limit"):
            sink.emit("shadow_candidates", row)
        self.assertTrue(sink.close())
        self.assertTrue(sink.closed)

    def test_real_sweep_and_record_writer_match_memory_sink_predictions(self):
        shadow_module, collector_module = DIAG.build_runtime()

        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            temp = Path(tmp)
            batch, _ = DIAG.make_batch([source_candidate(i) for i in range(8)], 1)
            snapshot = {"running_count": 2, "kv_usage": 0.25,
                        "token_budget": 8192, "scheduled_timestamp": 100.0}
            predictor = FakePredictor()
            config_a = shadow_module.ShadowConfig(Path("unused"), "test-run", temp,
                                                  candidates=1, scan_limit=1, every=1)
            writer_a = collector_module.RecordWriter(
                collector_module.CollectionConfig("test-run", temp / "real-writer"),
                "test-core", {},
            )
            shadow_a = shadow_module.Shadow(config_a, predictor, writer_a)
            config_b = shadow_module.ShadowConfig(Path("unused"), "test-run", temp,
                                                  candidates=1, scan_limit=1, every=1)
            sink_b = DIAG.MemorySink("memory-test", capture_limit=1)
            shadow_b = shadow_module.Shadow(config_b, predictor, sink_b)
            running = [object(), object()]
            condition_a = DIAG.Condition(shadow_a, writer_a, None, 0)
            condition_b = DIAG.Condition(shadow_b, sink_b, sink_b, 0)
            DIAG.invoke_sweep(condition_a, batch, running, snapshot, "diag-test", "measured")
            DIAG.invoke_sweep(condition_b, batch, running, snapshot, "diag-test", "measured")
            self.assertTrue(shadow_a.close())
            self.assertTrue(shadow_b.close())
            records = DIAG.read_jsonl(writer_a.path)
            a_predictions = [row for row in records
                             if row.get("record_type") == "shadow_candidates"]
            self.assertEqual(DIAG.compare_predictions(a_predictions,
                                                       sink_b.predictions)["successful_sweeps"], 1)
            footer = next(row for row in records if row.get("record_type") == "footer")
            self.assertTrue(footer["clean_shutdown"])
            self.assertTrue(footer["records_complete"])

    def test_b_starts_after_a_writer_thread_and_footer_are_closed(self):
        modules = DIAG.build_runtime()
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root = Path(tmp)
            args = SimpleNamespace(model=root / "unused.joblib", run_id="diag-test",
                                   out=root, every=1, warmup_sweeps=1,
                                   sweeps=2, block_size=2)
            batch, _ = DIAG.make_batch([source_candidate(i) for i in range(8)], 1)
            snapshot = {"running_count": 2, "kv_usage": 0.25,
                        "token_budget": 8192, "scheduled_timestamp": 100.0}
            predictor = FakePredictor()
            original = DIAG.new_condition
            a_conditions = []

            def observe_creation(name, *positional, **keyword):
                if name == "B":
                    self.assertTrue(a_conditions)
                    a = a_conditions[-1]
                    self.assertFalse(a.writer._thread.is_alive())
                    records = DIAG.read_jsonl(a.writer.path)
                    footer = next(row for row in records
                                  if row.get("record_type") == "footer")
                    self.assertTrue(footer["clean_shutdown"])
                    self.assertTrue(footer["records_complete"])
                condition = original(name, *positional, **keyword)
                if name == "A":
                    a_conditions.append(condition)
                return condition

            DIAG.new_condition = observe_creation
            try:
                pair = DIAG.run_paired_block(
                    ("A", "B"),
                    lambda name: run_test_block(
                        name, 1, 0, 0, 2, args, modules, predictor, batch,
                        snapshot, root / f"pair_{name}"),
                )
            finally:
                DIAG.new_condition = original
            self.assertTrue(pair["pair_passed"])

    def test_a_close_or_drain_failure_stops_before_b_is_created(self):
        class AliveThread:
            @staticmethod
            def is_alive():
                return True

        class FailedWriter:
            instance_id = "failed-writer"
            _thread = AliveThread()
            path = Path("failed-writer.jsonl")

            @staticmethod
            def emit(kind, values, construction_started=None):
                return True

            @staticmethod
            def close():
                return False

        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root = Path(tmp)
            args = SimpleNamespace(model=root / "unused.joblib", run_id="diag-test",
                                   out=root, every=1, warmup_sweeps=0,
                                   sweeps=1, block_size=1)
            modules = DIAG.build_runtime()
            shadow_module, _ = modules
            fake_writer = FailedWriter()
            config = shadow_module.ShadowConfig(
                args.model, args.run_id, root, candidates=1, scan_limit=1, every=1)
            failed_condition = DIAG.Condition(
                shadow_module.Shadow(config, FakePredictor(), fake_writer),
                fake_writer, None, 0)
            original = DIAG.new_condition
            started = []

            def fail_a(name, *positional, **keyword):
                started.append(name)
                if name == "A":
                    return failed_condition
                return original(name, *positional, **keyword)

            DIAG.new_condition = fail_a
            try:
                batch, _ = DIAG.make_batch([source_candidate(i) for i in range(8)], 1)
                pair = DIAG.run_paired_block(
                    ("A", "B"),
                    lambda name: run_test_block(
                        name, 1, 0, 0, 1, args, modules, FakePredictor(), batch,
                        {"running_count": 0, "kv_usage": 0.0,
                         "token_budget": 10}, root / f"pair_{name}"),
                )
            finally:
                DIAG.new_condition = original
            self.assertEqual(started, ["A"])
            self.assertFalse(pair["pair_passed"])
            self.assertEqual(pair["failed_condition"], "A")
            self.assertIsNone(pair["comparison"])

            calls = []

            def b_before_a(name):
                calls.append(name)
                return {"block_passed": name == "B", "predictions": {}}

            ba_pair = DIAG.run_paired_block(("B", "A"), b_before_a)
            self.assertEqual(calls, ["B", "A"])
            self.assertFalse(ba_pair["pair_passed"])
            self.assertEqual(ba_pair["failed_condition"], "A")
            self.assertIsNone(ba_pair["comparison"])

    def test_warmup_and_formal_histograms_are_disjoint(self):
        modules = DIAG.build_runtime()
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root = Path(tmp)
            args = SimpleNamespace(model=root / "unused.joblib", run_id="diag-test",
                                   out=root, every=1, warmup_sweeps=2,
                                   sweeps=3, block_size=3)
            batch, _ = DIAG.make_batch([source_candidate(i) for i in range(8)], 1)
            block = run_test_block(
                "A", 1, 0, 0, 3, args, modules, FakePredictor(), batch,
                {"running_count": 1, "kv_usage": 0.2,
                 "token_budget": 64}, root / "block_A")
            self.assertTrue(block["block_passed"])
            self.assertEqual(len(block["warmup_sweep_samples"]), 2)
            self.assertEqual(len(block["sweep_samples"]), 3)
            self.assertEqual(DIAG.dynamic_histogram_count(
                block["warmup_dynamic_foreground"]), 2)
            self.assertEqual(DIAG.dynamic_histogram_count(
                block["dynamic_foreground"]), 3)
            self.assertEqual(block["warmup_counts"]["dynamic_outcome_success"], 2)
            self.assertEqual(block["shadow_counts"]["dynamic_outcome_success"], 3)

    def test_multiple_blocks_merge_counts_predictions_and_writer_footers(self):
        modules = DIAG.build_runtime()
        shadow_module, _ = modules
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root = Path(tmp)
            args = SimpleNamespace(model=root / "unused.joblib", run_id="diag-test",
                                   out=root, every=1, warmup_sweeps=1,
                                   sweeps=4, block_size=2)
            batch, _ = DIAG.make_batch([source_candidate(i) for i in range(8)], 1)
            snapshot = {"running_count": 2, "kv_usage": 0.25,
                        "token_budget": 8192, "scheduled_timestamp": 100.0}
            predictor = FakePredictor()
            condition_blocks = {"A": [], "B": []}
            comparisons = []
            for block_index, start in enumerate((0, 2)):
                pair = DIAG.run_paired_block(
                    DIAG.paired_order(block_index),
                    lambda name, block_index=block_index, start=start:
                    run_test_block(name, 1, block_index, start, 2, args, modules,
                                   predictor, batch, snapshot,
                                   root / f"pair_{block_index}_{name}"),
                )
                self.assertTrue(pair["pair_passed"])
                comparisons.append(pair["comparison"]["measured"])
                for name, block in pair["results"].items():
                    condition_blocks[name].append(block)
            for name, blocks in condition_blocks.items():
                summary = DIAG.summarize_condition_blocks(shadow_module, blocks)
                self.assertEqual(summary["formal_external_sample_count"], 4)
                self.assertEqual(summary["formal_internal_histogram_count"], 4)
                self.assertEqual(summary["warmup_external_sample_count"], 2)
                self.assertEqual(summary["warmup_internal_histogram_count"], 2)
                self.assertTrue(summary["footer_checks_passed"])
                if name == "A":
                    self.assertTrue(all(block["backend_status"]["writer_thread_exited"]
                                        and block["backend_status"]["footer_complete"]
                                        for block in blocks))
            self.assertEqual(sum(row["successful_sweeps"] for row in comparisons), 4)
            self.assertEqual(sum(row["candidate_predictions"] for row in comparisons), 4)


if __name__ == "__main__":
    unittest.main()
