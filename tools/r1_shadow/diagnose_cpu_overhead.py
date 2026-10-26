# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline CPU diagnostic for the R1 shadow sweep and writer backend.

This is not a service or load generator. It reuses the real Predictor,
Shadow.sweep, Shadow.emit, and (for condition A) RecordWriter.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
MODEL_SHA256 = "23aff057712e7bec05db91d470af39a1b34eabddbf5852914710e1b8e4b96e1f"
CANDIDATE_SIZES = (1, 8, 32, 128)
MAX_SWEEPS = 150


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_source_event(paths: list[Path]) -> tuple[dict, dict]:
    """Choose the first intact core candidate record with at least eight rows."""
    for path in paths:
        header = None
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    record = json.loads(line)
                except (ValueError, TypeError):
                    continue
                if not isinstance(record, dict):
                    continue
                if line_number == 1 and record.get("record_type") == "header":
                    header = record
                    require(header.get("role") == "shadow-core",
                            f"{path.name}: input must be a Core shadow JSONL")
                    continue
                candidates = record.get("candidates")
                snapshot = record.get("snapshot")
                if (record.get("record_type") == "shadow_candidates"
                        and isinstance(candidates, list) and len(candidates) >= 8
                        and isinstance(snapshot, dict)):
                    require(record.get("candidate_count") == len(candidates),
                            f"{path.name}:{line_number}: candidate_count mismatch")
                    require(isinstance(record.get("run_id"), str)
                            and record.get("run_id") == (header or {}).get("run_id"),
                            f"{path.name}:{line_number}: run_id/header mismatch")
                    require(type(snapshot.get("running_count")) is int
                            and type(snapshot.get("token_budget")) is int
                            and type(snapshot.get("kv_usage")) in (int, float)
                            and math.isfinite(snapshot["kv_usage"])
                            and 0 <= snapshot["kv_usage"] <= 1,
                            f"{path.name}:{line_number}: invalid captured state")
                    require(len({row.get("engine_request_id")
                                 for row in candidates[:8]}) == 8,
                            f"{path.name}:{line_number}: duplicate captured candidate identity")
                    for index, candidate in enumerate(candidates[:8]):
                        lengths = candidate.get("length_quantiles")
                        require(isinstance(candidate.get("engine_request_id"), str)
                                and type(candidate.get("prompt_tokens")) is int
                                and candidate["prompt_tokens"] > 0
                                and type(candidate.get("max_tokens")) is int
                                and candidate["max_tokens"] == 2048
                                and isinstance(lengths, list) and len(lengths) == 3
                                and all(type(x) in (int, float) and math.isfinite(x)
                                        and 0 <= x <= 2048 for x in lengths)
                                and lengths == sorted(lengths),
                                f"{path.name}:{line_number}: invalid candidate {index}")
                    return record, {"file_name": path.name,
                                    "file_sha256": sha256(path),
                                    "line": line_number}
    raise ValueError("no intact shadow_candidates record with at least 8 candidates found")


def make_batch(source_candidates: list[dict], size: int) -> tuple[list[dict], list[dict]]:
    """Return numeric candidates and provenance; copies above eight are synthetic."""
    require(size in CANDIDATE_SIZES, "unsupported candidate size")
    require(len(source_candidates) >= 8, "source event needs at least 8 candidates")
    rows, provenance = [], []
    for index in range(size):
        original = source_candidates[index % 8]
        rows.append({
            "request_id": f"r1diag-{size}-{index:03d}",
            "num_prompt_tokens": original["prompt_tokens"],
            "max_tokens": original["max_tokens"],
            "r1_shadow_lengths": tuple(float(x) for x in original["length_quantiles"]),
            "r1_shadow_admitted": False,
            "r1_shadow_predicted": False,
            "status": SimpleNamespace(name="WAITING"),
        })
        provenance.append({
            "diagnostic_request_id": f"r1diag-{size}-{index:03d}",
            "source_engine_request_id": original["engine_request_id"],
            "source_candidate_index": index % 8,
            "synthetic_copy": index >= 8,
        })
    return rows, provenance


def nearest_rank(samples: list[int], percentile: int) -> int | None:
    if not samples:
        return None
    ordered = sorted(samples)
    rank = (percentile * len(ordered) + 99) // 100
    return ordered[rank - 1]


def summarize(samples: list[dict]) -> dict:
    by_group: dict[tuple, list[int]] = {}
    for sample in samples:
        by_group.setdefault((sample["phase"], sample["outcome"],
                             sample["candidate_count"]), []).append(sample["elapsed_ns"])
    groups = []
    for (phase, outcome, count), values in sorted(
            by_group.items(), key=lambda item: (item[0][0], item[0][1],
                                                -1 if item[0][2] is None else item[0][2])):
        groups.append({
            "phase": phase, "outcome": outcome, "candidate_count": count,
            "n": len(values), "p50_ns": nearest_rank(values, 50),
            "p95_ns": nearest_rank(values, 95), "p99_ns": nearest_rank(values, 99),
            "max_ns": max(values),
        })
    return {"quantile_method": "nearest_rank; rank=ceil(p*n), 1-based",
            "groups": groups}


def paired_order(block_index: int) -> tuple[str, str]:
    return ("A", "B") if block_index % 2 == 0 else ("B", "A")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class MemorySink:
    """Diagnostic sink: successful emit plus bounded in-memory predictions only."""
    def __init__(self, instance_id: str, capture_limit: int):
        self.instance_id = instance_id
        self.capture_limit = capture_limit
        self.predictions: list[dict] = []
        self.closed = False

    def emit(self, kind: str, values: dict, construction_started=None) -> bool:
        if kind == "shadow_candidates":
            if len(self.predictions) >= self.capture_limit:
                raise RuntimeError("memory sink prediction capture limit exceeded")
            self.predictions.append({
                "decision_batch_id": values["snapshot"]["decision_batch_id"],
                "candidate_count": values["candidate_count"],
                "candidates": values["candidates"],
            })
        return True

    def close(self) -> bool:
        self.closed = True
        return True


def build_runtime():
    shadow_module = load_module(ROOT / "vllm/v1/metrics/r1_shadow.py",
                                "r1_shadow_cpu_diagnostic_runtime")
    collector_module = load_module(ROOT / "vllm/v1/metrics/request_collector.py",
                                   "r1_shadow_cpu_diagnostic_collector")
    return shadow_module, collector_module


def read_jsonl(path: Path) -> list[dict]:
    result = []
    for line_number, line in enumerate(path.read_text("utf-8").splitlines(), 1):
        try:
            value = json.loads(line)
        except ValueError as exc:
            raise ValueError(f"invalid JSONL at {path.name}:{line_number}: {exc}") from exc
        if isinstance(value, dict):
            result.append(value)
    return result


@dataclass
class Condition:
    shadow: Any
    writer: Any
    sink: MemorySink | None
    writer_started_ns: int


def new_condition(name: str, size: int, block_dir: Path, block_sweeps: int,
                  args, modules, predictor) -> Condition:
    shadow_module, collector_module = modules
    config = shadow_module.ShadowConfig(
        model=args.model.resolve(), run_id=args.run_id, directory=args.out.resolve(),
        candidates=size, scan_limit=size, every=args.every,
    )
    started = time.perf_counter_ns()
    if name == "A":
        writer = collector_module.RecordWriter(
            collector_module.CollectionConfig(args.run_id, block_dir / "writer_a"),
            "cpu-diagnostic-a", {"condition": "A_real_record_writer",
                                  "model_sha256": MODEL_SHA256,
                                  "candidate_size": size},
        )
        sink = None
    else:
        writer = MemorySink(f"memory-{size}",
                            capture_limit=(args.warmup_sweeps + block_sweeps) * size)
        sink = writer
    try:
        shadow = shadow_module.Shadow(config, predictor, writer)
    except BaseException:
        writer.close()
        if name == "A" and writer._thread.is_alive():
            raise RuntimeError("A writer failed during setup and did not stop")
        raise
    return Condition(shadow, writer, sink, time.perf_counter_ns() - started)


def invoke_sweep(condition: Condition, batch: list[dict], running: list,
                 snapshot: dict, decision_id: str, phase: str) -> dict:
    waiting = [SimpleNamespace(**row) for row in batch]
    before_success = condition.shadow.counts["dynamic_outcome_success"]
    before_skip = condition.shadow.counts["dynamic_outcome_interval_skip"]
    before_empty = condition.shadow.counts["dynamic_outcome_empty"]
    started = time.perf_counter_ns()
    condition.shadow.sweep(
        waiting, running, SimpleNamespace(usage=float(snapshot["kv_usage"])),
        int(snapshot["token_budget"]), snapshot.get("scheduled_timestamp"), decision_id,
    )
    elapsed = time.perf_counter_ns() - started
    if condition.shadow.counts["dynamic_outcome_success"] > before_success:
        outcome, count = "success", len(batch)
    elif condition.shadow.counts["dynamic_outcome_interval_skip"] > before_skip:
        outcome, count = "interval_skip", None
    elif condition.shadow.counts["dynamic_outcome_empty"] > before_empty:
        outcome, count = "empty", 0
    else:
        outcome, count = "failed", len(batch)
    return {"phase": phase, "outcome": outcome, "candidate_count": count,
            "elapsed_ns": elapsed, "decision_batch_id": decision_id}


def compare_predictions(a_rows: list[dict], b_rows: list[dict]) -> dict:
    def project(rows):
        return [
            [(row.get("snapshot", {}).get("decision_batch_id",
                                         row.get("decision_batch_id")),
              candidate["engine_request_id"], candidate["prompt_tokens"],
              candidate["max_tokens"], list(candidate["length_quantiles"]),
              list(candidate["time_quantiles"])) for candidate in row["candidates"]]
            for row in rows
        ]
    left, right = project(a_rows), project(b_rows)
    require(len(left) == len(right), "A/B prediction sweep count mismatch")
    for index, (a, b) in enumerate(zip(left, right)):
        require(a == b, f"A/B prediction/input mismatch in successful sweep {index}")
    return {"successful_sweeps": len(left),
            "candidate_predictions": sum(len(row) for row in left),
            "exact_numeric_and_identity_match": True}


def prediction_phase(record: dict, condition: str) -> str | None:
    if condition == "A":
        decision_id = record.get("snapshot", {}).get("decision_batch_id")
    else:
        decision_id = record.get("decision_batch_id")
    if not isinstance(decision_id, str):
        return None
    if "-warmup-" in decision_id:
        return "warmup"
    if "-measured-" in decision_id:
        return "measured"
    return None


def validate_writer(condition: Condition, close_ok: bool) -> dict:
    writer = condition.writer
    thread_alive = writer._thread.is_alive()
    result = {"close_ok": bool(close_ok), "writer_thread_alive_after_close": thread_alive,
              "writer_thread_exited": not thread_alive, "footer_present": False,
              "footer_complete": False, "writer_file": writer.path.name}
    if thread_alive:
        result["error"] = "writer thread remains alive after close"
        result["passed"] = False
        return result
    records = read_jsonl(writer.path)
    headers = [row for row in records if row.get("record_type") == "header"]
    footers = [row for row in records if row.get("record_type") == "footer"]
    footer = footers[-1] if footers else None
    result["footer_present"] = footer is not None
    result["footer"] = footer
    data = [row for row in records
            if row.get("record_type") not in ("header", "footer")]
    metrics = footer.get("metrics", {}) if footer else {}
    sequences = [row.get("sequence") for row in data]
    sequence_ok = (
        all(type(value) is int for value in sequences)
        and sequences == list(range(1, len(sequences) + 1))
        and type(footer.get("last_sequence")) is int
        and footer.get("last_sequence") == len(sequences)
    ) if footer else False
    footer_complete = bool(
        close_ok and len(headers) == 1 and records[0] is headers[0]
        and len(footers) == 1 and records[-1] is footer
        and headers[0].get("run_id") == writer.config.run_id
        and footer.get("run_id") == writer.config.run_id
        and footer.get("clean_shutdown") is True
        and footer.get("records_complete") is True
        and metrics.get("written_records") == len(data)
        and metrics.get("dropped_records", 0) == 0
        and metrics.get("serialization_errors", 0) == 0
        and metrics.get("writer_errors", 0) == 0
        and metrics.get("close_timeouts", 0) == 0
        and metrics.get("unconfirmed_records", 0) == 0
        and sequence_ok
    )
    result.update(footer_complete=footer_complete, passed=footer_complete,
                  record_count=len(data), sequence_check=sequence_ok)
    if not footer_complete:
        result["error"] = "writer close/footer/drain validation failed"
    return result


def dynamic_histogram_count(distribution: dict) -> int:
    return sum(group["n"] for group in distribution["groups"]
               if group["stage"] == "dynamic_foreground")


def merge_dynamic_distributions(shadow_module, shadows: list[Any]) -> dict:
    overall = shadow_module.DurationHistogram()
    groups = {}
    for shadow in shadows:
        for (stage, outcome, candidate_count), histogram in shadow.duration_histograms.items():
            if stage != "dynamic_foreground":
                continue
            overall.merge(histogram)
            key = (outcome, candidate_count)
            if key not in groups:
                groups[key] = shadow_module.DurationHistogram()
            groups[key].merge(histogram)
    return {
        "schema_version": 1,
        "unit": "ns",
        "method": "nearest_rank_log_bucket_bounds_4_significant_bits",
        "bins_per_group": shadow_module.DurationHistogram.NUM_BINS,
        "overall": {"dynamic_foreground": overall.report()} if overall.n else {},
        "groups": [dict(stage="dynamic_foreground", outcome=outcome,
                        candidate_count=candidate_count, **histogram.report())
                   for (outcome, candidate_count), histogram in groups.items()],
    }


def make_shadow(shadow_module, condition: Condition, args, size: int,
                predictor, directory: Path):
    config = shadow_module.ShadowConfig(
        model=args.model.resolve(), run_id=args.run_id, directory=directory.resolve(),
        candidates=size, scan_limit=size, every=args.every,
    )
    return shadow_module.Shadow(config, predictor, condition.writer)


def run_condition_block(name: str, size: int, block_index: int,
                        successful_start: int, successful_count: int,
                        args, modules, predictor, batch: list[dict],
                        source_snapshot: dict, block_dir: Path) -> dict:
    """Run one isolated condition block through shutdown and backend verification."""
    shadow_module, _ = modules
    result = {"condition": name, "candidate_size": size,
              "block_index": block_index, "successful_start": successful_start,
              "successful_count": successful_count, "block_passed": False,
              "sweep_samples": [], "warmup_sweep_samples": [],
              "predictions": {"warmup": [], "measured": []}}
    condition = None
    active_shadow = None
    closed = False
    try:
        block_dir.mkdir(parents=True, exist_ok=False)
        condition = new_condition(name, size, block_dir, successful_count,
                                  args, modules, predictor)
        result["backend_startup_ns"] = condition.writer_started_ns
        active_shadow = condition.shadow
        running = [object() for _ in range(source_snapshot["running_count"])]

        def opportunity(shadow, phase: str, index: int) -> list[dict]:
            condition.shadow = shadow
            observations = []
            for inner in range(args.every):
                decision = f"diag-{block_index}-{size}-{phase}-{index}-{inner}"
                observations.append(invoke_sweep(condition, batch, running,
                                                 source_snapshot, decision, phase))
            return observations

        # Startup and warmup are outside formal sweep samples.
        warmup_shadow = active_shadow
        warmup_started = time.perf_counter_ns()
        for warm_index in range(args.warmup_sweeps):
            result["warmup_sweep_samples"].extend(
                opportunity(warmup_shadow, "warmup", warm_index))
        result["warmup_elapsed_ns"] = time.perf_counter_ns() - warmup_started
        result["warmup_dynamic_foreground"] = warmup_shadow.duration_report()
        result["warmup_counts"] = dict(warmup_shadow.counts)
        result["_warmup_shadow"] = warmup_shadow
        warmup_observations = dynamic_histogram_count(
            result["warmup_dynamic_foreground"])
        expected_warmup_calls = args.warmup_sweeps * args.every
        result["warmup_histogram_matches_external"] = (
            warmup_observations == len(result["warmup_sweep_samples"])
            == expected_warmup_calls
        )
        result["warmup_outcomes_complete"] = (
            warmup_shadow.counts["dynamic_outcome_success"] == args.warmup_sweeps
            and warmup_shadow.counts["dynamic_outcome_interval_skip"]
            == args.warmup_sweeps * (args.every - 1)
            and warmup_shadow.counts["dynamic_outcome_failed"] == 0
            and warmup_shadow.counts["dynamic_outcome_empty"] == 0
        )
        result["_warmup_shadow"] = warmup_shadow

        # A fresh Shadow resets its internal histogram while retaining the same
        # predictor and backend writer. No online runtime code is changed.
        active_shadow = make_shadow(shadow_module, condition, args, size,
                                    predictor, args.out)
        condition.shadow = active_shadow
        result["sweep_samples"] = []
        for sample_index in range(successful_start,
                                  successful_start + successful_count):
            result["sweep_samples"].extend(
                opportunity(active_shadow, "measured", sample_index))
        result["dynamic_foreground"] = active_shadow.duration_report()
        result["shadow_counts"] = dict(active_shadow.counts)
        result["_formal_shadow"] = active_shadow
        formal_observations = dynamic_histogram_count(result["dynamic_foreground"])
        expected_formal_calls = successful_count * args.every
        formal_successes = active_shadow.counts["dynamic_outcome_success"]
        expected_skips = successful_count * (args.every - 1)
        result["formal_histogram_matches_external"] = (
            formal_observations == len(result["sweep_samples"])
            == expected_formal_calls
        )
        result["formal_outcomes_complete"] = (
            formal_successes == successful_count
            and active_shadow.counts["dynamic_outcome_interval_skip"] == expected_skips
            and active_shadow.counts["dynamic_outcome_failed"] == 0
            and active_shadow.counts["dynamic_outcome_empty"] == 0
        )

        close_started = time.perf_counter_ns()
        close_ok = active_shadow.close()
        result["close_elapsed_ns"] = time.perf_counter_ns() - close_started
        closed = True
        if name == "A":
            result["backend_status"] = validate_writer(condition, close_ok)
            if result["backend_status"]["writer_thread_exited"]:
                records = read_jsonl(condition.writer.path)
                result["predictions"] = {
                    phase: [row for row in records
                            if row.get("record_type") == "shadow_candidates"
                            and prediction_phase(row, "A") == phase]
                    for phase in ("warmup", "measured")
                }
        else:
            result["backend_status"] = {
                "backend": "diagnostic_memory_sink",
                "close_ok": bool(close_ok),
                "sink_closed": condition.sink.closed,
                "queue_thread_or_disk": False,
                "captures_prediction_projection_in_bounded_memory": True,
                "online_log_completeness_claim": False,
                "passed": bool(close_ok and condition.sink.closed),
            }
            result["predictions"] = {
                phase: [row for row in condition.sink.predictions
                        if prediction_phase(row, "B") == phase]
                for phase in ("warmup", "measured")
            }

        expected_prediction_events = {
            "warmup": args.warmup_sweeps,
            "measured": successful_count,
        }
        result["prediction_capture_complete"] = all(
            len(result["predictions"][phase]) == expected
            for phase, expected in expected_prediction_events.items()
        )
        result["block_passed"] = all((
            result["backend_status"]["passed"],
            result["warmup_histogram_matches_external"],
            result["warmup_outcomes_complete"],
            result["formal_histogram_matches_external"],
            result["formal_outcomes_complete"],
            result["prediction_capture_complete"],
        ))
        if not result["block_passed"]:
            result["error"] = "condition block validation failed"
    except BaseException as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
        result["block_passed"] = False
    finally:
        if condition is not None and active_shadow is not None and not closed:
            try:
                close_started = time.perf_counter_ns()
                close_ok = active_shadow.close()
                result.setdefault("cleanup_close_elapsed_ns",
                                  time.perf_counter_ns() - close_started)
                closed = True
                if name == "A":
                    result["cleanup_backend_status"] = validate_writer(condition, close_ok)
                else:
                    result["cleanup_backend_status"] = {
                        "close_ok": bool(close_ok),
                        "sink_closed": condition.sink.closed,
                        "passed": bool(close_ok and condition.sink.closed),
                    }
            except BaseException as cleanup_exc:
                result["cleanup_error"] = {
                    "type": type(cleanup_exc).__name__, "message": str(cleanup_exc)}
    return result


def run_paired_block(order: tuple[str, str], runner: Callable[[str], dict]) -> dict:
    """Run sequentially; stop before the next condition if a block is invalid."""
    results = {}
    for name in order:
        block = runner(name)
        results[name] = block
        if not block.get("block_passed"):
            return {"pair_passed": False, "results": results,
                    "failed_condition": name, "comparison": None}
    try:
        comparisons = {
            phase: compare_predictions(results["A"]["predictions"][phase],
                                       results["B"]["predictions"][phase])
            for phase in ("warmup", "measured")
        }
    except BaseException as exc:
        return {"pair_passed": False, "results": results,
                "failed_condition": "prediction_comparison",
                "comparison_error": {"type": type(exc).__name__, "message": str(exc)},
                "comparison": None}
    return {"pair_passed": True, "results": results,
            "failed_condition": None, "comparison": comparisons}


def summarize_condition_blocks(shadow_module, blocks: list[dict]) -> dict:
    formal_samples = [sample for block in blocks for sample in block["sweep_samples"]]
    warmup_samples = [sample for block in blocks
                      for sample in block["warmup_sweep_samples"]]
    formal_shadows = [block["_formal_shadow"] for block in blocks]
    warmup_shadows = [block["_warmup_shadow"] for block in blocks]
    formal_distribution = merge_dynamic_distributions(shadow_module, formal_shadows)
    warmup_distribution = merge_dynamic_distributions(shadow_module, warmup_shadows)
    formal_count = dynamic_histogram_count(formal_distribution)
    warmup_count = dynamic_histogram_count(warmup_distribution)
    require(formal_count == len(formal_samples),
            "merged formal dynamic histogram and external sample count differ")
    require(warmup_count == len(warmup_samples),
            "merged warmup dynamic histogram and external sample count differ")
    return {
        "formal_sweep_summary": summarize(formal_samples),
        "warmup_sweep_summary": summarize(warmup_samples),
        "dynamic_foreground": formal_distribution,
        "warmup_dynamic_foreground": warmup_distribution,
        "formal_external_sample_count": len(formal_samples),
        "formal_internal_histogram_count": formal_count,
        "warmup_external_sample_count": len(warmup_samples),
        "warmup_internal_histogram_count": warmup_count,
        "footer_checks_passed": all(
            block["backend_status"].get("passed", False)
            for block in blocks if block["condition"] == "A"
        ),
        "successful_prediction_sweeps": sum(
            len(block["predictions"]["measured"]) for block in blocks),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--shadow-jsonl", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--every", type=int, default=8,
                        help="same Shadow prediction interval for A and B (default: 8)")
    parser.add_argument("--warmup-sweeps", type=int, default=3,
                        help="successful sweep calls per condition/size, excluded from samples")
    parser.add_argument("--sweeps", type=int, default=30,
                        help=f"measured successful sweeps per condition/size (1..{MAX_SWEEPS})")
    parser.add_argument("--block-size", type=int, default=5,
                        help="successful sweeps per alternating A/B block")
    args = parser.parse_args()
    require(1 <= args.every <= 64 and 0 <= args.warmup_sweeps <= 100
            and 1 <= args.sweeps <= MAX_SWEEPS and args.block_size > 0,
            "invalid frequency, warmup, sweep or block setting")
    require(args.sweeps * args.every <= 5000
            and args.warmup_sweeps * args.every <= 1000
            and args.sweeps * 128 <= 20000
            and args.warmup_sweeps * 128 <= 5000,
            "sample/call budget exceeded; reduce --sweeps or --every")
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", args.run_id) is not None,
            "run-id must contain 1-128 safe filename characters")
    require(not args.out.resolve().is_relative_to(ROOT),
            "output must be outside the repository")
    args.out.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 2, "passed": False,
              "scope": "CPU-only offline Shadow.sweep/writer-backend diagnostic",
              "model_expected_sha256": MODEL_SHA256,
              "candidate_sizes": list(CANDIDATE_SIZES), "every": args.every,
              "warmup_successful_sweeps_per_block": args.warmup_sweeps,
              "measured_successful_sweeps_per_condition_size": args.sweeps,
              "block_size": args.block_size,
              "python": platform.python_version(), "interpreter": sys.executable,
              "host": platform.node(), "cpu_model": platform.processor(),
              "cpu_affinity": sorted(os.sched_getaffinity(0))
              if hasattr(os, "sched_getaffinity") else None,
              "samples": [], "sizes": [], "blocks": [], "pairs": [],
              "size_summaries": {}, "prediction_checks": {}}
    error = None
    try:
        require(platform.python_version() == "3.10.22", "requires verified Python 3.10.22 service interpreter")
        for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                         "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            require(os.environ.get(variable) == "1", variable + " must equal 1")
        require(sha256(args.model) == MODEL_SHA256, "B model SHA256 mismatch")
        source_record, source = load_source_event(args.shadow_jsonl)
        report.update(source=source,
                      source_run_id=source_record.get("run_id"),
                      source_decision_batch_id=source_record["snapshot"].get("decision_batch_id"),
                      source_candidate_count=source_record["candidate_count"],
                      input_state={name: source_record["snapshot"].get(name)
                                   for name in ("running_count", "kv_usage",
                                                "token_budget", "scheduled_timestamp")},
                      input_provenance="first 8 from one original shadow_candidates event; sizes 32/128 repeat these numeric rows and are synthetic")
        runtime_started = time.perf_counter_ns()
        modules = build_runtime()
        report["runtime_module_setup_ns"] = time.perf_counter_ns() - runtime_started
        shadow_module, _ = modules
        shadow_source = ROOT / "vllm/v1/metrics/r1_shadow.py"
        report.update(script_sha256=sha256(Path(__file__).resolve()),
                      shadow_source_sha256=sha256(shadow_source),
                      model_sha256=sha256(args.model))
        started = time.perf_counter_ns()
        predictor = shadow_module.Predictor(args.model.resolve(), "core")
        report["predictor_cold_load_ns"] = time.perf_counter_ns() - started
        report["predictor_environment"] = predictor.environment
        report["threadpools"] = predictor.threadpools
        source_candidates = source_record["candidates"][:8]
        source_snapshot = source_record["snapshot"]
        pair_index = 0
        for size in CANDIDATE_SIZES:
            batch, provenance = make_batch(source_candidates, size)
            report["sizes"].append({"candidate_count": size,
                                    "candidate_origin": "captured" if size <= 8 else "synthetic_repetition",
                                    "provenance": provenance})
            group_dir = args.out / f"candidate_{size}"
            group_dir.mkdir()
            condition_blocks = {"A": [], "B": []}
            for block_start in range(0, args.sweeps, args.block_size):
                block_count = min(args.block_size, args.sweeps - block_start)
                current_pair_index = pair_index
                order = paired_order(current_pair_index)

                def run_one(name: str) -> dict:
                    block_dir = (group_dir /
                                 f"pair_{current_pair_index:03d}_{name}")
                    return run_condition_block(
                        name, size, current_pair_index, block_start, block_count,
                        args, modules, predictor, batch, source_snapshot, block_dir,
                    )

                pair = run_paired_block(order, run_one)
                pair_report = {
                    "candidate_size": size,
                    "pair_index": current_pair_index,
                    "order": list(order),
                    "pair_passed": pair["pair_passed"],
                    "failed_condition": pair.get("failed_condition"),
                    "comparison": pair.get("comparison"),
                    "comparison_error": pair.get("comparison_error"),
                }
                report["pairs"].append(pair_report)
                for name, block in pair["results"].items():
                    condition_blocks[name].append(block)
                    public_block = {
                        key: value for key, value in block.items()
                        if not key.startswith("_")
                        and key not in ("predictions", "sweep_samples",
                                        "warmup_sweep_samples")
                    }
                    public_block["block_dir"] = str(
                        (group_dir / f"pair_{current_pair_index:03d}_{name}")
                        .relative_to(args.out))
                    report["blocks"].append(public_block)
                    report["samples"].extend(
                        {"candidate_size": size, "condition": name, **sample}
                        for sample in block.get("warmup_sweep_samples", []))
                    report["samples"].extend(
                        {"candidate_size": size, "condition": name, **sample}
                        for sample in block.get("sweep_samples", []))
                if not pair["pair_passed"]:
                    report.setdefault("errors", []).append(
                        f"candidate size {size}, pair {current_pair_index}: "
                        f"{pair.get('failed_condition')} failed; pair is invalid")
                    raise RuntimeError("condition block or pair validation failed")
                pair_index += 1

            report["size_summaries"][str(size)] = {
                name: summarize_condition_blocks(shadow_module, blocks)
                for name, blocks in condition_blocks.items()
            }
            require(all(summary["footer_checks_passed"]
                        for summary in report["size_summaries"][str(size)].values()),
                    f"candidate size {size}: at least one block summary invalid")
            size_pairs = [pair for pair in report["pairs"]
                          if pair["candidate_size"] == size]
            report["prediction_checks"][str(size)] = {
                "pairs": [pair["comparison"] for pair in size_pairs],
                "measured_successful_sweeps_per_condition": sum(
                    pair["comparison"]["measured"]["successful_sweeps"]
                    for pair in size_pairs),
                "candidate_predictions_per_condition": sum(
                    pair["comparison"]["measured"]["candidate_predictions"]
                    for pair in size_pairs),
                "exact_numeric_and_identity_match": all(
                    pair["pair_passed"] for pair in size_pairs),
            }
            report["prediction_checks"][str(size)]["requested_counts_match"] = (
                report["prediction_checks"][str(size)][
                    "measured_successful_sweeps_per_condition"] == args.sweeps
                and report["prediction_checks"][str(size)][
                    "candidate_predictions_per_condition"] == args.sweeps * size
            )
        report["passed"] = bool(
            all(pair["pair_passed"] for pair in report["pairs"])
            and all(value["exact_numeric_and_identity_match"]
                    for value in report["prediction_checks"].values())
            and all(value["requested_counts_match"]
                    for value in report["prediction_checks"].values())
            and all(
                summary["footer_checks_passed"]
                and summary["formal_internal_histogram_count"]
                == summary["formal_external_sample_count"]
                and summary["warmup_internal_histogram_count"]
                == summary["warmup_external_sample_count"]
                for by_condition in report["size_summaries"].values()
                for summary in by_condition.values()
            )
        )
        report["timing_scope"] = (
            "External perf_counter_ns around the complete Shadow.sweep() call through return, "
            "including Shadow.emit, caller record construction, validation and foreground timing bookkeeping. "
            "dynamic_foreground is the implementation's own nested interval and excludes its final histogram update. "
            "Neither measure is client E2E or online scheduler overhead."
        )
        report["block_execution"] = (
            "Each A/B condition owns resources only within its block. The block is fully shut down and validated before the next condition starts; an invalid block stops the pair and no valid comparison is recorded. Pair order alternates AB/BA."
        )
        report["backend_scope"] = {
            "A": "real bounded RecordWriter queue, background thread and JSONL serialization/write/flush; close/footer checked",
            "B": "diagnostic memory sink; Shadow.emit and caller payload construction remain active, but emit only captures bounded candidate-prediction projections and returns success; no queue, background thread or file writing; not a no-logging mode or complete online logging",
        }
        report["interpretation_limits"] = [
            "CPU results do not estimate client E2E increment.",
            "This comparison cannot identify lock, disk, GIL or OS scheduling as an online spike cause.",
            "Candidate sizes 32 and 128 use repeated numeric candidates and are synthetic CPU diagnostics, not observed online workload coverage.",
            "No service, benchmark, GPU or model weights were loaded; only the fixed B predictor artifact is loaded on CPU.",
        ]
    except BaseException as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
        report["error"] = error
        raise
    finally:
        report["error"] = error
        report_path = args.out / "diagnostic.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                          allow_nan=False) + "\n", encoding="utf-8")
        print(f"diagnostic_report={report_path}", flush=True)
    if not report["passed"]:
        raise RuntimeError("CPU diagnostic failed; inspect diagnostic.json")


if __name__ == "__main__":
    main()
