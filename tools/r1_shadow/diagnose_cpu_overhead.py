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
from typing import Any

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


def new_condition(name: str, size: int, args, modules, predictor, group_dir: Path) -> Condition:
    shadow_module, collector_module = modules
    config = shadow_module.ShadowConfig(
        model=args.model.resolve(), run_id=args.run_id, directory=args.out.resolve(),
        candidates=size, scan_limit=size, every=args.every,
    )
    started = time.perf_counter_ns()
    if name == "A":
        writer = collector_module.RecordWriter(
            collector_module.CollectionConfig(args.run_id, group_dir / "writer_a"),
            "cpu-diagnostic-a", {"condition": "A_real_record_writer",
                                  "model_sha256": MODEL_SHA256,
                                  "candidate_size": size},
        )
        sink = None
    else:
        writer = MemorySink(f"memory-{size}",
                            capture_limit=(args.warmup_sweeps + args.sweeps) * size)
        sink = writer
    shadow = shadow_module.Shadow(config, predictor, writer)
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
            [(candidate["engine_request_id"], candidate["prompt_tokens"],
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
    report = {"schema_version": 1, "passed": False,
              "scope": "CPU-only offline Shadow.sweep/writer-backend diagnostic",
              "model_expected_sha256": MODEL_SHA256,
              "candidate_sizes": list(CANDIDATE_SIZES), "every": args.every,
              "warmup_successful_sweeps": args.warmup_sweeps,
              "measured_successful_sweeps_per_condition_size": args.sweeps,
              "block_size": args.block_size,
              "python": platform.python_version(), "interpreter": sys.executable,
              "host": platform.node(), "cpu_model": platform.processor(),
              "cpu_affinity": sorted(os.sched_getaffinity(0))
              if hasattr(os, "sched_getaffinity") else None,
              "samples": [], "sizes": []}
    error = None
    active_conditions: list[Condition] = []
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
        for size in CANDIDATE_SIZES:
            batch, provenance = make_batch(source_candidates, size)
            report["sizes"].append({"candidate_count": size,
                                    "candidate_origin": "captured" if size <= 8 else "synthetic_repetition",
                                    "provenance": provenance})
            group_dir = args.out / f"candidate_{size}"
            group_dir.mkdir()
            conditions = {}
            for name in ("A", "B"):
                condition = new_condition(name, size, args, modules, predictor,
                                          group_dir)
                conditions[name] = condition
                active_conditions.append(condition)
            report.setdefault("backend_startup_ns", {})[str(size)] = {
                name: condition.writer_started_ns for name, condition in conditions.items()
            }
            running = [object() for _ in range(source_snapshot["running_count"])]
            local_samples = {"A": [], "B": []}
            warmup_samples = {"A": [], "B": []}
            def one_opportunity(name: str, phase: str, successful_index: int):
                condition = conditions[name]
                # One logical candidate opportunity contains `every` real calls;
                # Shadow.sweep itself decides which call predicts and which skips.
                observed = []
                for inner in range(args.every):
                    decision = f"diag-{size}-{phase}-{successful_index}-{inner}"
                    observed.append(invoke_sweep(condition, batch, running,
                                                 source_snapshot, decision, phase))
                destination = warmup_samples if phase == "warmup" else local_samples
                destination[name].extend(observed)

            # Warmup is also paired and order-alternated, but never enters timing samples.
            for warm_index in range(args.warmup_sweeps):
                order = paired_order(warm_index)
                for name in order:
                    one_opportunity(name, "warmup", warm_index)

            # Formal measurements alternate A/B order in bounded successful-sweep blocks.
            for block_start in range(0, args.sweeps, args.block_size):
                block_index = block_start // args.block_size
                order = paired_order(block_index)
                end = min(args.sweeps, block_start + args.block_size)
                for name in order:
                    for successful_index in range(block_start, end):
                        one_opportunity(name, "measured", successful_index)

            condition_results = {}
            for name, condition in conditions.items():
                close_started = time.perf_counter_ns()
                closed_ok = condition.shadow.close()
                close_elapsed = time.perf_counter_ns() - close_started
                if name == "A":
                    records = read_jsonl(condition.writer.path)
                    footer = next((r for r in reversed(records)
                                   if r.get("record_type") == "footer"), None)
                    emitted = [r for r in records
                               if r.get("record_type") == "shadow_candidates"]
                    writer_ok = bool(closed_ok and footer
                                     and footer.get("clean_shutdown")
                                     and footer.get("records_complete")
                                     and footer.get("metrics", {}).get("dropped_records", 0) == 0
                                     and footer.get("metrics", {}).get("writer_errors", 0) == 0
                                     and footer.get("metrics", {}).get("serialization_errors", 0) == 0)
                    detail = {"writer_file": str(condition.writer.path.relative_to(args.out)),
                              "close_ok": closed_ok, "footer_present": footer is not None,
                              "footer": footer, "writer_complete": writer_ok}
                else:
                    emitted = condition.sink.predictions
                    writer_ok = condition.sink.closed and closed_ok
                    detail = {"backend": "diagnostic_memory_sink",
                              "emit_returns_success": True,
                              "queue_thread_or_disk": False,
                              "captures_prediction_projection_in_bounded_memory": True,
                              "captured_sweeps": len(emitted),
                              "online_log_completeness_claim": False,
                              "close_ok": closed_ok}
                if not writer_ok:
                    report.setdefault("errors", []).append(f"condition {name}, size {size}: writer incomplete")
                expected_prediction_sweeps = args.warmup_sweeps + args.sweeps
                observed_prediction_sweeps = condition.shadow.counts["dynamic_outcome_success"]
                detail["expected_prediction_sweeps"] = expected_prediction_sweeps
                detail["observed_prediction_sweeps"] = observed_prediction_sweeps
                detail["sweep_outcomes_complete"] = (
                    observed_prediction_sweeps == expected_prediction_sweeps
                    and condition.shadow.counts["dynamic_outcome_failed"] == 0
                    and condition.shadow.counts["dynamic_outcome_empty"] == 0
                )
                if not detail["sweep_outcomes_complete"]:
                    report.setdefault("errors", []).append(
                        f"condition {name}, size {size}: prediction sweep outcomes incomplete")
                condition_results[name] = {
                    "writer_status": detail,
                    "sweep_samples": local_samples[name],
                    "sweep_summary": summarize(local_samples[name]),
                    "warmup_sweep_samples": warmup_samples[name],
                    "warmup_sweep_summary": summarize(warmup_samples[name]),
                    "dynamic_foreground": condition.shadow.duration_report(),
                    "shadow_counts": dict(condition.shadow.counts),
                    "close_elapsed_ns": close_elapsed,
                    "predictions": emitted,
                }
            prediction_check = compare_predictions(
                condition_results["A"]["predictions"],
                condition_results["B"]["predictions"],
            )
            for name in ("A", "B"):
                report["samples"].extend(
                    {"candidate_size": size, "condition": name, **sample}
                    for sample in condition_results[name]["warmup_sweep_samples"]
                )
                report["samples"].extend(
                    {"candidate_size": size, "condition": name, **sample}
                    for sample in condition_results[name]["sweep_samples"]
                )
            report.setdefault("condition_summaries", {})[str(size)] = {
                name: {"writer_status": value["writer_status"],
                       "sweep_summary": value["sweep_summary"],
                       "warmup_sweep_summary": value["warmup_sweep_summary"],
                       "dynamic_foreground": value["dynamic_foreground"],
                       "shadow_counts": value["shadow_counts"],
                       "close_elapsed_ns": value["close_elapsed_ns"]}
                for name, value in condition_results.items()
            }
            report.setdefault("prediction_checks", {})[str(size)] = prediction_check
            del condition_results
        backend_checks_passed = all(
            summary[name]["writer_status"].get("writer_complete",
                      summary[name]["writer_status"].get("close_ok", False))
            and summary[name]["writer_status"]["sweep_outcomes_complete"]
            for summary in report["condition_summaries"].values()
            for name in ("A", "B")
        )
        report["passed"] = backend_checks_passed and all(
            value["exact_numeric_and_identity_match"]
            for value in report["prediction_checks"].values()
        )
        report["timing_scope"] = (
            "External perf_counter_ns around the complete Shadow.sweep() call through return, "
            "including Shadow.emit, caller record construction, validation and foreground timing bookkeeping. "
            "dynamic_foreground is the implementation's own nested interval and excludes its final histogram update. "
            "Neither measure is client E2E or online scheduler overhead."
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
        for condition in active_conditions:
            try:
                condition.shadow.close()
            except BaseException as close_exc:
                report.setdefault("cleanup_errors", []).append(
                    {"type": type(close_exc).__name__, "message": str(close_exc)})
        report["error"] = error
        report_path = args.out / "diagnostic.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                          allow_nan=False) + "\n", encoding="utf-8")
        print(f"diagnostic_report={report_path}", flush=True)
    if not report["passed"]:
        raise RuntimeError("CPU diagnostic failed; inspect diagnostic.json")


if __name__ == "__main__":
    main()
