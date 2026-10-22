# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in R1 collection, with no model/dependency imports or Core IPC changes.

Offline use (does not import the vllm package or start an engine)::

    python vllm/v1/metrics/request_collector.py merge \
        --input-dir /path/to/run_id --output /path/to/merged.json
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import queue
import re
import threading
import time
import uuid
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CollectionConfig:
    run_id: str
    output_dir: Path
    queue_size: int = 1024
    flush_interval: float = 1.0
    close_timeout: float = 2.0

    @classmethod
    def from_env(cls) -> Optional[CollectionConfig]:
        enabled = os.environ.get("VLLM_R1_COLLECT", "0")
        if enabled == "0":
            return None
        if enabled != "1":
            raise ValueError("VLLM_R1_COLLECT must be 0 or 1")
        run_id = os.environ.get("VLLM_R1_RUN_ID", "")
        directory = os.environ.get("VLLM_R1_COLLECT_DIR", "")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", run_id):
            raise ValueError(
                "VLLM_R1_RUN_ID must be 1-128 safe filename characters"
            )
        if not directory:
            raise ValueError("VLLM_R1_COLLECT_DIR is required when collecting")
        return cls(run_id, Path(directory).expanduser().resolve())

    def require_stats(self, log_stats: bool) -> None:
        if not log_stats:
            raise ValueError(
                "R1 collection requires native log_stats; remove "
                "--disable-log-stats"
            )

    def validate_engine_request(self, request_id: str, params: Any) -> None:
        prefix = f"cmpl-r1.{self.run_id}."
        if (
            not request_id.startswith(prefix)
            or not re.fullmatch(r"[0-9a-f]{32}-0", request_id[len(prefix) :])
            or params.n != 1
        ):
            raise ValueError(
                "R1 collection requires its matching-run client, "
                "vLLM Completions, one prompt, n=1, no beam search"
            )


class RecordWriter:
    """One owner process, one bounded queue, one daemon file-writing thread.

    emit never waits for disk. close is an explicit, bounded shutdown drain.
    Footer metrics distinguish foreground work from background work; their sum
    is not an end-to-end latency measurement. Records must own their values.
    """

    def __init__(
        self,
        config: CollectionConfig,
        role: str,
        metadata: Optional[dict] = None,
    ):
        if config.queue_size <= 0 or config.flush_interval <= 0:
            raise ValueError(
                "collection buffer size and flush interval must be positive"
            )
        self.config = config
        self.role = role
        self.instance_id = uuid.uuid4().hex
        directory = config.output_dir / config.run_id
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"{role}-{os.getpid()}-{self.instance_id}.jsonl"
        # Open exclusively before the thread; startup errors are visible.
        self._file = self.path.open("x", encoding="utf-8", newline="\n")
        self._queue: queue.Queue = queue.Queue(maxsize=config.queue_size)
        self._lock = threading.Lock()
        self._closing = threading.Event()
        self._failed = threading.Event()
        self._metrics: Counter = Counter()
        self._sequence = 0
        self._close_started = 0
        self._warned: set[str] = set()
        self._header = {
            "record_type": "header",
            "schema_version": SCHEMA_VERSION,
            "run_id": config.run_id,
            "role": role,
            "pid": os.getpid(),
            "instance_id": self.instance_id,
            "metadata": metadata or {},
        }
        self._thread = threading.Thread(
            target=self._run, name=f"r1-{role}-writer", daemon=True
        )
        self._thread.start()

    def observe(self, name: str, elapsed_ns: int) -> None:
        with self._lock:
            self._metrics[f"{name}_calls"] += 1
            self._metrics[f"{name}_ns"] += elapsed_ns

    def _warn_once(self, category: str, message: str) -> None:
        with self._lock:
            first = category not in self._warned
            self._warned.add(category)
        if first:
            logger.error(
                "R1 collection incomplete (%s, %s): %s; "
                "further failures are counted",
                self.path,
                category,
                message,
            )

    def emit(
        self,
        kind: str,
        values: dict,
        construction_started: Optional[int] = None,
    ) -> bool:
        built = time.perf_counter_ns()
        with self._lock:
            self._sequence += 1
            record = dict(
                values,
                record_type=kind,
                sequence=self._sequence,
                run_id=self.config.run_id,
            )
            self._metrics["attempted_records"] += 1
            self._metrics["record_build_calls"] += 1
            if construction_started is not None:
                self._metrics["record_build_ns"] += built - construction_started
            reason = None
            if self._closing.is_set() or self._failed.is_set():
                reason = "writer_unavailable"
            else:
                try:
                    self._queue.put_nowait(record)
                    self._metrics["enqueued_records"] += 1
                    self._metrics["queue_peak"] = max(
                        self._metrics["queue_peak"], self._queue.qsize()
                    )
                except queue.Full:
                    reason = "queue_full"
            if reason:
                self._metrics["dropped_records"] += 1
                self._metrics[reason] += 1
            self._metrics["enqueue_calls"] += 1
            self._metrics["enqueue_ns"] += time.perf_counter_ns() - built
        if reason:
            self._warn_once(reason, "record was not enqueued")
        return reason is None

    def _write(self, record: dict) -> None:
        started = time.perf_counter_ns()
        try:
            text = (
                json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
            )
        finally:
            self.observe(
                "background_serialize", time.perf_counter_ns() - started
            )
        started = time.perf_counter_ns()
        try:
            self._file.write(text)
        finally:
            self.observe("background_write", time.perf_counter_ns() - started)

    def _flush(self) -> None:
        started = time.perf_counter_ns()
        try:
            self._file.flush()
        finally:
            self.observe("background_flush", time.perf_counter_ns() - started)

    def _run(self) -> None:
        try:
            self._write(self._header)
            self._flush()
            last_flush = time.monotonic()
            since_flush = 0
            while not self._closing.is_set() or not self._queue.empty():
                try:
                    record = self._queue.get(
                        timeout=min(self.config.flush_interval, 0.05)
                    )
                except queue.Empty:
                    record = None
                if record is not None:
                    try:
                        self._write(record)
                    except (TypeError, ValueError) as exc:
                        with self._lock:
                            self._metrics["serialization_errors"] += 1
                            self._metrics["dropped_records"] += 1
                        self._warn_once("serialization", str(exc))
                    else:
                        with self._lock:
                            self._metrics["written_records"] += 1
                        since_flush += 1
                if (
                    since_flush >= 64
                    or time.monotonic() - last_flush
                    >= self.config.flush_interval
                ):
                    self._flush()
                    since_flush = 0
                    last_flush = time.monotonic()
            self._flush()
            with self._lock:
                metrics = dict(self._metrics)
                if self._close_started:
                    metrics["shutdown_drain_ns"] = (
                        time.perf_counter_ns() - self._close_started
                    )
                footer = {
                    "record_type": "footer",
                    "run_id": self.config.run_id,
                    "last_sequence": self._sequence,
                    "metrics": metrics,
                    "clean_shutdown": not self._metrics["close_timeouts"],
                    "records_complete": not self._metrics["dropped_records"],
                }
            self._write(footer)
            self._flush()
        except Exception as exc:
            self._failed.set()
            with self._lock:
                self._metrics["writer_errors"] += 1
                unwritten = (
                    self._metrics["enqueued_records"]
                    - self._metrics["written_records"]
                    - self._metrics["serialization_errors"]
                )
                self._metrics["unconfirmed_records"] += unwritten
            self._warn_once(
                "write", f"{exc!r}; unconfirmed records={unwritten}"
            )
        finally:
            try:
                self._file.close()
            except Exception as exc:
                self._failed.set()
                self._warn_once("close", repr(exc))

    def close(self) -> bool:
        with self._lock:
            if not self._closing.is_set():
                self._close_started = time.perf_counter_ns()
                self._closing.set()
        self._thread.join(timeout=self.config.close_timeout)
        if self._thread.is_alive():
            with self._lock:
                self._metrics["close_timeouts"] += 1
            self._warn_once(
                "close_timeout", "drain not confirmed before deadline"
            )
            return False
        with self._lock:
            complete = not (
                self._metrics["dropped_records"]
                or self._metrics["close_timeouts"]
            )
        return complete and not self._failed.is_set()


@dataclass(frozen=True)
class AdmissionSnapshot:
    decision_batch_id: str
    scheduled_timestamp: float
    running_count: int
    running_computed_tokens: int
    running_remaining_prompt_tokens: int
    token_budget: int
    kv_usage: float


class CoreCollector:
    def __init__(
        self, config: CollectionConfig, metadata: Optional[dict] = None
    ):
        self.writer = RecordWriter(config, "core", metadata)
        self.first_admitted: set[str] = set()
        self._batch = 0

    def snapshot(
        self,
        running: list,
        token_budget: int,
        kv_manager: Any,
        scheduled_timestamp: float,
    ) -> AdmissionSnapshot:
        started = time.perf_counter_ns()
        computed = remaining = 0
        for request in running:
            computed += request.num_computed_tokens
            remaining += max(
                request.num_prompt_tokens - request.num_computed_tokens, 0
            )
        self._batch += 1
        snapshot = AdmissionSnapshot(
            f"{self.writer.instance_id}:{self._batch}",
            scheduled_timestamp,
            len(running),
            computed,
            remaining,
            token_budget,
            float(kv_manager.usage),
        )
        self.writer.observe("snapshot", time.perf_counter_ns() - started)
        return snapshot

    def admit(self, request_id: str, snapshot: AdmissionSnapshot) -> None:
        if request_id in self.first_admitted:
            return
        started = time.perf_counter_ns()
        self.first_admitted.add(request_id)
        self.writer.emit(
            "first_admission",
            {
                "engine_request_id": request_id,
                "snapshot": asdict(snapshot),
            },
            started,
        )

    def forget(self, request_id: str) -> None:
        self.first_admitted.discard(request_id)

    def close(self) -> bool:
        return self.writer.close()


class FrontendCollector:
    def __init__(
        self, config: CollectionConfig, metadata: Optional[dict] = None
    ):
        self.writer = RecordWriter(config, "frontend", metadata)

    def finished(
        self, state: Any, finished_stats: Any, terminal_origin: str
    ) -> None:
        started = time.perf_counter_ns()
        raw = asdict(finished_stats) if finished_stats is not None else None
        if raw is not None:
            raw["finish_reason"] = str(finished_stats.finish_reason)
        timestamps = {
            name: getattr(state.stats, name, 0.0) or None
            for name in (
                "queued_ts",
                "scheduled_ts",
                "first_token_ts",
                "last_token_ts",
            )
        }
        times = list(timestamps.values())
        missing = [name for name, value in timestamps.items() if value is None]
        valid = (
            raw is not None
            and raw["finish_reason"] in ("stop", "length")
            and not missing
            and all(math.isfinite(t) for t in times)
            and times == sorted(times)
        )
        self.writer.emit(
            "finished",
            {
                "engine_request_id": state.request_id,
                "native_stats": raw,
                "timestamps": timestamps,
                "stats_status": "valid"
                if valid
                else "missing_or_invalid_native_stats",
                "missing_timestamps": missing,
                "inference_time_label": raw["inference_time"]
                if valid
                else None,
                "terminal_origin": terminal_origin,
                "length_capped": raw is not None
                and raw["finish_reason"] == "length",
                "length_semantics": "actual_end_under_requested_limit"
                if raw is not None and raw["finish_reason"] == "length"
                else None,
            },
            started,
        )

    def diagnostic(
        self, request_id: str, reason: str, detail: Optional[str] = None
    ) -> None:
        started = time.perf_counter_ns()
        self.writer.emit(
            "diagnostic",
            {
                "engine_request_id": request_id,
                "reason": reason,
                "detail": detail,
                "inference_time_label": None,
            },
            started,
        )

    def close(self) -> bool:
        return self.writer.close()


class ClientAttempt:
    def __init__(self, writer: RecordWriter, sample: dict, phase: str):
        self.writer = writer
        self.sample = copy.deepcopy(sample)
        self.phase = phase
        self.attempt_id: Optional[str] = None

    def start(self, payload: dict, api_url: str) -> str:
        started = time.perf_counter_ns()
        if (
            urlsplit(api_url).path != "/v1/completions"
            or not isinstance(payload.get("prompt"), str)
            or payload.get("prompt") != self.sample["prompt"]
            or type(payload.get("n", 1)) is not int
            or payload.get("n", 1) != 1
            or payload.get("best_of") not in (None, 1)
            or payload.get("use_beam_search", False)
        ):
            raise ValueError(
                "R1 collection supports vLLM /v1/completions, "
                "one unchanged text prompt, n=1, no beam search"
            )
        if self.attempt_id is not None:
            raise ValueError(
                "R1 attempt context cannot be reused for a second send"
            )
        self.attempt_id = f"r1.{self.writer.config.run_id}.{uuid.uuid4().hex}"
        self.writer.emit(
            "attempt",
            {
                "engine_request_id": f"cmpl-{self.attempt_id}-0",
                "api_request_id": f"cmpl-{self.attempt_id}",
                "attempt_id": self.attempt_id,
                "phase": self.phase,
                "sample": self.sample,
                "request_parameters": {
                    name: payload[name]
                    for name in (
                        "model",
                        "max_tokens",
                        "temperature",
                        "top_p",
                        "top_k",
                        "min_p",
                        "ignore_eos",
                        "add_special_tokens",
                        "n",
                    )
                    if name in payload
                },
                "id_mapping": "vllm091_single_prompt_n1",
            },
            started,
        )
        return self.attempt_id

    def finish(self, output: Any) -> None:
        started = time.perf_counter_ns()
        self.writer.emit(
            "attempt_result",
            {
                "engine_request_id": f"cmpl-{self.attempt_id}-0",
                "phase": self.phase,
                "success": output.success,
                "error": output.error,
                "output_tokens": output.output_tokens,
                "latency": output.latency,
                "ttft": output.ttft,
            },
            started,
        )


class ClientCollector:
    def __init__(
        self,
        config: CollectionConfig,
        *,
        backend: str,
        endpoint: str,
        dataset_name: str,
        profile: bool,
        lora_modules: Any,
        metadata: Optional[dict] = None,
    ):
        if (
            backend != "vllm"
            or endpoint != "/v1/completions"
            or dataset_name != "mixed"
            or profile
            or lora_modules
        ):
            raise ValueError(
                "R1 collection requires --backend vllm, "
                "--endpoint /v1/completions, --dataset-name mixed, "
                "without --profile or LoRA modules"
            )
        self.writer = RecordWriter(config, "client", metadata)

    def prepare(self, requests: list) -> dict[int, ClientAttempt]:
        started = time.perf_counter_ns()
        result = {
            id(request): ClientAttempt(self.writer, request.metadata, "main")
            for request in requests
        }
        self.writer.observe("client_prepare", time.perf_counter_ns() - started)
        return result

    def probe(self, request: Any) -> ClientAttempt:
        return ClientAttempt(self.writer, request.metadata, "probe")

    def close(self) -> bool:
        return self.writer.close()


def _client_batch_counts(header: dict, records: list[dict]) -> dict:
    """Check one existing client shard, including batches with no attempts."""
    status = []
    issues = []
    expected = {}
    metadata = header.get("metadata")
    for phase in ("main", "probe"):
        value = (
            metadata.get(f"expected_{phase}_requests")
            if isinstance(metadata, dict)
            else None
        )
        if type(value) is not int or value < 0:
            status.append(f"unverifiable_expected_{phase}_requests")
            value = None
        expected[phase] = value
    if all(value is not None for value in expected.values()) and not sum(
        expected.values()
    ):
        status.append("unverifiable_expected_total")
    instance_id = header.get("instance_id")
    if not isinstance(instance_id, str) or not instance_id:
        status.append("unverifiable_client_instance_id")
    seen = set()
    unique = {phase: set() for phase in expected}
    attempt_count = 0
    for record in records:
        if record.get("record_type") != "attempt":
            continue
        attempt_count += 1
        phase = record.get("phase")
        engine_id = record.get("engine_request_id")
        reasons = []
        valid_phase = phase in ("main", "probe")
        if not valid_phase:
            reasons.append("invalid_phase")
        if (
            not isinstance(engine_id, str)
            or not engine_id
            or not isinstance(record.get("run_id"), str)
            or not record["run_id"]
            or record["run_id"] != header.get("run_id")
        ):
            reasons.append("unverifiable_attempt_identity")
        else:
            if engine_id in seen:
                reasons.append("duplicate_attempts")
            seen.add(engine_id)
            if valid_phase:
                unique[phase].add(engine_id)
        if reasons:
            status.extend(reasons)
            issues.append(
                {"source_line": record["source_line"], "reasons": reasons}
            )
    actual = {phase: len(ids) for phase, ids in unique.items()}
    for phase, count in actual.items():
        if expected[phase] is not None and count != expected[phase]:
            direction = "shortfall" if count < expected[phase] else "excess"
            status.append(f"{phase}_attempt_{direction}")
    return {
        "run_id": header.get("run_id"),
        "instance_id": instance_id,
        "source_file": header["source_file"],
        "source_line": header["source_line"],
        "expected_requests": expected,
        "actual_unique_attempts": actual,
        "attempt_records": attempt_count,
        "status": sorted(set(status)) or ["complete"],
        "issues": issues,
    }


def merge_records(input_dir: Path) -> dict:
    """Full outer join retaining duplicates, bad lines and orphan records."""
    files = []
    groups: dict[tuple, dict] = {}
    unassociated = []
    client_batches = []
    bucket_names = {
        "attempt": "attempts",
        "attempt_result": "client_results",
        "first_admission": "admissions",
        "finished": "finished",
        "diagnostic": "diagnostics",
    }
    for path in sorted(input_dir.glob("*.jsonl")):
        records = []
        issues = []
        # A killed writer can leave a partial UTF-8 character at EOF. Decode
        # each line independently so earlier records remain inspectable.
        with path.open("rb") as handle:
            for number, raw_line in enumerate(handle, 1):
                line = raw_line.decode("utf-8", errors="backslashreplace")
                try:
                    record = json.loads(raw_line.decode("utf-8"))
                    if not isinstance(record, dict):
                        raise ValueError("expected object")
                except (ValueError, TypeError) as exc:
                    issues.append(
                        {"line": number, "error": str(exc), "raw": line}
                    )
                    continue
                record = dict(record, source_file=str(path), source_line=number)
                records.append(record)
        headers = [r for r in records if r.get("record_type") == "header"]
        footers = [r for r in records if r.get("record_type") == "footer"]
        status = []
        if len(headers) != 1 or (
            records and records[0].get("record_type") != "header"
        ):
            status.append("invalid_header")
        if len(footers) != 1 or (
            records and records[-1].get("record_type") != "footer"
        ):
            status.append("incomplete_or_unconfirmed_close")
        if footers:
            footer = footers[-1]
            metrics = footer.get("metrics", {})
            if not isinstance(metrics, dict):
                metrics = {}
                status.append("invalid_footer_metrics")
            if not footer.get("clean_shutdown"):
                status.append("incomplete_or_unconfirmed_close")
            if (
                not footer.get("records_complete")
                or metrics.get("dropped_records", 0)
                or metrics.get("writer_errors", 0)
            ):
                status.append("records_lost_or_writer_failed")
            data = [
                r
                for r in records
                if r.get("record_type") not in ("header", "footer")
            ]
            if metrics.get("written_records", 0) != len(data):
                status.append("record_count_mismatch")
            seq = [r.get("sequence") for r in data]
            last_sequence = footer.get("last_sequence", 0)
            if (
                any(type(s) is not int for s in seq)
                or type(last_sequence) is not int
                or (
                    all(type(s) is int for s in seq)
                    and (
                        len(set(seq)) != len(seq)
                        or (seq and max(seq) > last_sequence)
                    )
                )
            ):
                status.append("invalid_sequence")
        if issues:
            status.append("malformed_records")
        header = headers[0] if headers else {}
        if header.get("role") == "client":
            batch = _client_batch_counts(header, records)
            if "invalid_header" in status:
                batch["status"] = sorted(
                    (set(batch["status"]) - {"complete"})
                    | {"unverifiable_client_header"}
                )
            client_batches.append(batch)
        for record in records:
            kind = record.get("record_type")
            if kind in ("header", "footer"):
                continue
            run_id, engine_id = (
                record.get("run_id"),
                record.get("engine_request_id"),
            )
            if run_id != header.get("run_id"):
                status.append("run_id_mismatch")
            if (
                not isinstance(run_id, str)
                or not run_id
                or not isinstance(engine_id, str)
                or not engine_id
                or not isinstance(kind, str)
                or kind not in bucket_names
            ):
                unassociated.append(record)
                continue
            key = (run_id, engine_id)
            if key not in groups:
                groups[key] = dict(
                    run_id=run_id,
                    engine_request_id=engine_id,
                    **{name: [] for name in bucket_names.values()},
                )
            groups[key][bucket_names[kind]].append(record)
        files.append(
            {
                "path": str(path),
                "header": header,
                "footer": footers[-1] if footers else None,
                "status": sorted(set(status)) or ["complete"],
                "issues": issues,
            }
        )
    requests = []
    for key in sorted(groups):
        row = groups[key]
        status = []
        for name in ("attempts", "client_results", "admissions", "finished"):
            count = len(row[name])
            if count != 1:
                status.append(
                    f"{'missing' if not count else 'duplicate'}_{name}"
                )
        if any(not r.get("success") for r in row["client_results"]):
            status.append("http_or_client_error")
        if row["diagnostics"]:
            status.extend(
                f"diagnostic:{r.get('reason')}" for r in row["diagnostics"]
            )
        finished = row["finished"][0] if len(row["finished"]) == 1 else None
        if finished and finished.get("stats_status") != "valid":
            status.append("invalid_native_stats")
        row["status"] = sorted(set(status)) or ["complete"]
        row["inference_time_label"] = (
            finished.get("inference_time_label")
            if not status and finished
            else None
        )
        row["label_status"] = (
            "available"
            if row["inference_time_label"] is not None
            else "missing_or_ambiguous"
        )
        requests.append(row)
    roles = {f["header"].get("role") for f in files}
    missing_roles = sorted({"client", "core", "frontend"} - roles)
    recording_complete = (
        bool(files)
        and not missing_roles
        and all(f["status"] == ["complete"] for f in files)
    )
    # instance_id is already present in client headers; it is not a new
    # online request field or the Scheduler's decision_batch_id.
    instances = Counter(
        (batch["run_id"], batch["instance_id"])
        for batch in client_batches
        if isinstance(batch["run_id"], str)
        and isinstance(batch["instance_id"], str)
        and batch["instance_id"]
    )
    for batch in client_batches:
        if (
            isinstance(batch["run_id"], str)
            and isinstance(batch["instance_id"], str)
            and instances[batch["run_id"], batch["instance_id"]] > 1
        ):
            batch["status"] = sorted(
                (set(batch["status"]) - {"complete"})
                | {"duplicate_client_instance_id"}
            )
    batch_counts_complete = bool(client_batches) and all(
        batch["status"] == ["complete"] for batch in client_batches
    )
    complete = (
        recording_complete
        and batch_counts_complete
        and not unassociated
        and all(r["status"] == ["complete"] for r in requests)
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "collection_complete": complete,
        "recording_status": "complete"
        if recording_complete
        else "incomplete_or_unconfirmed",
        "missing_roles": missing_roles,
        "client_batches": client_batches,
        "client_batch_counts_complete": batch_counts_complete,
        "files": files,
        "requests": requests,
        "unassociated_records": unassociated,
        "summary": dict(Counter(s for r in requests for s in r["status"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    merge = sub.add_parser("merge")
    merge.add_argument("--input-dir", type=Path, required=True)
    merge.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.input_dir.is_dir():
        parser.error("--input-dir must be an existing run directory")
    if args.output.resolve() in {
        p.resolve() for p in args.input_dir.glob("*.jsonl")
    }:
        parser.error("output must not overwrite a collection shard")
    report = merge_records(args.input_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            {
                "collection_complete": report["collection_complete"],
                "requests": len(report["requests"]),
                "missing_roles": report["missing_roles"],
                "summary": report["summary"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
