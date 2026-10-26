# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Diagnose real ECR Msgpack + spawn round trips, without an LLM or GPU model.

--timeout is a functionality acceptance wait limit, NOT an IPC latency metric.
Each process owns flushed stage/stack files. Watchdogs cannot inspect a child
stuck before target entry; missing child evidence remains explicitly unknown.
"""
import argparse
import faulthandler
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import queue
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
EXIT_TIMEOUT = 10.0
CLEANUP_TIMEOUT = 5.0
READER_JOIN_TIMEOUT = 1.0
STACK_GRACE = 0.25
MAX_STAGE_BYTES = 1024 * 1024
_CHILD_EXIT_STACKS = None


def exception_record(exc):
    trace = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return {"error_type": type(exc).__name__, "error": str(exc)[:4096],
            "traceback": trace[-32768:], "traceback_truncated": len(trace) > 32768}


class Events:
    """Canonical per-process JSONL; stdout is a convenience mirror only."""
    def __init__(self, directory, role, case_id):
        self.path = Path(directory) / f"{case_id}.{role}.stages.jsonl"
        self.file = self.path.open("x", encoding="utf-8", newline="\n")
        self.role, self.case_id = role, case_id
        self.started = time.monotonic_ns()
        self.last = None
        self.errors = []

    def emit(self, stage, **values):
        now = time.monotonic_ns()
        event = dict(values, stage=stage, role=self.role, case_id=self.case_id,
                     pid=os.getpid(), monotonic_ns=now,
                     elapsed_seconds=(now - self.started) / 1e9)
        line = json.dumps(event, allow_nan=False)
        self.file.write(line + "\n")
        self.file.flush()
        self.last = event  # Only successfully flushed events are authoritative.
        try:
            print(line, flush=True)
        except (OSError, ValueError) as exc:
            if len(self.errors) < 8:
                self.errors.append(exception_record(exc))

    def safely(self, stage, **values):
        try:
            self.emit(stage, **values)
        except Exception as exc:
            if len(self.errors) < 8:
                self.errors.append(exception_record(exc))

    def close(self):
        try:
            self.file.close()
        except Exception as exc:
            self.errors.append(exception_record(exc))


class Stacks:
    """One non-repeating watchdog. No signals or external debugger processes."""
    def __init__(self, path, deadline):
        self.path, self.file, self.errors = Path(path), None, []
        self.armed = False
        try:
            self.file = self.path.open("x", encoding="utf-8")
            self.arm(deadline)
        except Exception as exc:
            self.errors.append(exception_record(exc))

    def arm(self, deadline):
        try:
            if self.file is None:
                return
            if self.armed:
                faulthandler.cancel_dump_traceback_later()
                self.armed = False
            remaining = deadline - time.monotonic()
            if remaining > 0:
                faulthandler.dump_traceback_later(remaining, repeat=False, file=self.file)
                self.armed = True
            else:
                self.armed = False
                self.dump()
        except Exception as exc:
            self.errors.append(exception_record(exc))

    def cancel(self):
        if self.armed:
            faulthandler.cancel_dump_traceback_later()
            self.armed = False

    def dump(self):
        try:
            # Never race a manual dump with this process's watchdog dump.
            self.cancel()
            if self.file is not None:
                faulthandler.dump_traceback(file=self.file, all_threads=True)
                self.file.flush()
        except Exception as exc:
            self.errors.append(exception_record(exc))

    def close(self):
        if self.armed:
            try:
                faulthandler.cancel_dump_traceback_later()
                self.armed = False
            except Exception as exc:
                self.errors.append(exception_record(exc))
        if self.file is not None:
            try:
                self.file.close()
            except Exception as exc:
                self.errors.append(exception_record(exc))


def child(incoming, outgoing, case_id, directory, stack_deadline):
    events = stacks = None
    try:
        events = Events(directory, "child", case_id)
        events.emit("child.started")
        stacks = Stacks(Path(directory) / f"{case_id}.child.stacks.log", stack_deadline)
        events.emit("child.stack_watchdog_ready", armed=stacks.armed, errors=stacks.errors)
        events.emit("child.import_ecr.before")
        from vllm.v1.engine import EngineCoreRequest
        events.emit("child.import_ecr.after")
        imported = Path(sys.modules["vllm"].__file__).resolve()
        if imported != ROOT / "vllm/__init__.py":
            raise ValueError("wrong child vLLM checkout: " + str(imported))
        events.emit("child.source_verified", vllm_file=str(imported))
        events.emit("child.import_request.before")
        from vllm.v1.request import Request
        events.emit("child.import_request.after")
        events.emit("child.import_decoder.before")
        from vllm.v1.serial_utils import MsgpackDecoder
        events.emit("child.import_decoder.after")
        events.emit("child.decode.before")
        request = MsgpackDecoder(EngineCoreRequest).decode(incoming)
        events.emit("child.decode.after")
        events.emit("child.request_construct.before")
        native = Request.from_engine_core_request(request)
        events.emit("child.request_construct.after")
        events.emit("child.import_torch.before")
        import torch
        events.emit("child.import_torch.after")
        events.emit("child.cuda_check.before")
        initialized = bool(torch.cuda.is_initialized())
        events.emit("child.cuda_check.after", initialized=initialized)
        if initialized:
            raise RuntimeError("CUDA was initialized")
        value = {"request_id": native.request_id, "tokens": native.num_prompt_tokens,
                 "lengths": native.r1_shadow_lengths}
        events.emit("child.send_result.before")
        outgoing.send({"kind": "result", "value": value})
        events.emit("child.send_result.after")
    except BaseException as exc:
        error = exception_record(exc)
        last_event = events.last if events is not None else None
        if events is not None:
            events.safely("child.exception", exception=error,
                          last_known_event_at_exception=last_event)
        try:
            outgoing.send({"kind": "error", "exception": error,
                           "last_known_event_at_exception": last_event})
        except BaseException as send_exc:
            if events is not None:
                events.safely("child.send_error.failed", exception=exception_record(send_exc))
    finally:
        if stacks is not None:
            # Keep the file and watchdog alive through Python/library atexit.
            # Closing it here would hide a hang after a successful send.
            stacks.arm(time.monotonic() + EXIT_TIMEOUT)
            global _CHILD_EXIT_STACKS
            _CHILD_EXIT_STACKS = stacks
            if events is not None:
                events.safely("child.exit_watchdog_ready", armed=stacks.armed,
                              timeout_seconds=EXIT_TIMEOUT, errors=stacks.errors)
        outgoing.close()
        if events is not None:
            events.safely("child.target_exit")
            events.close()


def read_child_stage(directory, case_id, pid):
    """Last complete JSONL event, checked by case/role/PID, never stdout order."""
    path = Path(directory) / f"{case_id}.child.stages.jsonl"
    result = {"source": str(path), "last_event": None, "verified": False}
    try:
        with path.open("rb") as file:
            data = file.read(MAX_STAGE_BYTES + 1)
        if len(data) > MAX_STAGE_BYTES:
            raise ValueError("stage file exceeds bounded diagnostic read")
        for line in data.splitlines(keepends=True):
            if not line.endswith(b"\n"):
                continue
            event = json.loads(line)
            if not isinstance(event, dict):
                raise ValueError("stage event is not a JSON object")
            if event.get("case_id") != case_id or event.get("role") != "child" or event.get("pid") != pid:
                raise ValueError("stage identity mismatch")
            result.update(last_event=event, verified=True)
        if result["last_event"] is None:
            result["unverified_reason"] = "no complete matching child event"
    except (OSError, ValueError) as exc:
        result.update(read_error=exception_record(exc), verified=False, last_event=None)
    return result


def receive_one(connection, mailbox):
    # poll() guarantees bytes/EOF, not a complete frame. recv() is isolated so
    # a partial frame cannot defeat the parent's result wait deadline.
    try:
        item = ("message", connection.recv())
    except BaseException as exc:
        item = ("receive_error", exception_record(exc))
    mailbox.put_nowait(item)


class IPCWaitTimeout(RuntimeError):
    pass


class ChildExitTimeout(RuntimeError):
    pass


def cleanup(process, receive, send, reader, events):
    state = {"actions": [], "errors": [], "reaped": False}
    for action in ("terminate", "kill"):
        try:
            if process is not None and process.is_alive():
                events.safely("parent.cleanup." + action, child_pid=process.pid)
                getattr(process, action)()
                state["actions"].append(action)
                process.join(timeout=CLEANUP_TIMEOUT)
        except BaseException as exc:
            state["errors"].append(exception_record(exc))
    if process is not None:
        try:
            state.update(child_pid=process.pid, child_alive=process.is_alive(),
                         child_exitcode=process.exitcode)
            state["state_verified"] = process.pid is not None
            if not state["state_verified"]:
                state.update(child_alive=None, child_exitcode=None,
                             unverified_reason="no child PID/handle available after start failure")
            state["reaped"] = state["state_verified"] and not state["child_alive"]
            if not state["child_alive"]:
                process.close()
        except BaseException as exc:
            state["errors"].append(exception_record(exc))
    for connection in (send, receive):
        if connection is not None:
            try:
                connection.close()
            except BaseException as exc:
                state["errors"].append(exception_record(exc))
    if reader is not None:
        try:
            if reader.ident is not None:
                reader.join(timeout=READER_JOIN_TIMEOUT)
        except BaseException as exc:
            state["errors"].append(exception_record(exc))
    state["reader_alive"] = reader is not None and reader.is_alive()
    state["complete"] = state["reaped"] and not state["reader_alive"] and not state["errors"]
    events.safely("parent.cleanup.after", **state)
    return state


def stack_file_status(path):
    result = {"path": str(path), "available": False}
    try:
        result["bytes"] = path.stat().st_size
        result["available"] = result["bytes"] > 0
    except OSError as exc:
        result["read_error"] = exception_record(exc)
    return result


def json_value(value):
    try:
        json.dumps(value, allow_nan=False)
        return value
    except (TypeError, ValueError):
        return {"unserializable_type": type(value).__name__, "repr": repr(value)[:4096]}


def run_case(context, encode, expected, record, directory, timeout, target=child):
    """Retain failures. Target injection is solely for standard-library tests."""
    case_id = record["case_id"]
    events = Events(directory, "parent", case_id)
    started = time.monotonic()
    stacks = Stacks(Path(directory) / f"{case_id}.parent.stacks.log", started + timeout)
    process = receive = send = reader = None
    failure = None
    record.update(passed=False, expected=expected, wait_timeout_seconds=timeout,
                  wait_elapsed_seconds=None, child_pid=None, process_start_returned=False)
    try:
        events.emit("parent.encode.before")
        buffers = encode()
        events.emit("parent.encode.after", buffer_bytes=sum(len(b) for b in buffers))
        receive, send = context.Pipe(duplex=False)
        process = context.Process(target=target,
            args=(buffers, send, case_id, str(directory), started + timeout))
        events.emit("parent.start.before")
        process.start()
        record.update(child_pid=process.pid, process_start_returned=True)
        events.emit("parent.start.after", child_pid=process.pid)
        send.close()  # A parent writer handle must not keep EOF hidden.
        # Startup/imports may block outside our wait loop, hence their watchdog.
        # The following result wait is bounded and can dump synchronously on
        # timeout: do not race an asynchronous parent dump with error handling.
        stacks.cancel()
        mailbox = queue.Queue(maxsize=1)
        reader = threading.Thread(target=receive_one, args=(receive, mailbox), daemon=True,
                                  name="r1-ipc-result-reader")
        reader.start()
        waiting = time.monotonic()
        deadline, exit_observed = waiting + timeout, None
        events.emit("parent.wait.before", timeout_seconds=timeout,
                    startup_watchdog_armed=stacks.armed)
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise IPCWaitTimeout("child IPC check timed out")
                try:
                    kind, value = mailbox.get(timeout=min(.1, remaining))
                    break
                except queue.Empty:
                    if not process.is_alive():
                        if exit_observed is None:
                            exit_observed = time.monotonic()
                            events.emit("parent.exit_observed_while_waiting", exitcode=process.exitcode)
                        elif time.monotonic() - exit_observed >= READER_JOIN_TIMEOUT:
                            raise RuntimeError("child exited without a complete result")
        finally:
            record["wait_elapsed_seconds"] = time.monotonic() - waiting
        events.emit("parent.receive.after", message_kind=kind,
                    wait_elapsed_seconds=record["wait_elapsed_seconds"])
        if kind != "message":
            record["receive_exception"] = value
            raise RuntimeError("result pipe failed: " + value["error_type"])
        if not isinstance(value, dict) or value.get("kind") != "result":
            record["child_response"] = json_value(value)
            raise RuntimeError("child did not return a successful result")
        actual = value["value"]
        record["actual"] = json_value(actual)
        if actual != expected:
            raise RuntimeError("round-trip mismatch: " + repr(actual))
        events.emit("parent.exit_check.before", timeout_seconds=EXIT_TIMEOUT)
        process.join(timeout=EXIT_TIMEOUT)
        events.emit("parent.exit_check.after", alive=process.is_alive(), exitcode=process.exitcode)
        if process.is_alive():
            raise ChildExitTimeout("child did not close within exit timeout")
        if process.exitcode != 0:
            raise RuntimeError("child did not close normally")
        record["passed"] = True
    except BaseException as exc:
        failure = exc
        timed_out = isinstance(exc, (IPCWaitTimeout, ChildExitTimeout))
        record.update(exception=exception_record(exc), timed_out=timed_out)
        record["parent_stage_at_exception"] = events.last
        events.safely("parent.exception", **record["exception"])
        if timed_out:
            record["child_stage_at_timeout"] = read_child_stage(directory, case_id, record["child_pid"])
            record["child_alive_at_timeout"] = process is not None and process.is_alive()
            stacks.dump()
            # Allow an already-armed child watchdog to flush, bounded in time.
            # Never signal a process that might not have installed a handler.
            if process is not None and process.is_alive():
                process.join(timeout=STACK_GRACE)
    finally:
        record["cleanup"] = cleanup(process, receive, send, reader, events)
        record["child_last_known_stage"] = read_child_stage(directory, case_id, record["child_pid"])
        record["child_exitcode"] = record["cleanup"].get("child_exitcode")
        record["child_alive_after_cleanup"] = record["cleanup"].get("child_alive")
        for role in ("parent", "child"):
            record[role + "_stack"] = stack_file_status(Path(directory) / f"{case_id}.{role}.stacks.log")
        stacks.close()
        record["stack_diagnostic_errors"] = stacks.errors
        record["elapsed_seconds"] = time.monotonic() - started
        record["parent_last_known_stage"] = events.last
        record["parent_stage_file"] = str(events.path)
        events.close()
        record["parent_log_errors"] = events.errors
        if not record["cleanup"]["complete"]:
            record["passed"] = False
            if failure is None:
                failure = RuntimeError("child cleanup was not confirmed complete")
                record["exception"] = exception_record(failure)
                record["parent_stage_at_exception"] = events.last
    if failure is not None:
        raise failure.with_traceback(failure.__traceback__)
    return record


def positive_seconds(value):
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("timeout must be positive finite seconds")
    return seconds


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--timeout", type=positive_seconds, default=60.0,
                   help="Result wait seconds per case (default 60); not an IPC cost")
    a = p.parse_args()
    if a.out.resolve().is_relative_to(ROOT):
        raise ValueError("keep outputs outside repository")
    a.out.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 2, "passed": False, "status": "running", "cases": [],
              "parent_pid": os.getpid(), "timeout_seconds": a.timeout,
              "interpreter": sys.executable, "python": sys.version,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    setup = Events(a.out, "parent", "setup")
    stacks = Stacks(a.out / "setup.parent.stacks.log", time.monotonic() + a.timeout)
    try:
        setup.emit("parent.import_vllm.before")
        import vllm
        setup.emit("parent.import_vllm.after", vllm_file=vllm.__file__)
        setup.emit("parent.import_torch.before")
        import torch
        setup.emit("parent.import_torch.after")
        setup.emit("parent.import_sampling_params.before")
        from vllm.sampling_params import SamplingParams
        setup.emit("parent.import_sampling_params.after")
        setup.emit("parent.import_ecr.before")
        from vllm.v1.engine import EngineCoreRequest
        setup.emit("parent.import_ecr.after")
        setup.emit("parent.import_encoder.before")
        from vllm.v1.serial_utils import MsgpackEncoder
        setup.emit("parent.import_encoder.after")
        if Path(vllm.__file__).resolve() != ROOT / "vllm/__init__.py":
            raise ValueError("wrong imported vLLM checkout: " + vllm.__file__)
        report.update(vllm_file=vllm.__file__, vllm_version=vllm.__version__)
        stacks.close()  # Sequential per-case watchdogs own the next deadlines.
        context = multiprocessing.get_context("spawn")
        for case_id, lengths in (("none", None), ("three_lengths", (1., 2., 3.))):
            case = {"case_id": case_id, "passed": False}
            report["cases"].append(case)  # Retain the first failing case too.
            def encode():
                request = EngineCoreRequest(request_id="ordinary-shadow-ipc-id", prompt_token_ids=[1, 2],
                    mm_inputs=None, mm_hashes=None, mm_placeholders=None,
                    sampling_params=SamplingParams(max_tokens=2048), eos_token_id=None,
                    arrival_time=1., lora_request=None, cache_salt=None,
                    data_parallel_rank=None, r1_shadow_lengths=lengths)
                return [bytes(b) for b in MsgpackEncoder().encode(request)]
            expected = {"request_id": "ordinary-shadow-ipc-id", "tokens": 2, "lengths": lengths}
            run_case(context, encode, expected, case, a.out, a.timeout)
        setup.emit("parent.cuda_check.before")
        initialized = bool(torch.cuda.is_initialized())
        setup.emit("parent.cuda_check.after", initialized=initialized)
        if initialized:
            raise RuntimeError("CUDA was initialized")
        report.update(passed=True, status="passed")
    except BaseException as exc:
        report.update(status="failed", **exception_record(exc))
        report["failed_case"] = next((c["case_id"] for c in report["cases"] if not c["passed"]), "parent_setup_or_final_check")
        raise
    finally:
        stacks.close()
        setup.close()
        report["parent_setup_last_stage"] = setup.last
        report["parent_setup_log_errors"] = setup.errors
        (a.out / "ipc.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if any(c.get("child_alive_after_cleanup") for c in report["cases"]):
            # Do not enter multiprocessing's implicit unbounded exit join if
            # even terminate/kill could not reap an owned OS process.
            os._exit(1)
    print("PASS: both actual ECR/native codec + spawn cases and CUDA/exit checks", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
