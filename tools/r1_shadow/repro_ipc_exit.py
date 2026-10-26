# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only isolation of spawn, imports, and the real ECR/Request IPC path.

The fixed child-exit acceptance timeout remains check_ipc.EXIT_TIMEOUT. This
tool measures separate child cases for diagnosis; it does not measure online
IPC latency, load a model, or intentionally initialize CUDA.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time

import check_ipc as C


def isolated_child(incoming, outgoing, case_id, directory, stack_deadline,
                   exit_stack_timeout, mode):
    """Run a light child case, or delegate the real case to check_ipc.child."""
    if mode == "actual_ecr_ipc":
        C.child(incoming, outgoing, case_id, directory, stack_deadline,
                exit_stack_timeout)
        return

    events = C.Events(directory, "child", case_id)
    stacks = C.Stacks(Path(directory) / f"{case_id}.child.stacks.log", stack_deadline)
    exit_watchdog_armed = False

    def keep_exit_watchdog():
        nonlocal exit_watchdog_armed
        global _CHILD_STACKS
        if exit_watchdog_armed:
            return
        stacks.arm(time.monotonic() + exit_stack_timeout)
        _CHILD_STACKS = stacks
        exit_watchdog_armed = True
        events.safely("child.exit_watchdog_ready", armed=stacks.armed,
                      timeout_seconds=exit_stack_timeout)

    events.emit("child.started", isolation=mode,
                torch_loaded="torch" in sys.modules,
                vllm_loaded="vllm" in sys.modules)
    try:
        if mode == "imports_torch_vllm":
            events.emit("child.import_torch.before")
            import torch
            events.emit("child.import_torch.after", torch_version=torch.__version__)
            events.emit("child.import_vllm.before")
            import vllm
            events.emit("child.import_vllm.after", vllm_version=vllm.__version__,
                        vllm_file=str(Path(vllm.__file__).resolve()))
            if Path(vllm.__file__).resolve() != C.ROOT / "vllm/__init__.py":
                raise ValueError("wrong imported vLLM checkout: " + vllm.__file__)
            events.emit("child.cuda_check.before")
            initialized = bool(torch.cuda.is_initialized())
            events.emit("child.cuda_check.after", initialized=initialized)
            if initialized:
                raise RuntimeError("CUDA was initialized")
        elif mode != "pure_spawn":
            raise ValueError("unknown isolation mode: " + mode)
        elif "torch" in sys.modules or "vllm" in sys.modules:
            raise RuntimeError("pure spawn unexpectedly imported torch or vLLM")

        value = pickle.loads(incoming[0])
        events.emit("child.send_result.before")
        keep_exit_watchdog()
        outgoing.send({"kind": "result", "value": value})
        events.emit("child.send_result.after")
    except BaseException as exc:
        error = C.exception_record(exc)
        last_event = events.last
        events.safely("child.exception", exception=error,
                      last_known_event_at_exception=last_event)
        try:
            keep_exit_watchdog()
            outgoing.send({"kind": "error", "exception": error,
                           "last_known_event_at_exception": last_event})
        except BaseException as send_exc:
            events.safely("child.send_error.failed",
                          exception=C.exception_record(send_exc))
    finally:
        # Mirror the production diagnostic's shutdown watchdog lifetime.
        keep_exit_watchdog()
        outgoing.close()
        events.safely("child.target_exit")
        events.close()


_CHILD_STACKS = None


def parent_encode_actual(case_id, output, expected):
    """Construct and encode the smallest real ECR; imports are stage logged."""
    events = C.Events(output, "parent_import", case_id)
    try:
        events.emit("parent.import_vllm.before")
        import vllm
        events.emit("parent.import_vllm.after", vllm_file=str(Path(vllm.__file__).resolve()),
                    vllm_version=vllm.__version__)
        if Path(vllm.__file__).resolve() != C.ROOT / "vllm/__init__.py":
            raise ValueError("wrong imported vLLM checkout: " + vllm.__file__)
        events.emit("parent.import_sampling_params.before")
        from vllm.sampling_params import SamplingParams
        events.emit("parent.import_sampling_params.after")
        events.emit("parent.import_ecr.before")
        from vllm.v1.engine import EngineCoreRequest
        events.emit("parent.import_ecr.after")
        events.emit("parent.import_encoder.before")
        from vllm.v1.serial_utils import MsgpackEncoder
        events.emit("parent.import_encoder.after")
        events.emit("parent.import_torch.before")
        import torch
        initialized = bool(torch.cuda.is_initialized())
        events.emit("parent.cuda_check.after", initialized=initialized)
        if initialized:
            raise RuntimeError("CUDA was initialized")
        request = EngineCoreRequest(
            request_id=expected["request_id"], prompt_token_ids=[1, 2],
            mm_inputs=None, mm_hashes=None, mm_placeholders=None,
            sampling_params=SamplingParams(max_tokens=2048), eos_token_id=None,
            arrival_time=1., lora_request=None, cache_salt=None,
            data_parallel_rank=None, r1_shadow_lengths=None)
        buffers = [bytes(buffer) for buffer in MsgpackEncoder().encode(request)]
        events.emit("parent.encode.after", buffer_bytes=sum(map(len, buffers)))
        return buffers
    finally:
        events.close()


def positive_seconds(value):
    return C.positive_seconds(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True,
                        help="new output directory outside the repository")
    parser.add_argument("--timeout", type=positive_seconds, default=60.,
                        help="bounded wait for each result (default 60 seconds)")
    args = parser.parse_args()
    output = args.out.resolve()
    if output.is_relative_to(C.ROOT):
        raise ValueError("keep outputs outside repository")
    output.mkdir(parents=True, exist_ok=False)

    import multiprocessing
    expected = {"request_id": "ordinary-shadow-ipc-id", "tokens": 2, "lengths": None}
    report = {
        "schema_version": 1,
        "passed": False,
        "cpu_only": True,
        "model_loaded": False,
        "timeout_seconds": args.timeout,
        "exit_timeout_seconds": C.EXIT_TIMEOUT,
        "exit_stack_timeout_seconds": C.EXIT_STACK_TIMEOUT,
        "interpreter": sys.executable,
        "python": sys.version,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cases": [],
    }
    context = multiprocessing.get_context("spawn")
    modes = (
        ("pure_spawn", "pure_spawn"),
        ("imports_torch_vllm", "imports_torch_vllm"),
        ("actual_ecr_ipc", "actual_ecr_ipc"),
    )
    for case_id, mode in modes:
        record = {"case_id": case_id, "isolation": mode, "passed": False}
        report["cases"].append(record)
        if mode == "actual_ecr_ipc":
            encode = lambda cid=case_id: parent_encode_actual(cid, output, expected)
        else:
            encode = lambda: [pickle.dumps(expected)]
        try:
            C.run_case(context, encode, expected, record, output, args.timeout,
                       target=isolated_child, target_args=(mode,))
        except Exception as exc:
            # Keep the case evidence and continue the independent comparisons.
            record.setdefault("exception", C.exception_record(exc))
        print(json.dumps({"case_id": case_id, "passed": record.get("passed"),
                          "wait_elapsed_seconds": record.get("wait_elapsed_seconds"),
                          "exit_wait_elapsed_seconds": record.get("exit_wait_elapsed_seconds"),
                          "child_exitcode": record.get("child_exitcode"),
                          "child_last_known_stage": record.get("child_last_known_stage"),
                          "child_stack": record.get("child_stack")},
                         allow_nan=False), flush=True)

    report["passed"] = all(case.get("passed") is True for case in report["cases"])
    report["status"] = "passed" if report["passed"] else "failed"
    report["parent_modules_loaded_after_run"] = {
        "torch": "torch" in sys.modules,
        "vllm": "vllm" in sys.modules,
    }
    report_path = output / "ipc_exit_repro.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8")
    print(f"report={report_path}", flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
