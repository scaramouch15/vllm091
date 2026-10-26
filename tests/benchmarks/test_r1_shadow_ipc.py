# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stdlib IPC diagnostic tests; spawned fixtures never import real vLLM/torch.

Real Pipe/spawn tests exercise lifecycle, not the actual Msgpack/native model
environment. Real codec/import diagnosis remains a server acceptance step.
"""
import argparse
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/r1_shadow"))
import check_ipc as C
import repro_ipc_exit as R

EXPECTED = {"request_id": "ordinary-shadow-ipc-id", "tokens": 2, "lengths": None}


def fake_modules(cuda=False, decode_error=False, missing_ecr=False):
    modules = {name: types.ModuleType(name) for name in (
        "vllm", "vllm.v1", "vllm.v1.engine", "vllm.v1.request",
        "vllm.v1.serial_utils", "vllm.sampling_params", "torch")}
    modules["vllm"].__file__ = str(ROOT / "vllm/__init__.py")
    modules["vllm"].__version__ = "0.9.1"
    class ECR:
        def __init__(self, **kw): self.__dict__.update(kw)
    class Decoder:
        def __init__(self, kind): self.kind = kind
        def decode(self, incoming):
            if decode_error:
                raise ValueError("fixture decode failed")
            return types.SimpleNamespace(**pickle.loads(incoming[0]))
    class Encoder:
        def encode(self, request):
            return [pickle.dumps(dict(request_id=request.request_id,
                tokens=len(request.prompt_token_ids), lengths=request.r1_shadow_lengths))]
    if not missing_ecr:
        modules["vllm.v1.engine"].EngineCoreRequest = ECR
    modules["vllm.v1.request"].Request = types.SimpleNamespace(
        from_engine_core_request=lambda r: types.SimpleNamespace(
            request_id=r.request_id, num_prompt_tokens=r.tokens, r1_shadow_lengths=r.lengths))
    modules["vllm.v1.serial_utils"].MsgpackDecoder = Decoder
    modules["vllm.v1.serial_utils"].MsgpackEncoder = Encoder
    modules["vllm.sampling_params"].SamplingParams = lambda **kw: types.SimpleNamespace(**kw)
    modules["torch"].cuda = types.SimpleNamespace(is_initialized=lambda: cuda)
    modules["torch"].__version__ = "fixture"
    return modules


def fixture_actual_child(incoming, outgoing, case_id, directory, deadline,
                         exit_stack_timeout=C.EXIT_STACK_TIMEOUT):
    # Exercise the actual child body, with deliberately fake dependency/codec
    # modules installed only in this newly spawned test process.
    sys.modules.update(fake_modules())
    C.child(incoming, outgoing, case_id, directory, deadline, exit_stack_timeout)


def fixture_hang(incoming, outgoing, case_id, directory, deadline, exit_stack_timeout=None):
    events = C.Events(directory, "child", case_id)
    events.emit("fixture.child_started")
    guard = C.Stacks(Path(directory) / f"{case_id}.child.stacks.log", deadline)
    events.emit("fixture.blocked.before")
    while True:
        time.sleep(.05)


def fixture_exit(incoming, outgoing, case_id, directory, deadline, exit_stack_timeout=None):
    events = C.Events(directory, "child", case_id)
    events.emit("fixture.exit_7.before")
    os._exit(7)


def fixture_result_nonzero(incoming, outgoing, case_id, directory, deadline,
                           exit_stack_timeout=None):
    outgoing.send({"kind": "result", "value": pickle.loads(incoming[0])})
    outgoing.close()
    os._exit(9)


def fixture_result_then_hang(incoming, outgoing, case_id, directory, deadline,
                             exit_stack_timeout=None):
    outgoing.send({"kind": "result", "value": pickle.loads(incoming[0])})
    outgoing.close()
    while True:
        time.sleep(.05)


def fixture_target_exit_then_shutdown_hangs(incoming, outgoing, case_id, directory,
                                             deadline, exit_stack_timeout):
    """Model a target return followed by a stuck multiprocessing finalizer."""
    sys.modules.update(fake_modules())
    C.child(incoming, outgoing, case_id, directory, deadline, exit_stack_timeout)

    def shutdown_finalizer_blocks():
        while True:
            time.sleep(.01)

    # multiprocessing runs this after the target returns. child.target_exit is
    # already flushed at that point, matching a post-target shutdown hang.
    from multiprocessing.util import Finalize
    Finalize(None, shutdown_finalizer_blocks, exitpriority=0)


def fixture_isolation_child(incoming, outgoing, case_id, directory, deadline,
                            exit_stack_timeout, mode):
    if mode != "pure_spawn":
        sys.modules.update(fake_modules())
    R.isolated_child(incoming, outgoing, case_id, directory, deadline,
                     exit_stack_timeout, mode)


class IPCTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="r1-ipc-test-", dir=ROOT.parent)
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)
        self.context = multiprocessing.get_context("spawn")

    def run_case(self, target=fixture_actual_child, timeout=5., expected=None, case_id="case"):
        record = {"case_id": case_id}
        C.run_case(self.context, lambda: [pickle.dumps(expected or EXPECTED)],
                   expected or EXPECTED, record, self.out, timeout, target=target)
        return record

    def test_real_spawn_both_values_and_flushed_phase_identity(self):
        for case_id, lengths in (("none", None), ("three", (1., 2., 3.))):
            expected = dict(EXPECTED, lengths=lengths)
            record = self.run_case(expected=expected, case_id=case_id)
            self.assertTrue(record["passed"])
            self.assertEqual(record["actual"], expected)
            self.assertEqual(record["child_exitcode"], 0)
            self.assertEqual(record["exit_timeout_seconds"], 10.)
            self.assertFalse(record["child_alive_after_cleanup"])
            self.assertTrue(record["cleanup"]["complete"])
            self.assertEqual(record["cleanup"]["actions"], [])
            last = record["child_last_known_stage"]
            self.assertTrue(last["verified"])
            self.assertEqual(last["last_event"]["stage"], "child.target_exit")
            rows = [json.loads(line) for line in (self.out / f"{case_id}.child.stages.jsonl").read_text().splitlines()]
            self.assertTrue(all(r["pid"] == record["child_pid"] and r["case_id"] == case_id
                                and r["elapsed_seconds"] >= 0 for r in rows))
            self.assertEqual([r["monotonic_ns"] for r in rows], sorted(r["monotonic_ns"] for r in rows))
            for stage in ("import_ecr", "import_request", "import_decoder", "decode",
                          "request_construct", "import_torch", "cuda_check", "send_result"):
                names = [r["stage"] for r in rows]
                self.assertIn("child." + stage + ".before", names)
                self.assertIn("child." + stage + ".after", names)
            self.assertLess(names.index("child.exit_watchdog_ready"),
                            names.index("child.send_result.after"))

    def test_isolation_repro_distinguishes_spawn_imports_and_real_ipc(self):
        for mode in ("pure_spawn", "imports_torch_vllm", "actual_ecr_ipc"):
            case_id = "isolate_" + mode
            record = {"case_id": case_id}
            C.run_case(self.context, lambda: [pickle.dumps(EXPECTED)], EXPECTED,
                       record, self.out, 5., target=fixture_isolation_child,
                       target_args=(mode,))
            self.assertTrue(record["passed"], mode)
            self.assertEqual(record["actual"], EXPECTED)
            self.assertEqual(record["child_exitcode"], 0)
            self.assertEqual(record["child_last_known_stage"]["last_event"]["stage"],
                             "child.target_exit")

    def test_real_spawn_timeout_keeps_case_stage_stacks_and_reaps(self):
        record = {"case_id": "timeout"}
        with self.assertRaises(C.IPCWaitTimeout):
            C.run_case(self.context, lambda: [b"fixture"], EXPECTED, record,
                       self.out, 1., target=fixture_hang)
        self.assertFalse(record["passed"])
        self.assertTrue(record["timed_out"])
        self.assertGreaterEqual(record["wait_elapsed_seconds"], 1.)
        self.assertTrue(record["child_alive_at_timeout"])
        self.assertEqual(record["child_stage_at_timeout"]["last_event"]["stage"], "fixture.blocked.before")
        self.assertTrue(record["parent_stack"]["available"])
        self.assertTrue(record["child_stack"]["available"])
        self.assertFalse(record["child_alive_after_cleanup"])
        self.assertTrue(record["cleanup"]["reaped"])
        self.assertFalse(record["cleanup"]["reader_alive"])

    def test_real_spawn_exit_without_result_is_not_success(self):
        record = {"case_id": "exit"}
        with self.assertRaises(RuntimeError):
            C.run_case(self.context, lambda: [b"fixture"], EXPECTED, record,
                       self.out, 5., target=fixture_exit)
        self.assertFalse(record["passed"])
        self.assertEqual(record["child_exitcode"], 7)
        self.assertTrue(record["cleanup"]["reaped"])
        self.assertEqual(record["child_last_known_stage"]["last_event"]["stage"], "fixture.exit_7.before")

    def test_correct_result_followed_by_bad_exit_is_still_failure(self):
        record = {"case_id": "bad_exit"}
        with self.assertRaisesRegex(RuntimeError, "close normally"):
            C.run_case(self.context, lambda: [pickle.dumps(EXPECTED)], EXPECTED, record,
                       self.out, 5., target=fixture_result_nonzero)
        self.assertFalse(record["passed"])
        self.assertEqual(record["actual"], EXPECTED)
        self.assertEqual(record["child_exitcode"], 9)

    def test_result_does_not_bypass_exit_timeout(self):
        record = {"case_id": "exit_hang"}
        with patch.object(C, "EXIT_TIMEOUT", .15), self.assertRaises(C.ChildExitTimeout):
            C.run_case(self.context, lambda: [pickle.dumps(EXPECTED)], EXPECTED, record,
                       self.out, 5., target=fixture_result_then_hang)
        self.assertFalse(record["passed"])
        self.assertEqual(record["actual"], EXPECTED)
        self.assertTrue(record["cleanup"]["reaped"])

    def test_target_exit_watchdog_captures_post_target_shutdown_hang(self):
        record = {"case_id": "target_exit_hang"}
        with patch.object(C, "EXIT_TIMEOUT", .4), \
             patch.object(C, "EXIT_STACK_TIMEOUT", .05), \
             patch.object(C, "STACK_GRACE", .1), \
             self.assertRaises(C.ChildExitTimeout):
            C.run_case(self.context, lambda: [pickle.dumps(EXPECTED)], EXPECTED,
                       record, self.out, 5., target=fixture_target_exit_then_shutdown_hangs)

        self.assertFalse(record["passed"])
        self.assertEqual(record["actual"], EXPECTED)
        self.assertGreaterEqual(record["exit_wait_elapsed_seconds"], .4)
        self.assertEqual(record["exit_timeout_seconds"], .4)
        self.assertEqual(record["exit_stack_timeout_seconds"], .05)
        self.assertEqual(C.EXIT_TIMEOUT, 10.)
        self.assertEqual(record["child_last_known_stage"]["last_event"]["stage"],
                         "child.target_exit")
        self.assertTrue(record["child_stack"]["available"])
        stack_text = (self.out / "target_exit_hang.child.stacks.log").read_text()
        self.assertIn("shutdown_finalizer_blocks", stack_text)
        self.assertFalse(record["child_alive_after_cleanup"])
        self.assertTrue(record["cleanup"]["reaped"])

    def test_blocked_frame_reader_cannot_defeat_parent_wait_deadline(self):
        released = threading.Event()
        class Receive:
            def recv(self):
                released.wait()
                raise EOFError("fixture frame interrupted")
            def close(self): released.set()
        sender = types.SimpleNamespace(close=lambda: None)
        process = Mock(pid=1234, exitcode=None)
        process.is_alive.side_effect = lambda: process.exitcode is None
        process.terminate.side_effect = lambda: setattr(process, "exitcode", -15)
        context = types.SimpleNamespace(Pipe=lambda **kw: (Receive(), sender),
                                        Process=lambda **kw: process)
        record = {"case_id": "blocked_frame"}
        with self.assertRaises(C.IPCWaitTimeout):
            C.run_case(context, lambda: [b"fixture"], EXPECTED, record, self.out, .03)
        self.assertGreaterEqual(record["wait_elapsed_seconds"], .03)
        self.assertTrue(released.is_set())
        self.assertFalse(record["cleanup"]["reader_alive"])
        self.assertFalse(record["child_last_known_stage"]["verified"])
        parent_events = [json.loads(line) for line in
                         (self.out / "blocked_frame.parent.stages.jsonl").read_text().splitlines()]
        wait_event = next(e for e in parent_events if e["stage"] == "parent.wait.before")
        self.assertFalse(wait_event["startup_watchdog_armed"])

    def test_start_failure_closes_both_pipe_ends_and_preserves_original(self):
        receive, send = Mock(), Mock()
        process = Mock(pid=None, exitcode=None)
        process.start.side_effect = ValueError("fixture original start error")
        process.is_alive.return_value = False
        process.close.side_effect = OSError("fixture cleanup error")
        context = types.SimpleNamespace(Pipe=lambda **kw: (receive, send),
                                        Process=lambda **kw: process)
        record = {"case_id": "start_failure"}
        with self.assertRaisesRegex(ValueError, "original start error"):
            C.run_case(context, lambda: [b"fixture"], EXPECTED, record, self.out, 1.)
        receive.close.assert_called_once()
        send.close.assert_called_once()
        self.assertFalse(record["process_start_returned"])
        self.assertFalse(record["cleanup"]["state_verified"])
        self.assertIsNone(record["child_alive_after_cleanup"])
        self.assertEqual(record["cleanup"]["errors"][0]["error_type"], "OSError")

    def test_cleanup_escalates_to_kill_with_only_bounded_joins(self):
        events = C.Events(self.out, "parent", "kill")
        self.addCleanup(events.close)
        process = Mock(pid=1234, exitcode=None)
        process.is_alive.side_effect = lambda: process.exitcode is None
        process.kill.side_effect = lambda: setattr(process, "exitcode", -9)
        result = C.cleanup(process, Mock(), Mock(), None, events)
        self.assertEqual(result["actions"], ["terminate", "kill"])
        self.assertTrue(result["reaped"])
        self.assertTrue(result["complete"])
        self.assertEqual(process.join.call_count, 2)
        self.assertTrue(all(c.kwargs["timeout"] == C.CLEANUP_TIMEOUT for c in process.join.call_args_list))

    def test_watchdog_is_cancelled_before_manual_stack_dump(self):
        calls = []
        with patch.object(C.faulthandler, "dump_traceback_later",
                          side_effect=lambda *a, **kw: calls.append("arm")), \
             patch.object(C.faulthandler, "cancel_dump_traceback_later",
                          side_effect=lambda: calls.append("cancel")), \
             patch.object(C.faulthandler, "dump_traceback",
                          side_effect=lambda *a, **kw: calls.append("dump")):
            guard = C.Stacks(self.out / "ordered.stacks.log", time.monotonic() + 10.)
            guard.dump()
            guard.close()
        self.assertEqual(calls, ["arm", "cancel", "dump"])
        self.assertFalse(guard.armed)

    def test_real_child_import_decode_and_cuda_errors_keep_stage_and_error(self):
        for name, options, stage, error_type in (
            ("import", {"missing_ecr": True}, "child.import_ecr.before", "ImportError"),
            ("decode", {"decode_error": True}, "child.decode.before", "ValueError"),
            ("cuda", {"cuda": True}, "child.cuda_check.after", "RuntimeError")):
            outgoing = Mock()
            with patch.dict(sys.modules, fake_modules(**options)):
                C.child([pickle.dumps(EXPECTED)], outgoing, name, str(self.out), time.monotonic() + 5.)
            if C._CHILD_EXIT_STACKS is not None:
                C._CHILD_EXIT_STACKS.close()
                C._CHILD_EXIT_STACKS = None
            message = outgoing.send.call_args.args[0]
            self.assertEqual(message["kind"], "error")
            self.assertEqual(message["exception"]["error_type"], error_type)
            self.assertEqual(message["last_known_event_at_exception"]["stage"], stage)
            outgoing.close.assert_called_once()

    def test_stage_source_rejects_wrong_pid_and_ignores_partial_tail(self):
        events = C.Events(self.out, "child", "stage")
        events.emit("fixture.last_flushed")
        events.close()
        with events.path.open("ab") as file:
            file.write(b'{"stage":')
        result = C.read_child_stage(self.out, "stage", os.getpid())
        self.assertTrue(result["verified"])
        self.assertEqual(result["last_event"]["stage"], "fixture.last_flushed")
        result = C.read_child_stage(self.out, "stage", -999)
        self.assertFalse(result["verified"])
        self.assertIsNone(result["last_event"])
        events.path.write_text("[]\n")
        result = C.read_child_stage(self.out, "stage", os.getpid())
        self.assertFalse(result["verified"])
        self.assertIsNone(result["last_event"])

    def test_main_retains_first_failed_case_in_json(self):
        output = self.out / "main_failure"
        original = C.run_case
        def fail_case(*args, **kw):
            return original(*args, **kw, target=fixture_hang)
        with patch.dict(sys.modules, fake_modules()), \
             patch.object(C, "run_case", side_effect=fail_case), \
             patch.object(sys, "argv", ["check_ipc", "--out", str(output), "--timeout", "1"]), \
             self.assertRaises(C.IPCWaitTimeout):
            C.main()
        report = json.loads((output / "ipc.json").read_text())
        self.assertFalse(report["passed"])
        self.assertEqual(report["failed_case"], "none")
        self.assertEqual(len(report["cases"]), 1)
        self.assertFalse(report["cases"][0]["passed"])
        self.assertGreaterEqual(report["cases"][0]["wait_elapsed_seconds"], 1.)
        self.assertIsNotNone(report["cases"][0]["child_exitcode"])

    def test_main_default_timeout_and_both_case_checks(self):
        output = self.out / "main_success"
        original = C.run_case
        def run_fixture(*args, **kw):
            return original(*args, **kw, target=fixture_actual_child)
        with patch.dict(sys.modules, fake_modules()), \
             patch.object(C, "run_case", side_effect=run_fixture), \
             patch.object(sys, "argv", ["check_ipc", "--out", str(output)]):
            C.main()
        report = json.loads((output / "ipc.json").read_text())
        self.assertTrue(report["passed"])
        self.assertEqual(report["timeout_seconds"], 60.)
        self.assertEqual([c["case_id"] for c in report["cases"]], ["none", "three_lengths"])
        self.assertTrue(all(c["passed"] and c["child_exitcode"] == 0 for c in report["cases"]))

    def test_main_parent_cuda_check_remains_required(self):
        output = self.out / "cuda_failure"
        def complete_fixture(context, encode, expected, record, directory, timeout):
            record.update(passed=True, child_exitcode=0, child_alive_after_cleanup=False)
        with patch.dict(sys.modules, fake_modules(cuda=True)), \
             patch.object(C, "run_case", side_effect=complete_fixture), \
             patch.object(sys, "argv", ["check_ipc", "--out", str(output)]), \
             self.assertRaisesRegex(RuntimeError, "CUDA was initialized"):
            C.main()
        report = json.loads((output / "ipc.json").read_text())
        self.assertFalse(report["passed"])
        self.assertEqual(report["parent_setup_last_stage"]["stage"], "parent.cuda_check.after")

    def test_unreaped_child_report_is_written_before_exit_guard(self):
        output = self.out / "unclean"
        def unclean(context, encode, expected, record, directory, timeout):
            record.update(passed=False, child_alive_after_cleanup=True)
            raise RuntimeError("fixture cleanup unconfirmed")
        with patch.dict(sys.modules, fake_modules()), \
             patch.object(C, "run_case", side_effect=unclean), \
             patch.object(C.os, "_exit") as exit_guard, \
             patch.object(sys, "argv", ["check_ipc", "--out", str(output)]), \
             self.assertRaisesRegex(RuntimeError, "cleanup unconfirmed"):
            C.main()
        exit_guard.assert_called_once_with(1)
        report = json.loads((output / "ipc.json").read_text())
        self.assertFalse(report["passed"])
        self.assertTrue(report["cases"][0]["child_alive_after_cleanup"])

    def test_timeout_validation_and_existing_outputs_not_overwritten(self):
        self.assertEqual(C.positive_seconds("300"), 300.)
        for value in ("0", "-1", "nan", "inf"):
            with self.assertRaises(argparse.ArgumentTypeError):
                C.positive_seconds(value)
        output = self.out / "existing"
        output.mkdir()
        marker = output / "original_failure.txt"
        marker.write_text("keep me")
        with patch.object(sys, "argv", ["check_ipc", "--out", str(output)]), self.assertRaises(FileExistsError):
            C.main()
        self.assertEqual(marker.read_text(), "keep me")


if __name__ == "__main__":
    unittest.main()
