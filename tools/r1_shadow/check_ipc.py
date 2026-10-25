"""CPU-only real ECR Msgpack + spawn round trip; no EngineCore or CUDA model.

Execute in the verified full service clone, using this checkout's sources.
This is a serialization/process check, not a ZMQ/service integration test.
"""
import argparse
import json
import multiprocessing
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]


def child(incoming, outgoing):
    try:
        from vllm.v1.engine import EngineCoreRequest
        from vllm.v1.request import Request
        from vllm.v1.serial_utils import MsgpackDecoder
        request = MsgpackDecoder(EngineCoreRequest).decode(incoming)
        native = Request.from_engine_core_request(request)
        import torch
        if torch.cuda.is_initialized():
            raise RuntimeError("CUDA was initialized")
        outgoing.send({"request_id": native.request_id, "tokens": native.num_prompt_tokens,
                       "lengths": native.r1_shadow_lengths})
    except BaseException as exc:
        outgoing.send({"error": type(exc).__name__ + ": " + str(exc)})
    finally:
        outgoing.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.resolve().is_relative_to(ROOT):
        raise ValueError("keep outputs outside repository")
    a.out.mkdir(parents=True, exist_ok=False)
    report = {"passed": False}
    try:
        import vllm
        import torch
        from vllm.sampling_params import SamplingParams
        from vllm.v1.engine import EngineCoreRequest
        from vllm.v1.serial_utils import MsgpackEncoder
        if Path(vllm.__file__).resolve() != ROOT / "vllm/__init__.py":
            raise ValueError("wrong imported vLLM checkout: " + vllm.__file__)
        context = multiprocessing.get_context("spawn")
        report["cases"] = []
        for lengths in (None, (1., 2., 3.)):
            request = EngineCoreRequest(request_id="ordinary-shadow-ipc-id", prompt_token_ids=[1, 2],
                mm_inputs=None, mm_hashes=None, mm_placeholders=None,
                sampling_params=SamplingParams(max_tokens=2048), eos_token_id=None,
                arrival_time=1., lora_request=None, cache_salt=None,
                data_parallel_rank=None, r1_shadow_lengths=lengths)
            buffers = [bytes(b) for b in MsgpackEncoder().encode(request)]
            receive, send = context.Pipe(duplex=False)
            process = context.Process(target=child, args=(buffers, send))
            process.start()
            send.close()
            try:
                if not receive.poll(60):
                    raise RuntimeError("child IPC check timed out")
                actual = receive.recv()
                expected = {"request_id": request.request_id, "tokens": 2, "lengths": lengths}
                if actual != expected:
                    raise RuntimeError("round-trip mismatch: " + repr(actual))
                process.join(timeout=10)
                if process.is_alive() or process.exitcode != 0:
                    raise RuntimeError("child did not close normally")
                report["cases"].append(actual)
            finally:
                receive.close()
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
        if torch.cuda.is_initialized():
            raise RuntimeError("CUDA was initialized")
        report.update(passed=True, vllm_file=vllm.__file__, interpreter=sys.executable)
    except BaseException as exc:
        report.update(error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        (a.out / "ipc.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("PASS: actual ECR/native codec, spawned process and Request association")


if __name__ == "__main__":
    # Only this diagnostic process and its spawned child see the checkout path.
    sys.path.insert(0, str(ROOT))
    main()
