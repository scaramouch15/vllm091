"""Validate the actual shadow inference code against B's own saved validation.

No training, test split, LLM, CUDA or service startup. Use the verified B/service
clone interpreter. Outputs are private and must be outside the source checkout.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
DATA_SHA256 = "1b7fe1e7e8036bc7f7b943866d1a2a28ace2dc7bd6e5875e44f5ef6e2e6f5d9f"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line]


def comparison(actual, expected):
    import numpy as np
    actual, expected = np.asarray(actual), np.asarray(expected)
    require(actual.shape == expected.shape and np.isfinite(actual).all()
            and np.isfinite(expected).all(), "invalid prediction shape/value")
    delta = np.abs(actual - expected)
    relative = np.divide(delta, np.abs(expected), out=np.zeros_like(delta), where=expected != 0)
    relative[(expected == 0) & (delta != 0)] = np.inf
    fail = ~np.isclose(actual, expected, rtol=1e-10, atol=1e-10)
    maximum = float(relative.max(initial=0))
    return {"max_absolute_error": float(delta.max(initial=0)),
            "max_relative_error": maximum if np.isfinite(maximum) else "inf",
            "failed_records": int(np.any(fail, axis=1).sum()),
            "failed_elements": int(fail.sum())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--validation", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    require(not a.out.resolve().is_relative_to(ROOT), "keep outputs outside repository")
    a.out.mkdir(parents=True, exist_ok=False)
    report = {"passed": False, "rtol": 1e-10, "atol": 1e-10,
              "python": platform.python_version(), "interpreter": sys.executable}
    try:
        source = ROOT / "vllm/v1/metrics/r1_shadow.py"
        spec = importlib.util.spec_from_file_location("standalone_r1_shadow", source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        require(sha(a.validation) == DATA_SHA256, "wrong validation SHA256")
        rows, previous = read_rows(a.validation), read_rows(a.reference)
        identities = lambda rs: [(r["run_id"], r["engine_request_id"]) for r in rs]
        require(len(rows) == len(previous) == 1440, "expected 1440 records")
        require(identities(rows) == identities(previous)
                and len(set(identities(rows))) == 1440, "identity/order mismatch")
        require(all(r["split"] == "validation" and r["max_tokens"] == 2048 for r in rows),
                "wrong split/output limit")
        require(all(hashlib.sha256(r["prompt"].encode()).hexdigest() == r["prompt_hash"]
                    for r in rows), "prompt hash mismatch")
        front = module.Predictor(a.model, "frontend")
        core = module.Predictor(a.model, "core")
        length = [front.static(r["prompt"], r["num_prompt_tokens"])[0] for r in rows]
        import numpy as np
        matrix = np.asarray([[r["num_prompt_tokens"], r["running_count"], r["kv_usage"], *q]
                             for r, q in zip(rows, length)], dtype=np.float64)
        predicted = core.dynamic(matrix)
        report.update(model_sha256=sha(a.model), validation_sha256=sha(a.validation),
                      reference_sha256=sha(a.reference), source_sha256=sha(source), n=1440,
                      identity_order_verified=True, threadpools=front.threadpools,
                      source_commit=subprocess.check_output(
                          ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
                      stages={"length": comparison(length, [r["predicted_length_quantiles"] for r in previous]),
                              "time": comparison(predicted, [r["predicted_time_quantiles"] for r in previous])})
        # Same B runtime, singleton versus actual scheduler-sized batches.
        for size in (1, 8, 32, 64, 128):
            batches = [q for offset in range(0, len(matrix), size)
                       for q in core.dynamic(matrix[offset:offset+size])]
            report["stages"]["batch_" + str(size)] = comparison(batches, predicted)
        require(not any(v["failed_elements"] for v in report["stages"].values()),
                "strict self-reference comparison failed")
        report["passed"] = True
    except BaseException as exc:
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        raise
    finally:
        (a.out / "validation.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("PASS: shadow runtime, 1440 B validation records; no training or LLM")


if __name__ == "__main__":
    main()
