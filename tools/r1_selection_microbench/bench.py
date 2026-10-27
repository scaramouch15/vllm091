"""CPU-only, standard-library selection microbenchmark."""
import argparse
import gc
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

from selection import METHODS, fresh_waiting, maintain_window, pure_select, validate_scores


def parse_positive_list(value):
    try:
        items = list(dict.fromkeys(int(x) for x in value.split(",")))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not items or any(x <= 0 for x in items):
        raise argparse.ArgumentTypeError("values must be positive")
    return items


def scores_for(k, kind, seed):
    rng = random.Random(seed + k)
    if kind == "random":
        return [rng.random() for _ in range(k)]
    if kind == "ascending":
        return list(range(k))
    if kind == "descending":
        return list(range(k, 0, -1))
    return [rng.randrange(4) for _ in range(k)]


def percentile(values, p):
    ordered = sorted(values)
    return ordered[max(0, (len(ordered) * p + 99) // 100 - 1)]


def git_output(*args):
    try:
        return subprocess.check_output(["git", *args], cwd=Path(__file__).resolve().parents[2],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def measure(layer, n, scores, attempts, method, batch):
    # All independent input state allocation and preparation is outside timing.
    if layer == "pure":
        entries = [(score, i, i) for i, score in enumerate(scores)]
        workloads = [entries] * batch  # immutable inputs; selector makes its own structure
    else:
        workloads = [fresh_waiting(n) for _ in range(batch)]
    start = time.perf_counter_ns()
    if layer == "pure":
        for entries in workloads:
            pure_select(entries, attempts, method)
    else:
        for waiting in workloads:
            maintain_window(waiting, scores, attempts, attempts, None, method)
    return (time.perf_counter_ns() - start) / batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=parse_positive_list, default=[8, 24, 32, 128, 512, 1024])
    parser.add_argument("--tail-k", type=parse_positive_list, default=[24, 128])
    parser.add_argument("--tail-n", type=parse_positive_list, default=[4096, 65536])
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batch", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 1 or args.batch < 1:
        parser.error("warmup must be nonnegative; repeats and batch must be positive")
    cases = []
    for k in args.k:
        for a in sorted({min(k, x) for x in (1, 2, 4, 8, max(1, k // 2), k)}):
            for kind in ("random", "ascending", "descending", "ties"):
                for layer in ("pure", "window"):
                    cases.append((layer, k, k, a, kind))
    for k in args.tail_k:
        for n in args.tail_n:
            if n < k:
                parser.error("each tail N must be >= each tail K")
            for a in sorted({1, k}):
                cases.append(("window_tail", n, k, a, "random"))
    names = list(METHODS)
    rows = []
    gc_initial = gc.isenabled()
    try:
        gc.disable()
        for case_index, (layer, n, k, a, kind) in enumerate(cases):
            scores = scores_for(k, kind, args.seed)
            validate_scores(scores)
            actual_layer = "pure" if layer == "pure" else "window"
            # Rotate first method by both case and repeat; each round runs all three.
            for w in range(args.warmup):
                for offset in range(3):
                    name = names[(case_index + w + offset) % 3]
                    measure(actual_layer, n, scores, a, METHODS[name], 1)
            samples = {name: [] for name in names}
            orders = []
            for rep in range(args.repeats):
                order = [names[(case_index + rep + offset) % 3] for offset in range(3)]
                orders.append(order)
                for name in order:
                    samples[name].append(measure(actual_layer, n, scores, a,
                                                 METHODS[name], args.batch))
            summary = {name: {"median_ns_per_call": statistics.median(values),
                              "p95_batch_mean_ns_per_call": percentile(values, 95),
                              "p99_batch_mean_ns_per_call": percentile(values, 99),
                              "min_ns_per_call": min(values), "max_ns_per_call": max(values)}
                       for name, values in samples.items()}
            rows.append({"layer": layer, "N": n, "K": k, "A": a, "M": a,
                         "distribution": kind, "order_by_repeat": orders,
                         "raw_batch_mean_ns_per_call": samples, "summary": summary})
    finally:
        if gc_initial:
            gc.enable()
    result = {"metadata": {"python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "cpu": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "unknown"),
        "cpu_logical_count": os.cpu_count(), "git_sha": git_output("rev-parse", "HEAD"),
        "tracked_dirty": bool(git_output("status", "--porcelain", "--untracked-files=no")),
        "arguments": vars(args) | {"output": str(args.output)}, "gc_initially_enabled": gc_initial,
        "gc_during_timing": "disabled equally for all methods",
        "timer": "perf_counter_ns; elapsed batch / calls; nearest-rank over batch means",
        "attempt_policy": "all attempts succeed, capacity=A; no allocator model",
        "sample_unit": "one batch mean per method/case/repeat, not individual-call latency"},
        "cases": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} cases to {args.output}")


if __name__ == "__main__":
    main()
