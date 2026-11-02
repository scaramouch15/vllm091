"""Shared, dependency-free parsing for per-type benchmark SLOs."""
import hashlib
import json
import math

REQUIRED = frozenset(("chat.ttft", "chat.tpot", "code.e2el", "summary.e2el"))


def parse_goodput_by_type(items):
    """Validate CLI milliseconds and return a canonical mapping in seconds."""
    values = {}
    for item in items:
        try:
            key, raw = item.split(":")
            value = float(raw)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"invalid type SLO: {item!r}") from exc
        if key not in REQUIRED:
            raise ValueError(f"unsupported type SLO: {key}")
        if key in values:
            raise ValueError(f"duplicate type SLO: {key}")
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"type SLO must be finite and positive: {key}")
        seconds = value / 1000.0
        if seconds <= 0 or not math.isfinite(seconds):
            raise ValueError(f"type SLO cannot be represented in seconds: {key}")
        values[key] = seconds
    if set(values) != REQUIRED:
        raise ValueError(f"missing type SLOs: {', '.join(sorted(REQUIRED - values.keys()))}")
    return dict(sorted(values.items()))


def check_type_args(goodput, items, dataset_name, output):
    if not items:
        return None
    if goodput:
        raise ValueError("--goodput and --goodput-by-type are mutually exclusive")
    if dataset_name != "mixed":
        raise ValueError("--goodput-by-type requires --dataset-name mixed")
    if not output:
        raise ValueError("--goodput-by-type requires --goodput-by-type-output")
    return parse_goodput_by_type(items)


def config_digest(config):
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def evaluate(config, attempts, results, duration_s):
    """Evaluate explicit formal attempt IDs; absent result IDs count as failures.

    Inputs use seconds, except output_tokens. An attempt carries request_id,
    task_type, and phase; results are keyed by request_id.
    """
    if set(config) != REQUIRED or any(not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(x) or x <= 0 for x in config.values()):
        raise ValueError("invalid normalized type SLO configuration")
    if isinstance(duration_s, bool) or not isinstance(duration_s, (int, float)) or not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("duration_s must be finite and positive")
    groups = {name: {"attempted": 0, "successful": 0, "attained": 0} for name in ("chat", "code", "summary")}
    seen = set()
    for attempt in attempts:
        rid = attempt.get("request_id")
        kind = attempt.get("task_type")
        if not isinstance(rid, str) or not rid or rid in seen or kind not in groups:
            raise ValueError("missing or duplicate request identity, or invalid type")
        seen.add(rid)
        phase = attempt.get("phase")
        if phase not in ("warmup", "formal"):
            raise ValueError("invalid or missing attempt phase")
        if phase == "warmup":
            continue
        group = groups[kind]
        group["attempted"] += 1
        result = results.get(rid)
        if not result or result.get("success") is not True:
            continue
        group["successful"] += 1
        def metric(name):
            value = result.get(name)
            return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0 else None
        latency = metric("latency_s")
        ttft = metric("ttft_s")
        if kind == "chat":
            count = result.get("output_tokens")
            valid_count = isinstance(count, int) and not isinstance(count, bool) and count >= 0
            tpot = None
            if valid_count and latency is not None and ttft is not None and latency >= ttft:
                tpot = 0.0 if count <= 1 else (latency - ttft) / (count - 1)
            attained = ttft is not None and tpot is not None and ttft <= config["chat.ttft"] and tpot <= config["chat.tpot"]
        else:
            attained = latency is not None and latency <= config[f"{kind}.e2el"]
        group["attained"] += int(attained)
    if set(results) - seen:
        raise ValueError("result without matching attempt")
    for group in groups.values():
        group["attainment_rate"] = group["attained"] / group["attempted"] if group["attempted"] else None
        group["request_goodput"] = group["attained"] / duration_s
    total_attempts = sum(g["attempted"] for g in groups.values())
    total_attained = sum(g["attained"] for g in groups.values())
    return {"types": groups, "overall": {"attempted": total_attempts, "attained": total_attained,
            "attainment_rate": total_attained / total_attempts if total_attempts else None,
            "request_goodput": total_attained / duration_s}}


def write_evaluation(path, config, attempts, results, duration_s, summary):
    """Save reproducible records without overwriting an earlier experiment."""
    document = {"schema_version": 1, "config_seconds": config,
                "config_sha256": config_digest(config), "duration_s": duration_s,
                "attempts": attempts, "results": results, "summary": summary}
    def clean(value):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [clean(v) for v in value]
        return value
    with open(path, "x", encoding="utf-8") as stream:
        json.dump(clean(document), stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def load_and_evaluate(path):
    with open(path, encoding="utf-8") as stream:
        data = json.load(stream)
    if data.get("schema_version") != 1:
        raise ValueError("unsupported or missing evaluation schema; historical data cannot be reliably associated")
    config = data["config_seconds"]
    if data.get("config_sha256") != config_digest(config):
        raise ValueError("configuration digest mismatch")
    return evaluate(config, data["attempts"], data["results"], data["duration_s"])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Recalculate per-type goodput from recorded attempts and results")
    parser.add_argument("record_file")
    args = parser.parse_args()
    print(json.dumps(load_and_evaluate(args.record_file), indent=2, sort_keys=True))
