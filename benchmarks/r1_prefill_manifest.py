"""Offline manifest splitting and collected-run validation for R1 prefill."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

TASKS = ("chat", "code", "summary")
SPLITS = ("train", "calibration", "test")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tokenizer_hashes(identifier):
    path = Path(identifier)
    if not path.is_dir():
        raise ValueError("candidate tokenizer_id must be an accessible local directory")
    names = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
             "vocab.json", "merges.txt", "tokenizer.model", "added_tokens.json",
             "vocab.txt", "sentencepiece.bpe.model")
    found = {name: file_hash(path / name) for name in names if (path / name).is_file()}
    if not found:
        raise ValueError("no recognized local tokenizer files")
    return found


def identity(row):
    a, b = row.get("input_hash"), row.get("prompt_hash")
    if any(not isinstance(x, str) or len(x) != 64 or
           any(c not in "0123456789abcdef" for c in x) for x in (a, b)):
        raise ValueError("invalid input_hash or prompt_hash")
    if row.get("task_type") not in TASKS:
        raise ValueError("missing or invalid task_type")
    if not isinstance(row.get("prompt"), str) or digest(row["prompt"].encode()) != b:
        raise ValueError("prompt hash/content mismatch")
    if type(row.get("prompt_len")) is not int or row["prompt_len"] <= 0:
        raise ValueError("invalid prompt_len")
    if type(row.get("max_tokens")) is not int or row["max_tokens"] <= 0:
        raise ValueError("invalid max_tokens")
    return a, b


def exclusion(path):
    """Identity-only JSON or JSONL; never inspect labels or evaluation outputs."""
    p = Path(path)
    with p.open(encoding="utf-8") as h:
        if p.suffix == ".jsonl":
            rows = [json.loads(line) for line in h if line.strip()]
        else:
            obj = json.load(h)
            rows = obj if isinstance(obj, list) else obj.get("requests", [])
    inputs, prompts = set(), set()
    for r in rows:
        for key, target in (("input_hash", inputs), ("prompt_hash", prompts)):
            v = r.get(key)
            if v is not None:
                if not isinstance(v, str) or len(v) != 64 or any(c not in "0123456789abcdef" for c in v):
                    raise ValueError(f"invalid {key} in {p}")
                target.add(v)
    if not inputs and not prompts:
        raise ValueError(f"empty identity exclusion: {p}")
    return inputs, prompts


def partition(manifest, exclusions, seed, target=1800, old_test_complete=False):
    run, rows = manifest.get("run"), manifest.get("requests")
    if not isinstance(run, dict) or not isinstance(rows, list) or not rows:
        raise ValueError("invalid mixed manifest")
    if not isinstance(run.get("files"), dict) or not isinstance(run.get("tokenizer"), dict):
        raise ValueError("missing source or tokenizer configuration")
    indices = [r.get("request_index") for r in rows]
    if any(type(i) is not int for i in indices) or len(set(indices)) != len(indices):
        raise ValueError("missing or duplicate request_index")
    pairs = [identity(r) for r in rows]
    if len(set(pairs)) != len(pairs):
        raise ValueError("duplicate input/prompt identity")
    parent = list(range(len(rows)))
    def find(x):
        while x != parent[x]:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        a, b = find(a), find(b)
        if a != b: parent[b] = a
    seen = {}
    for i, (a, b) in enumerate(pairs):
        for key in (("input", a), ("prompt", b)):
            if key in seen: union(i, seen[key])
            else: seen[key] = i
    groups = defaultdict(list)
    for i in range(len(rows)): groups[find(i)].append(i)
    overlaps = {}
    excluded = set()
    for name, (ins, prs) in exclusions.items():
        matches = {i for i, (a, b) in enumerate(pairs) if a in ins or b in prs}
        overlaps[name] = {"candidate_rows": len(matches),
                          "unique_input_hashes": len({pairs[i][0] for i in matches})}
        excluded.update(find(i) for i in matches)
    ordered = sorted((g for root, g in groups.items() if root not in excluded),
                     key=lambda g: digest((str(seed) + ":" + min(pairs[i][0] for i in g)).encode()))
    total = sum(len({pairs[i][0] for i in g}) for g in ordered)
    assigned = {s: [] for s in SPLITS}
    sizes = {s: 0 for s in SPLITS}
    limits = (total * .6, total * .8)
    cumulative = 0
    for group in ordered:
        split = SPLITS[0] if cumulative < limits[0] else SPLITS[1] if cumulative < limits[1] else SPLITS[2]
        assigned[split].extend(rows[i] for i in group)
        n = len({pairs[i][0] for i in group}); sizes[split] += n; cumulative += n
    source_hashes = {}
    for entries in run["files"].values():
        for f in entries:
            path = f["source_file"]
            source_hashes[path] = file_hash(path)
    summary = {"candidate_rows": len(rows), "by_type": dict(Counter(r["task_type"] for r in rows)),
               "by_source": dict(Counter(r.get("source_dataset", "unknown") for r in rows)),
               "exclusion_overlaps": overlaps, "excluded_groups": len(excluded),
               "unique_inputs": total, "target": target, "shortfall": max(0, target-total),
               "split_unique_inputs": sizes,
               "lengths": {s: {"count": len(rs), "min": min((r["prompt_len"] for r in rs), default=None),
                                 "max": max((r["prompt_len"] for r in rs), default=None),
                                 "mean": sum(r["prompt_len"] for r in rs)/len(rs) if rs else None,
                                 "types": dict(Counter(r["task_type"] for r in rs))}
                           for s, rs in assigned.items()},
               "old_test_identity_completeness": ("operator_attested_not_independently_verified"
                                                  if old_test_complete else "unverified")}
    return {"schema": "r1-prefill-manifest-v1", "hash_algorithm": "sha256-utf8",
            "split_seed": str(seed), "candidate_run": run, "source_sha256": source_hashes,
            "tokenizer_sha256": tokenizer_hashes(run["tokenizer_id"]),
            "candidate_requests": rows, "splits": assigned, "summary": summary}


def replay(approved, dataset, tokenizer, tokenizer_id, tokenizer_mode, split):
    if approved.get("schema") != "r1-prefill-manifest-v1" or split not in SPLITS:
        raise ValueError("invalid approved manifest or split")
    run = approved["candidate_run"]
    if (dataset.random_seed != run["seed"] or dataset.counts != run["counts"] or
        dataset.output_len != run["mixed_output_len"] or
        str(tokenizer_id) != run["tokenizer_id"] or tokenizer_mode != run["tokenizer_mode"]):
        raise ValueError("replay configuration mismatch")
    for path, expected in approved["source_sha256"].items():
        if file_hash(path) != expected: raise ValueError("source file hash mismatch: " + path)
    if tokenizer_hashes(tokenizer_id) != approved["tokenizer_sha256"]:
        raise ValueError("tokenizer file hash mismatch")
    candidates = dataset.sample(tokenizer, sum(dataset.counts.values()))
    if dataset.tokenizer_info != run["tokenizer"] or dataset.files != run["files"]:
        raise ValueError("tokenizer or source configuration mismatch")
    expected = approved["candidate_requests"]
    if [r.metadata for r in candidates] != expected:
        raise ValueError("candidate reconstruction mismatch")
    chosen = approved["splits"][split]
    keys = [identity(r) for r in chosen]
    if not keys or len(keys) != len(set(keys)):
        raise ValueError("empty or duplicate approved split")
    # Verify all splits are exact subsets and have no shared hash across splits.
    all_expected = {tuple(identity(r)): r for r in expected}
    seen_a, seen_b, seen_pairs = {}, {}, set()
    for name in SPLITS:
        for row in approved["splits"][name]:
            a,b = identity(row)
            if (a,b) not in all_expected or (a,b) in seen_pairs or row != all_expected[(a,b)]:
                raise ValueError("unexpected or duplicate approved request")
            if (a in seen_a and seen_a[a] != name) or (b in seen_b and seen_b[b] != name):
                raise ValueError("cross-split identity overlap")
            seen_a[a]=name; seen_b[b]=name; seen_pairs.add((a,b))
        if len({row["input_hash"] for row in approved["splits"][name]}) != approved["summary"]["split_unique_inputs"][name]:
            raise ValueError("approved split count mismatch")
    if len(seen_a) != approved["summary"]["unique_inputs"]:
        raise ValueError("approved total count mismatch")
    by_key = {identity(r.metadata): r for r in candidates}
    result = [by_key[key] for key in keys]
    if [r.metadata for r in result] != chosen: raise ValueError("approved content mismatch")
    dataset.requests = result
    return result


def validate_collections(approved, split, reports):
    chosen = approved["splits"][split]
    expected = {identity(r): r for r in chosen}
    if len(expected) != len(chosen): raise ValueError("duplicate approved identities")
    summaries, batches = [], {}
    run_ids = set()
    prior_report_runs = set()
    for report in reports:
        errors = []
        report_runs = set()
        if (report.get("recording_status") != "complete" or not report.get("client_batch_counts_complete")
            or report.get("unassociated_records") or not report.get("collection_complete")):
            errors.append("incomplete_recording_or_batch")
        formal, probes, seen, seen_ids = 0, 0, set(), set()
        for row in report.get("requests", []):
            key_id = (row.get("run_id"), row.get("engine_request_id"))
            if all(key_id):
                if key_id in seen_ids: errors.append("duplicate_engine_request_id")
                seen_ids.add(key_id)
            attempts = row.get("attempts", [])
            if len(attempts) != 1:
                errors.append("missing_or_duplicate_attempt"); continue
            att = attempts[0]
            if att.get("phase") == "probe": probes += 1; continue
            if att.get("phase") != "main": errors.append("invalid_phase"); continue
            formal += 1
            rid, eid = row.get("run_id"), row.get("engine_request_id")
            if not rid or not eid or att.get("run_id") != rid or att.get("engine_request_id") != eid:
                errors.append("request_identity_mismatch"); continue
            run_ids.add(rid)
            report_runs.add(rid)
            try: key = identity(att["sample"])
            except (ValueError, KeyError, TypeError): errors.append("invalid_sample"); continue
            if key in seen: errors.append("duplicate_formal_identity")
            seen.add(key)
            if key not in expected or att["sample"] != expected[key]: errors.append("unexpected_or_changed_sample")
            for field in ("admissions", "finished", "client_results"):
                if len(row.get(field, [])) != 1: errors.append("missing_or_duplicate_" + field)
                elif (row[field][0].get("run_id") != rid or
                      row[field][0].get("engine_request_id") != eid):
                    errors.append("joined_identity_mismatch")
            if len(row.get("admissions", [])) == 1:
                batch = row["admissions"][0].get("snapshot", {}).get("decision_batch_id")
                if not batch: errors.append("missing_decision_batch")
                else:
                    bid = (rid,batch)
                    if bid in batches and batches[bid] != split: errors.append("batch_cross_split")
                    batches[bid] = split
            if len(row.get("finished", [])) == 1:
                f = row["finished"][0]; t = f.get("timestamps", {}); native = f.get("native_stats") or {}
                values = [t.get(k) for k in ("queued_ts", "scheduled_ts", "first_token_ts", "last_token_ts")]
                if (f.get("stats_status") != "valid" or any(type(v) not in (int,float) or not math.isfinite(v) for v in values)
                    or values != sorted(values) or type(native.get("prefill_time")) not in (int,float)
                    or not math.isfinite(native["prefill_time"]) or not math.isclose(native["prefill_time"], values[2]-values[1], rel_tol=1e-9, abs_tol=1e-9)):
                    errors.append("invalid_prefill_time")
            if row.get("status") != ["complete"]: errors.append("incomplete_request")
        if seen != set(expected) or formal != len(expected): errors.append("formal_manifest_mismatch")
        if len(report_runs) != 1: errors.append("expected_single_run")
        if report_runs & prior_report_runs: errors.append("duplicate_run_across_reports")
        prior_report_runs.update(report_runs)
        summaries.append({"formal_attempts": formal, "probe_attempts": probes, "errors": sorted(set(errors))})
    return {"split": split, "runs": sorted(run_ids),
            "decision_batches": [list(x) for x in sorted(batches)], "reports": summaries,
            "valid": bool(reports) and all(not r["errors"] for r in summaries)}


def compare_validations(results, require_all=True):
    """Compare local results; only a three-split set may claim completeness."""
    errors = set()
    by_split = {}
    hashes = set()
    for result in results:
        split = result.get("split")
        if split not in SPLITS:
            errors.add("invalid_split")
            continue
        if split in by_split:
            errors.add("duplicate_split:" + split)
        else:
            by_split[split] = result
        if result.get("valid") is not True:
            errors.add("invalid_validation:" + split)
        value = result.get("approved_sha256")
        if (not isinstance(value, str) or len(value) != 64 or
                any(c not in "0123456789abcdef" for c in value)):
            errors.add("invalid_approved_sha256:" + split)
        else:
            hashes.add(value)
        if (not isinstance(result.get("runs"), list) or not result["runs"] or
                any(not isinstance(x, str) or not x for x in result["runs"]) or
                len(set(result["runs"])) != len(result["runs"])):
            errors.add("invalid_runs:" + split)
        if not isinstance(result.get("decision_batches"), list):
            errors.add("invalid_batches:" + split)
    if len(hashes) > 1:
        errors.add("approved_manifest_mismatch")
    missing = sorted(set(SPLITS) - set(by_split))
    if require_all:
        errors.update("missing_split:" + split for split in missing)
    for i, left in enumerate(SPLITS):
        if left not in by_split:
            continue
        for right in SPLITS[i + 1:]:
            if right not in by_split:
                continue
            a, b = by_split[left], by_split[right]
            if (isinstance(a.get("runs"), list) and isinstance(b.get("runs"), list) and
                    all(isinstance(x, str) for x in a["runs"] + b["runs"]) and
                    set(a["runs"]) & set(b["runs"])):
                errors.add("run_id_cross_split:" + left + ":" + right)
            def pairs(result):
                batches = result.get("decision_batches", [])
                if (not isinstance(batches, list) or any(not isinstance(x, list) or len(x) != 2 or
                       any(not isinstance(v, str) or not v for v in x) or
                       x[0] not in result.get("runs", []) for x in batches)):
                    errors.add("invalid_batches:" + result["split"])
                    return set()
                return {tuple(x) for x in batches}
            if pairs(a) & pairs(b):
                errors.add("batch_cross_split:" + left + ":" + right)
    return {"valid": not errors, "complete": not missing and not errors,
            "compared_splits": sorted(by_split), "missing_splits": missing,
            "errors": sorted(errors)}


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest="cmd", required=True)
    s=sub.add_parser("split"); s.add_argument("--candidate", required=True); s.add_argument("--train-exclusion", required=True); s.add_argument("--validation-exclusion", required=True); s.add_argument("--diagnostic-exclusion", required=True); s.add_argument("--old-test-identities"); s.add_argument("--old-test-complete", action="store_true"); s.add_argument("--seed", required=True); s.add_argument("--target", type=int, default=1800); s.add_argument("--output", required=True)
    v=sub.add_parser("validate"); v.add_argument("--approved", required=True); v.add_argument("--split", choices=SPLITS, required=True); v.add_argument("--merged", nargs="+", required=True); v.add_argument("--other-validation", nargs="*", default=[]); v.add_argument("--output", required=True)
    c=sub.add_parser("compare"); c.add_argument("--validations", nargs="+", required=True); c.add_argument("--output", required=True)
    a=p.parse_args()
    if a.cmd == "split":
        paths={"train":a.train_exclusion,"validation":a.validation_exclusion,"diagnostic":a.diagnostic_exclusion}
        if a.old_test_identities: paths["old_test"]=a.old_test_identities
        if a.old_test_complete and not a.old_test_identities: p.error("old-test-complete requires identity manifest")
        result=partition(read_json(a.candidate),{k:exclusion(v) for k,v in paths.items()},a.seed,a.target,a.old_test_complete)
        result["candidate_sha256"]=file_hash(a.candidate); result["exclusion_sha256"]={k:file_hash(v) for k,v in paths.items()}
    elif a.cmd == "validate":
        result=validate_collections(read_json(a.approved),a.split,[read_json(x) for x in a.merged])
        result["approved_sha256"] = file_hash(a.approved)
        comparison = compare_validations([result]+[read_json(path) for path in a.other_validation], require_all=False)
        result["cross_split_check"] = ("complete" if comparison["complete"] else
                                       "partial" if a.other_validation else "not_performed")
        result["cross_split_errors"] = comparison["errors"]
        if not comparison["valid"]: result["valid"] = False
    else:
        result=compare_validations([read_json(path) for path in a.validations])
    with open(a.output,"x",encoding="utf-8") as f: json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False); f.write("\n")
    if a.cmd in ("validate", "compare") and not result["valid"]: raise SystemExit(1)

if __name__ == "__main__": main()
