"""Per-Scheduler state for the opt-in mixed-type SLO experiment.

This module only stores metadata; it does not take scheduling decisions.
"""
import hashlib
import json
import math

_KEYS = frozenset(("chat.ttft", "chat.tpot", "code.e2el", "summary.e2el"))
_TYPES = frozenset(("chat", "code", "summary"))


def normalize_slos(values):
    if not isinstance(values, dict) or set(values) != _KEYS:
        raise ValueError("exactly four supported per-type SLOs are required")
    result = {}
    for key in sorted(_KEYS):
        value = values[key]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value <= 0):
            raise ValueError(f"invalid finite positive seconds value: {key}")
        result[key] = float(value)
    return result


def slo_digest(values):
    payload = json.dumps(values, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ExperimentState:
    def __init__(self):
        self._run_id = None
        self._values = None
        self._digest = None

    def initialize(self, run_id, values, expected_digest):
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id must be a nonempty string")
        normalized = normalize_slos(values)
        digest = slo_digest(normalized)
        if expected_digest != digest:
            raise ValueError("SLO configuration digest mismatch")
        if self._run_id is not None and (run_id != self._run_id or normalized != self._values):
            raise ValueError("experiment already initialized with a different run or SLO configuration")
        self._run_id, self._values, self._digest = run_id, normalized, digest
        return {"run_id": self._run_id, "config_sha256": self._digest}

    def validate_request_type(self, request_type):
        if self._run_id is None:
            raise ValueError("experiment is not initialized")
        if not isinstance(request_type, str) or request_type not in _TYPES:
            raise ValueError("r1_request_type must be chat, code or summary")
        return request_type

    def snapshot(self):
        if self._run_id is None:
            return None
        return {"run_id": self._run_id, "config_sha256": self._digest,
                "config_seconds": dict(self._values)}
