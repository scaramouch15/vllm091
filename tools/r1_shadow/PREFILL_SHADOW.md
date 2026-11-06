# Prefill shadow observation

This research path is disabled by default. It adds no waiting reorder or new admission policy. With `VLLM_R1_SHADOW=1`, `VLLM_R1_ADMISSION_MODE=off`, and `VLLM_R1_PREFILL_SHADOW=1`, the Core evaluates a separate prefill model once for the same eligible candidates as a successful existing B shadow sweep. This is the skip-ineligible bounded sweep, not the admission contiguous prefix. Predictor errors are recorded and do not replace B's decision. `VLLM_R1_PREFILL_TIMING=1` independently enables front-end `schedule()` timing for both off/on comparison groups; it requires B shadow and admission off.

The B artifact SHA256 remains `23aff057712e7bec05db91d470af39a1b34eabddbf5852914710e1b8e4b96e1f`. When enabling prefill, specify a trusted local `VLLM_R1_PREFILL_MODEL`, its full `VLLM_R1_PREFILL_MODEL_SHA256`, and a non-sensitive `VLLM_R1_PREFILL_PROVENANCE` identifier referring to a reviewed training report. The latter must match `[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}`. Missing or invalid settings fail at startup. To reproduce the previous fixed artifact explicitly use SHA `4ab94aa14148645ab06dab7d7adfc49a882836b5305e96657d37b075863c021c`; the newer experimental candidate is `94868fdd98d9f3466247d3962d9459be0e9236042491755dedfe9cb2ccdd16f1`. Neither value establishes quality, trust or compatibility. No artifact or machine-specific path is committed.

Core reads the prefill file once, verifies the supplied hash, and deserializes those exact bytes. The schema must contain features `[num_prompt_tokens, running_count, kv_usage]`, quantiles `[0.5,0.9,0.95]`, and three estimators with three input features. Predictions must have the expected shape and finite values. Only load trusted joblib artifacts: hashes establish identity, not deserialization safety. The declared target is seconds from first native scheduled timestamp to first token output-batch timestamp. This target and unit are operator-attested from the training report identified by the provenance ID, not verified from the joblib schema. Headers record the expected SHA, whether it was verified by this process, provenance and target; prediction records include the verified SHA. Existing header environment metadata records Python and dependency versions. Keep the full training report and data identifiers separately and associate them with the run ID and provenance ID. See `PREFILL_PROVENANCE.md` for the evidence boundary.

Example (the placeholder path must be a trusted local file):

```sh
export VLLM_R1_PREFILL_SHADOW=1
export VLLM_R1_PREFILL_MODEL=/path/to/trusted/prefill.joblib
export VLLM_R1_PREFILL_MODEL_SHA256=94868fdd98d9f3466247d3962d9459be0e9236042491755dedfe9cb2ccdd16f1
export VLLM_R1_PREFILL_PROVENANCE=reviewed-training-report-v1
```

Existing B shadow and admission-mode settings still apply. Old configurations with prefill enabled but without explicit SHA and provenance now fail rather than automatically trusting an artifact. When prefill is disabled these variables are not required.

The actual server candidate and server joblib runtime were not available for
this local change. Before use, independently verify loading the trusted artifact
from an in-memory byte stream in the intended service Python environment (this
does not start a service):

```sh
"$R1_SERVICEPY" - <<'PY'
import hashlib, io, os, pathlib, platform
import joblib, numpy, scipy, sklearn
p = pathlib.Path(os.environ['VLLM_R1_PREFILL_MODEL'])
expected = os.environ['VLLM_R1_PREFILL_MODEL_SHA256'].lower()
data = p.read_bytes()
assert len(expected) == 64 and hashlib.sha256(data).hexdigest() == expected
artifact = joblib.load(io.BytesIO(data))  # only a trusted, independently identified artifact
assert artifact['features'] == ['num_prompt_tokens', 'running_count', 'kv_usage']
assert artifact['quantiles'] == [0.5, 0.9, 0.95]
assert len(artifact['models']) == 3
assert all(m.n_features_in_ == 3 and callable(m.predict) for m in artifact['models'])
print('SHA and stream load verified:', expected)
print('python/numpy/scipy/sklearn/joblib:', platform.python_version(), numpy.__version__, scipy.__version__, sklearn.__version__, joblib.__version__)
PY
```

Use the same trusted artifact provenance review as the production preflight;
this check alone cannot verify target semantics or prediction quality.

`shadow_prefill_v1` includes same-decision candidate IDs, quantiles, count and prediction time; `shadow_prefill_error_v1` and summary counters expose failures. Logging uses the existing bounded asynchronous writer and footer. Its model path and metadata are deliberately separate from the model bytes.

`tools/r1_shadow/analyze_prefill.py` requires the separately supplied fixed `input_split.jsonl`, SHA256 `7590e6f446b1cb6014bcbfe4bca65e51f3f7aafcd3cdc3086bd105714087bb5e`. Formal requests are grouped by verified input/prompt identity. The train group is training-input analysis, not out-of-sample; validation is internal, not a final independent test. `missing` counts formal requests in each group without a successful same-decision first-admission/native-finished match. Probe requests are excluded. Unknown, duplicate, and mismatched identities fail. Unadmitted or later-admitted candidates have no same-decision label and cannot be used as complete cohort coverage.

Pass positive `--min-sweeps`, `--min-calls`, and `--min-matches` explicitly; these are operational sample screens, not statistical sufficiency or quality guarantees. Invalid writer/collector state, prediction or logging errors fail closed. Insufficient runs produce `insufficient_samples`; otherwise results are exploratory only. `tools/r1_shadow/compare_prefill_runs.py` compares two independent off/on runs after identity and client configuration checks, and reports their distributions separately. These are not paired per-schedule deltas. `prediction_ns` is nested in `prefill_foreground_v1`, itself nested in `schedule_wall_v1`; do not sum their percentiles. Schedule timing excludes its final timing bookkeeping, background writing, GPU work, IPC and the client.

Real server verification, existing service configuration and candidate model provenance must be reviewed separately. This branch does not assert online CPU cost, fairness, TTFT, SLO or Goodput gains. Do not use final-test labels for tuning.
