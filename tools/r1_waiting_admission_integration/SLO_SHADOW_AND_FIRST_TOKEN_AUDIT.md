# SLO shadow observation and first-token target audit

## SLO shadow (opt in)

Set `VLLM_R1_ADMISSION_MODE=observe`, `VLLM_R1_SHADOW=1`, and
`VLLM_R1_SLO_SHADOW=1`; initialize the existing `VLLM_R1_TYPE_SLO=1`
configuration. The new switch defaults to `0`, accepts only `0`/`1`, and `1`
requires `observe`: it cannot apply a waiting reorder. Existing shadow model,
FCFS, feature, environment, and candidate requirements remain unchanged. If
shadow or admission is disabled by an unsupported configuration, existing
startup warnings apply and no SLO record is produced. No model is loaded in CPU
unit tests and no additional dynamic inference is invoked.

The observation is a nested `slo_shadow_v1` member of the existing
`r1_admission_v1` event. Its source is the admission contiguous eligible
prefix, not the older skip-through shadow sweep. One record is emitted per
admission frame, including `interval_skip`, preemption and capacity outcomes;
`opportunity_id` identifies eligible opportunities. Unscanned candidate count
is null, not zero. The barrier describes the first already-inspected
ineligible request; `window_outside_checked` counts positions not examined,
not a statement about their eligibility. Actual attempts, admissions and exit
reason are the enclosing admission record's fields. Loss uses the existing
shadow writer's counters/footer; a missing record cannot be inferred to mean an
empty window.

With native `log_stats`, enabling this switch preserves the same first QUEUED
monotonic value already used by native events, on each initial add; it does
not consume events or resample the clock. Existing `TIME_OBSERVE` behavior is
preserved. With stats disabled, times are null with `native_stats_disabled`.
The schedule timestamp was sampled before this round's running loop, so the
running loop, dynamic prediction and logging of this round are excluded from
`wait_s`; it is not the actual admission timestamp. No cross-clock subtraction
is performed. The observation is constructed after the admission loop from the
bounded candidate references and only the already validated current decision's
P90; skipped, failed and invalid decisions never reuse a previous prediction.

For code/summary with valid data, `wait_plus_p90_s = wait_s + inference_p90_s`
and `reference_slack_s = e2e_slo_s - wait_plus_p90_s`. These are service-side
components compared with the numerical client SLO, not client E2E predictions;
negative is not proven violation and positive is not a guarantee. P90
coverage is empirical, not guaranteed. Chat logs both SLOs and marks TTFT and
TPOT risk `not_estimable`. Comparable pairs and differing reference values are
reported only within the same code/summary type, without defining a cross-type
formula. Preempted/already-admitted requests truncate the prefix; no resumed
request is interpreted as a first-admission prediction.

`slo_construct_ns` measures bounded construction only and is included inside
existing `admission_log_foreground_v1`; the writer's `log_enqueue` measurement
includes its enqueue call and is reported separately. Existing
`admission_schedule_foreground_v1` encloses these frontend operations; its own
final bookkeeping/return and background disk writes are not included. Never add
the nested intervals or their quantiles. Tests use stubbed Scheduler resources,
not a real vLLM EngineCore process or GPU service.

Example (synthetic seconds): an event with `candidate_count=2`, a code row with
`wait_s=5`, `inference_p90_s=3`, `e2e_slo_s=30` yields
`wait_plus_p90_s=8`, `reference_slack_s=22`; an accompanying chat row has its
TTFT/TPOT SLOs and both risk statuses `not_estimable`. Missing timestamps,
configurations or predictions are null with explicit reasons. JSONL can contain
request IDs and types but not prompt contents.

## First-token label audit (read only)

The current time head was trained on `inference_time`, the native first
SCHEDULED to last token-batch timestamp interval; the length head predicts
actual output-token count under the requested 2048 maximum. The frozen split
builder asserts that training label equals `last_token_ts - scheduled_ts` and
stores it in `inference_time`; it does not retain the individual timestamps in
its output rows. FrontendCollector may export `queued_ts`, `scheduled_ts`,
`first_token_ts`, `last_token_ts` and joins by run/engine request ID; admission
records provide decision_batch_id. Client collector stores relative `ttft`,
not a universally comparable absolute first-token timestamp. A compatible raw
merged collection with complete provenance could construct
`first_token_ts - first_scheduled_ts` in seconds. This is a batch-associated
Core output timestamp recorded when the first output is processed, not a
precise GPU token-generation timestamp. Scheduled time is sampled before the
running loop. It is not complete client TTFT and cannot predict TPOT.

Exclude records with missing/duplicate joins, failed/cancelled/incomplete
collection, absent/nonfinite/out-of-order timestamps, incompatible run/clock
origin, missing first token, or unknown decision identity; report all counts.
Review length-capped samples separately rather than asserting they are natural
finishes. Existing exported inference-time-only split files cannot reconstruct
first-token times; availability and completeness of original raw merged logs
must be checked on the server. No absolute monotonic values from different
hosts should be subtracted. Check that the labels/features precede their use;
never add actual output length, first-token outcome or online real type to the
feature set. Split without decision_batch_id overlap; also check input overlap
and grouped connected components. The historical split groups by input_hash
and explicitly allows decision_batch_id overlap, so it is not sufficient to
claim batch-disjoint validation. If group constraints collapse the data or
sample coverage is inadequate, collect independently separated runs rather
than weakening the split.

Compare an independent first-token head plus unchanged completion head against
a multi-output model on the same samples, features, batch-grouped split and
preprocessing. Independent heads permit separate release and reuse; multi-task
training may share representation but requires extra implementation, calibration
and coordinated deployment, and is not assumed superior. Report per-target
MAE, signed error/underestimation, pinball loss and P90 empirical coverage,
sample/valid/missing rates by workload and input-length bin; assess new-run
generalization. Measure actual frontend and Scheduler inference, batch sizes,
logging and schedule wall time; do not add nested quantiles. No new training or
feature selection was performed here.

Locally verifiable artifacts: the original A CPU-comparison model SHA256 is
`64fb5bca549524ad5d7b975852bc1298f4bf140072d3157d98a222ed27019292`; the B
new310 artifact is `23aff057712e7bec05db91d470af39a1b34eabddbf5852914710e1b8e4b96e1f`,
which is the exact hash currently enforced in source. The A800 AB/BA report
records both model artifacts being exercised in an offline CPU inference
comparison, not online scheduling. The B execution record says trained with
sklearn 1.7.2 and native inference-time labels; its training summary identifies
the B SHA. Recorded artifact timestamps are not proof of training time. An
older research note reports length-predictor underestimation on an early
500-request test, not necessarily this model or target. The separately reported
server candidate-A SHA `901646db31702c90495c610c40a3c3d47fa4ea3b1f48446489d7e059e7b68412`
is different and its exact model, training origin, timestamp and online use
remain unverified here. Do not replace the enforced hash or model based on a
filename. On the server, inspect the actual configured model path and code SHA,
run `sha256sum` on the explicitly named file, and inspect only authorized raw
collection manifests, timestamps, and training records; do not load joblib.

Illustrative nested event (values are synthetic, fields abbreviated):

```json
{"candidate_source":"admission_contiguous_prefix","candidate_count":2,
 "valid_prediction_count":2,"opportunity_id":1,
 "same_type_comparable_pair":false,
 "candidates":[
   {"request_id":"example-code","request_type":"code","wait_s":5.0,
    "inference_p90_s":3.0,"e2e_slo_s":30.0,"wait_plus_p90_s":8.0,
    "reference_slack_s":22.0,"missing_reason":null},
   {"request_id":"example-chat","request_type":"chat","wait_s":5.0,
    "chat_ttft_slo_s":2.0,"chat_tpot_slo_s":0.03,
    "ttft_risk_status":"not_estimable","tpot_risk_status":"not_estimable"}]}
```

The snapshot of Scheduler SLO metadata is read within `trigger_prefix_ns`;
`slo_construct_ns` is the later bounded record construction. With the new
switch disabled neither operation is added. `max_tokens=128` requests from the
previous six-request type/SLO test did not meet this predictor's existing
`max_tokens=2048` static gate, and do not demonstrate prediction coverage.
For later Linux verification use an already authorized model with an exactly
matching enforced SHA/environment and requests meeting all existing static
conditions; verify startup, coverage/missing reasons, log footer and queue
behavior before any performance experiment.

The saved B quality-comparison file reports 1,440 validation rows and an
inference-time P90 empirical coverage of 0.9145833 for both old and new
processed predictions in that fixed offline comparison. This does not prove
calibration on a new workload or show that every subgroup is calibrated. The
historical length-predictor note reports a 500-request early test with 55%
underestimation of output length; it must not be transferred to this time head.
The later B training artifact's exact training wall-clock date is not proven by
its current filesystem modification time; the dated 2026-10-24 AB/BA report is
an offline CPU comparison using both artifacts.
