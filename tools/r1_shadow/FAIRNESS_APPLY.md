# Optional fairness order apply

The new switch `VLLM_R1_FAIRNESS_APPLY=1` is disabled by default. It requires
`VLLM_R1_ADMISSION_MODE=apply`, `VLLM_R1_ADMISSION_HEAD_POLICY=type_slo`,
`VLLM_R1_ADMISSION_TIME_OBSERVE=1` and an explicit finite positive
`VLLM_R1_FAIRNESS_WAIT_LIMIT_SECONDS`. The existing type-SLO, B shadow,
prefill shadow, native log_stats, and model configuration requirements still
apply. `VLLM_R1_ADMISSION_FEASIBLE_PROTECTION=1` and
`VLLM_R1_FAIRNESS_COMPARE_OBSERVE=1` conflict with this switch and are rejected.
The observe-only comparison switch is not an apply switch. No experimental
threshold is provided or recommended by this option.

The existing admission continuous eligible prefix, state snapshot and the
single B/prefill batch are reused. `observe_fairness_order` supplies exactly
the fixed-wait order: protected requests in original order, then unprotected
nonnegative scores ascending with stable ties, then negative scores in original
order. If all scores are negative and none protected, the full prefix remains
in original order. Scores estimate margin rather than establish SLO outcome.
Incomplete data, prediction failures, identity errors or invalid sort results
fall back before queue mutation. Native token/capacity/KV checks, preemption,
skips, and restoration remain in the existing Scheduler path.

In `r1_admission_v1`, `score_rule=type_slo_fairness_apply_v1` identifies the
selected policy; `candidate_ids`, `sorted_ids`, `applied_order_changed`, and
`actual_admission_order` describe original, planned, applied-window and actual
admission respectively. `fairness_apply_branch`, `fairness_apply_status`, and
`fairness_apply_wait_limit_s` give the planned policy details; a planned order
is not necessarily an actual admission change. `fairness_apply_ns` measures the
additional selection and is nested inside the existing `type_rank_ns` and full
schedule time; do not add nested measures or their percentiles. Existing
`shadow_type_rank_v1` remains the previous shadow reference, not actual order.

The benchmark currently supplies a true `r1_request_type` carried through
completions and EngineCore to Scheduler. This is a mechanism experiment, not a
final online scheme independent of true type labels. Engine waiting time starts
at the native first QUEUED event, not the client latency start. The threshold
is not a waiting-time guarantee, nor are estimated scores SLO guarantees.
CPU tests with stubs do not validate online resource behavior or performance.
