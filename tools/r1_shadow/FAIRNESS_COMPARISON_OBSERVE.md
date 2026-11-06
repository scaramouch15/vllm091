# Same-snapshot fairness comparison (experimental)

This opt-in diagnostic runs only with admission `observe`, `type_slo`, and the
existing feasible-protection mode enabled. It computes a second planned order on
the same candidate snapshot, without extra model calls or modifying waiting.
It is not a counterfactual performance experiment: the actual queue remains FCFS.

Set `VLLM_R1_FAIRNESS_COMPARE_OBSERVE=1` and explicitly set
`VLLM_R1_FAIRNESS_WAIT_LIMIT_SECONDS` to a finite positive number (seconds).
The threshold is an experimental parameter, not a recommended value or wait bound.
Apply mode with this option enabled is rejected at configuration time.

The alternate plan orders protected candidates (`wait_s >= threshold`) by
original position, then unprotected candidates with nonnegative predicted score
by increasing score and position, then remaining candidates by original position.
All-negative unprotected candidates retain their original order. Predicted score
is not ground truth. Requests still have to pass original admission, KV, and skip
handling. This diagnostic uses the existing request type field, which in the
current benchmark is a client-provided true type label.

In `r1_admission_v1`, `candidate_ids` is the original prefix, `sorted_ids` is
the current policy's planned order, `fairness_compare_order_ids` is the alternate
plan, and `actual_admission_order` is the observed admission. The fields
`fairness_compare_status`, `fairness_compare_branch`,
`fairness_compare_wait_limit_s`, `fairness_compare_order_changed`,
`fairness_compare_vs_plan_changed`, and `fairness_compare_ns` explain the
comparison. Per-candidate wait and score are in `type_rank_candidates`. An error
is recorded in `fairness_compare_error` and does not change the current plan or
actual admission. Incomplete candidate data leaves comparison not evaluated.
The extra diagnostic has timing cost and should be reported separately.
