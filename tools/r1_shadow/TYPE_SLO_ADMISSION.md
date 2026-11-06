# Experimental type-SLO waiting admission

Default admission mode remains `off`. To opt in use `VLLM_R1_ADMISSION_MODE=apply`,
`VLLM_R1_ADMISSION_HEAD_POLICY=type_slo`, `VLLM_R1_ADMISSION_TIME_OBSERVE=1`,
`VLLM_R1_TYPE_RANK_SHADOW=1`, `VLLM_R1_TYPE_SLO=1`, `VLLM_R1_SHADOW=1`, and
`VLLM_R1_PREFILL_SHADOW=1`, with the trusted B/prefill model configuration in
`PREFILL_SHADOW.md`. Native `log_stats` is required. Skip-single-slot is
incompatible, since ordering can matter when only one slot is free. Admission
`observe` computes the same order but does not apply it. This configuration
uses admission's bounded contiguous eligible prefix, not the old shadow
sweep's skip-ineligible candidates. Existing admission every/prefix settings
remain independently configurable.

On an update opportunity after running scheduling, B and prefill each predict
once using the same snapshot for the prefix. Chat score is
`(TTFT_SLO - wait - prefill_P90) / TTFT_SLO`; code and summary use their
respective E2EL SLO and B inference P90 instead. B includes prefill; it is not
added again. B length quantiles are recorded. All candidates at or above
`VLLM_R1_TYPE_RANK_WAIT_LIMIT_SECONDS` (default 30 experimental seconds) lead
in original order; the rest are stable ascending by normalized score. Negative
scores remain negative and sort before larger unprotected scores. Neither
P90, this threshold, nor this score is claimed optimal or a TPOT guarantee.

The queue is changed only after identity, prefix and all scores validate.
Missing type/SLO/first QUEUED time or any invalid/failed prediction causes the
whole window to use native order. Requests previously admitted or preempted
and other special requests truncate the window. The original native loop
performs all capacity, token and KV checks and break/skip behavior. After the
loop, remaining window members are restored in original relative order before
native skipped backfill. A window is predicted only once per schedule; multiple
vacancies reuse the order and snapshot without recalculating after admission.
Window exhaustion allows normal processing of the tail. Resource exceptions
are not turned into queue-only rollback.

The prefill and type-rank records identify their source as
`admission_contiguous_prefix`. Existing offline tools designed for B shadow
`shadow_candidates` sweeps assume matching sweep counters and must not be used
unchanged to claim completeness for this admission path.

The `r1_admission_v1` record retains candidate IDs, sorted IDs, order change,
application, actual admission order and timings; `type_rank_candidates`
contains score components, output length quantiles, protection and missing
reason. `type_prefill_ns` includes prefill observation/log construction;
`type_rank_ns` includes scoring/sorting and its foreground logging. These are
separate from B `prediction_ns` and nested in full schedule time. The
`actual_window_order_changed` field compares relative order among actually
admitted window members; a single later candidate admitted while the original
head remains is instead identifiable from `candidate_ids` and
`actual_admission_order`. Planned and actual outcomes are distinct.

Waiting starts at the engine's first native QUEUED event; decision timestamp
is sampled before running scheduling. Neither is client send time. TPOT is not
predicted; evaluate client TTFT, TPOT and joint Goodput separately. The finite
prefix, sampling and resource constraints mean waiting protection does not
establish a global bound or no-starvation guarantee. CPU stub tests are not
real model/GPU/service validation.
# Experimental feasible-budget protection

Opt in only with the existing `type_slo` admission setup by setting
`VLLM_R1_ADMISSION_FEASIBLE_PROTECTION=1` and optionally
`VLLM_R1_ADMISSION_PROTECTION_SLO_MULTIPLIER=1` (finite and positive; default 1).
The existing shadow `TYPE_RANK_WAIT_LIMIT` supplies the protection cap. For
each candidate the threshold is `min(cap, multiplier * slo_seconds)`, using
chat TTFT and code/summary E2E targets. Protected candidates retain queue order;
other nonnegative-score candidates follow in increasing score, then negative-score
candidates in original order. When all scores are negative and none is protected,
the original order is retained. Missing inputs preserve the existing fail-closed
window fallback. This affects only the normal eligible prefix; resources and native
preemption/skipping logic remain unchanged. Observe mode records without applying.

`r1_admission_v1` records `feasible_rule_branch`, multiplier, and each candidate's
`feasible_protection_limit_s` and `feasible_protected`. The separate
`shadow_type_rank_v1` record continues to describe the original shadow ranking,
not this experimental policy; use the admission record for actual planned order.
Protection is not a hard wait-time or SLO guarantee. Chat's TPOT is not predicted.
Client E2E SLO results include waiting; predictor target is admission to finish.
Both threshold and multiplier are experimental and must be evaluated in live runs.

In this experiment, the request type is the real `r1_request_type` supplied by
the mixed client and carried through the completions API and EngineCore request
to the Scheduler. The predictor does not take this label as a feature. This is
not a type-label-free online deployment or a final type-inference solution.
A negative predicted margin is an estimate, not proof of an actual SLO miss.
