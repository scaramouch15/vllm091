# Type-aware SLO shadow ranking

`VLLM_R1_TYPE_RANK_SHADOW=1` enables a read-only experimental ranking;
the default is `0`. It requires `VLLM_R1_SHADOW=1`,
`VLLM_R1_PREFILL_SHADOW=1`, `VLLM_R1_TYPE_SLO=1`, native `log_stats`, and
`VLLM_R1_ADMISSION_MODE=off`. Configure both trusted models as described in
`PREFILL_SHADOW.md`, and initialize the per-type SLO experiment as before.
`VLLM_R1_TYPE_RANK_WAIT_LIMIT_SECONDS` defaults to 30 seconds and must be
finite and positive. This is an initial experimental setting, not an optimum.

The existing B shadow sweep scans at most its configured scan limit, skips
ineligible requests, and takes at most its configured candidate cap. It runs
only at the configured sampling interval, after running scheduling, if there
was no preemption and waiting, token budget and running capacity permit.
It is not the admission contiguous-prefix observer. A successful B prediction
and the same sweep's prefill prediction feed this observer; no new inference
or queue operation is performed. Requests previously admitted and requests not
in WAITING state are excluded by that sweep, so these predictions are not used
as remaining-time estimates after preemption. Its candidate and error records
describe skipped or failed sweeps; absence of a ranking is not an empty queue.

For chat, the score is `(chat.ttft - wait - prefill_p90) / chat.ttft`.
For code/summary it is `(type.e2el - wait - inference_p90) / type.e2el`.
B inference time already contains prefill. Length quantiles are logged, but
not used as a second penalty. Stable ascending score orders non-protected
candidates; candidates with wait >= the configured limit are placed first,
in their original relative order. This is observation only and does not give
a waiting-time guarantee. Missing type, time, SLO or prefill makes the result
partial: the comparable subset is recorded, never presented as a full order.

The saved first QUEUED event and schedule timestamp are monotonic and belong
to the same engine process. The timestamp is sampled before running scheduling
and prediction; wait excludes that work. It starts at engine queue insertion,
not client send, and these scores are not complete client TTFT/E2EL or SLO
violation probabilities. P90 coverage requires separate validation. Chat
TPOT SLO is recorded but there is no scheduler-local per-request live ITL
prediction; the output processor's native statistics use output-batch times
and are not an estimate of a waiting request's future TPOT.

`shadow_type_rank_v1` records candidate IDs, individual scores, reasons,
protection, B length quantiles, prefill SHA, state snapshot and compute_ns.
`compute_ns` covers row construction and sorting through that timestamp;
`type_rank_foreground_v1` additionally includes foreground logging and
bookkeeping. Both are nested within the existing dynamic foreground path;
background writing and model prediction are outside compute_ns. Do not sum
nested timings or their percentiles. Writer loss/closure uses existing shadow
records and counters. Full schedule timing requires the existing timing switch.
