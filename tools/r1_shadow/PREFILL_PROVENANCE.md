# Candidate prefill provenance and limits

This is a summary of supplied evidence, not validation of the model bytes. The
operator must retain the original training report and identify it with the
configured provenance ID. Never include model binaries, request text, or
private absolute paths in this repository.

The supplied `training_report.json` has SHA256
`d5c3603a952f46845bd9b89b50dae94041cd5eb256abc5e36ebe9dcfa78f000b`.
It identifies candidate model SHA256
`94868fdd98d9f3466247d3962d9459be0e9236042491755dedfe9cb2ccdd16f1`,
sklearn 1.7.2, and the target `first_scheduled_to_first_token_batch_seconds`.
Its source is a merged collection from the exploratory `r1_pf_graph_on`
run; a source-data content hash and the full Python/dependency versions are
not provided in that report and remain to be verified. It reports 1,444
training requests and 365 validation requests, split by decision batch using
GroupShuffleSplit with seed 42. Same-run offline P50/P90/P95 empirical
coverage is 48.49% / 89.04% / 97.26%; these are exploratory results, not an
independent test.

The user separately reported a later shadow run with 365 matched holdout
inputs and coverage 44.38% / 89.04% / 98.36%. ChatGPT calculated those
figures from uploaded merged, shadow and validation identity files. Those
source files were not supplied for this review, so this result is reported
as supplied and has not been independently recalculated. It is a different
run from the report above and is not a final independent test. Hashes alone
do not prove training provenance or generalization.

The prior candidate SHA is
`4ab94aa14148645ab06dab7d7adfc49a882836b5305e96657d37b075863c021c`.
The present source does not establish its full training data provenance. The
original B model remains independently checked against its existing SHA and
is not retrained or changed here. Confirm exact dataset identities, inputs,
versions and artifacts on the server before making any comparative claims.
