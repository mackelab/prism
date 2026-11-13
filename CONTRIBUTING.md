# Contributing

Responsible maintainer: [@shanzi-bao](https://github.com/shanzi-bao).

## Working changes

- Commit useful work regularly.
- Keep changes focused so reviewers can distinguish restructuring from behavior changes.
- Verify new numerical code on synthetic data with known ground truth, or a
  previously analyzed reference input, before applying it to unfamiliar data.
- Preserve upstream licenses, attribution, and source revision records on import.
- Keep credentials, private datasets, checkpoints, and generated outputs out of Git.

## Verification

Run the checks relevant to the changed behavior and
record their results in the pull request. For numerical changes, include the
reference input, random seed, comparison metric, and tolerance where relevant.
Profile expensive workflows before attempting performance refactors.

## Reported analyses

Commit code and configuration before an analysis used in a paper, presentation,
or other external result. Identify the analysis in the commit message and retain
the exact commit with the analysis metadata. Use immutable tags for release or
figure states that must remain easy to retrieve.

Record both the completed training step and the evaluated checkpoint step, as
well as whether inference used EMA or raw parameters. A moving selector such as
`latest` alone is not sufficient provenance.
