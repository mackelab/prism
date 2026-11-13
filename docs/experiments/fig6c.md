# Figure 6c: synthetic dMRI posterior quality

Evaluate the public `msb3s_2_4_6_128` model using the two acquisition generators
and model prior saved with its checkpoint. The workflow restores EMA weights,
generates synthetic observations, computes model/parameter SBC and relative
KSD, and writes raw arrays, summaries and plots. It does not train a model.

The commands are in the
[Figure 6c recipe](https://github.com/mackelab/prism/tree/main/experiments/dmri/synthetic_evaluation#reproduce-figure-6c-synthetic).
This page describes the input, metrics, outputs and plotting options behind them.

## Input and execution

Use a local bundle from `manugloeck/dmri-pretrained` at revision
`017561ffbf77b109c9fe6c3d9c7d48831780ddc3`. The bundle root must contain
`config.yaml` and `checkpoints/`. The runner checks the public best step
1,520,000 or latest step 2,280,000 and requires exact, finite EMA restoration,
so a newly trained run is evaluated with the [Figure 6b](fig6b.md) workflow
instead. The runner does not download checkpoints; the recipe stages the
specified revision before running.

From the repository root, a reduced CPU check is:

```bash
JAX_PLATFORMS=cpu uv run python -m experiments.dmri.synthetic_evaluation.run \
  --checkpoint /path/to/msb3s_2_4_6_128 --which best \
  --output results/dmri_synthetic_smoke \
  --num-samples 2 --batch-size 1 \
  --mask-samples 4 --theta-samples 4 --sbc-steps 4 --ksd-steps 4
```

Use a new output directory for each run. The default evaluation uses 1,000
datasets per acquisition, seed 0, 200 mask draws and 50 parameter draws.
Run that workload on a GPU; add `--slurm a100` to submit it as a Slurm job,
which checks that JAX uses the GPU. Select
`--sequence short|long|both` and `--mode sbc|ksd|both` to limit the task.
The saved short and long acquisition generators use 105 and 297 measurements.

Tracking is off by default. To enable it, supply `--wandb`, `--wandb-entity`,
`--wandb-project` and a new `--run-id` explicitly. Logging uploads metrics and
plots, not checkpoints.

## Metrics

- **Model SBC:** randomized ranks of posterior model log probabilities.
- **Parameter SBC:** both the frozen evaluator's sum of active latent
  parameters and ranks of individual active coordinates.
- **Synthetic rKSD:** rooted posterior KSD divided by the mean rooted prior
  KSD for the same exact model mask and acquisition group. Masks with the same
  number of sticks are kept separate.

SBC defaults to 64 ODE steps; KSD uses 128 steps and minimum diffusion time
0.005. Both disable the final Euler step, as in the frozen evaluation scripts.
The indexed RNG schedule keeps observation and sampling keys stable across
batch sizes and metric selection.

`--ksd-variant frozen` preserves the original kernel. `--ksd-variant corrected`
selects the separately tested score-transform correction. Always keep these
outputs separate. Numerical summaries use untrimmed rKSD values; violin plots
omit each mask's upper one percent, following the source plotting convention.

## Outputs and paper relationship

`protocol.json` records the checkpoint, current code revision, source hashes,
sampling settings and environment. Each acquisition/condition directory contains
batch arrays, `samples.npz`, `summary.json` and plots. `results.json` collects
the completed summaries. Inactive parameter PIT entries are NaN by definition;
nonfinite active values fail evaluation after saving the offending batch.

This supports the synthetic diagnostics related to Figure 6b and the synthetic
part of Figure 6c. The public checkpoint has not been identified as the paper's
original checkpoint. Per-mask SBC here groups the overall sample by mask;
the paper's condition-wise calibration experiment uses a separate allocation,
implemented in [calibration](fig6b.md).
Real-voxel experiments and their data are not included in this workflow.

## Plot Figure 6c

`plot.py` draws per-mask posterior-quality violins from saved arrays, using
`experiments/dmri/_shared/paper.mplstyle`.
NPZ files with `mask_support`, `ksd` and `ksd_prior` can be used directly.
If available, add `--real-selected selected.npz --real-fixed fixed.npz` from the
same acquisition and checkpoint to show the other two groups. Supporting these
stored inputs does not provide the real-data evaluation workflow.

Values are normalized by the prior within each mask group and the upper one
percent is trimmed. Matplotlib violins are calculated in linear space, then the
axis is set to logarithmic scale. Mask selection drops rows; it does not pool
masks with the same label. Missing real-data groups are omitted.

Set `--measurements` to the input acquisition count. `--kernel corrected` labels
score-corrected outputs separately; the original kernel is the default. Current
per-trial inputs are checked against their saved summary and protocol. For legacy
inputs without a kernel field, the caller's declaration is recorded explicitly.
Outputs are SVG/PNG/PDF, the plotted arrays in `plot_data.npz`, and input hashes
and checkpoint metadata in `plot.json`.
