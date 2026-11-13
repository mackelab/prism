# Figure 4b: simulation efficiency

Evaluate a checkpoint on the fixed-prior symbolic task used for Figure 4b.
For each synthetic dataset, sample a model mask, sample its parameters, and
generate a noisy prediction. Report mean RMSE minus the historical noise floor,
using the original notebook's 500-point grid, seed and grouping.

The commands are in the
[Figure 4b recipe](https://github.com/mackelab/prism/tree/main/experiments/symbolic/simulation_efficiency#reproduce-figure-4b).
This page describes the training recipes, completion evidence and evaluation
protocol behind them.

## Training recipes

All seven presets use seed 0, batch size 128, RAdam at `1e-4`, width 64,
EMA 0.999, and the same fixed empirical model prior. Each starts independently.
Dataset sizes, architectures, warmup and stopping policies, checkpoint retention
and scheduler limits are listed in the
[preset catalog](../reference/experiments.md#fixed-empirical-prior-family).

The three smaller points (10k, 25k, 50k) train with `prism train` and their
preset. Use a new `run.name` for an independent run. See the
[training guide](../guides/training.md#choose-how-the-run-executes)
for launcher settings and resume.

### Completion recorder for the larger points

The four larger points (100k, 200k, 500k, 1000k) train through
`experiments.symbolic.simulation_efficiency.train --point POINT --output DIR`.
With `--slurm a100`, the job runs the recorder, which calls the existing
`prism.train.train_script` in the same process; the preset's launcher is not
used. The `a100` preset's 700 minutes leave time for data generation,
compilation and final writes around the 8-hour training limit. Without
`--slurm`, the recorder runs in the current process, for example inside an
existing GPU allocation. Plain `prism train +experiment/train=symbolic_sbmi_200k`
also submits a job, but writes no completion record for the evaluator. Append
`tracking.enabled=true`, `tracking.wandb.entity=YOUR_ENTITY` and
`tracking.wandb.project=YOUR_PROJECT` to enable W&B. Tracking is otherwise
disabled. Use a fresh output directory.
The recorder saves `training-process.json` even on process failure and writes
`training-complete.json` only after the saved step, terminal log and best
checkpoint have been checked. Its elapsed timer includes setup and compilation;
the trainer's limit applies to the training loop.

## Evaluation

The `--fig4b-point` choices cover all seven sizes. They check the matching
training preset and require 1,000 datasets in ten groups. Step-capped runs must
reach the exact cap. Larger runs additionally require successful process and
stop evidence, read from `training-complete.json` in the run directory, or from
an explicit `--completion PATH` for previously recorded runs. Merely finding a
checkpoint is not evidence that a time-limited run completed.

The run directory can be given as `results/<run.name>`. A Slurm run keeps its
checkpoints in `results/<run.name>/<timestamp>/<job>/`; the evaluator then uses
the newest timestamp and prints the directory it selected. It refuses a sweep
with several jobs, or a newest job without a saved step. Pass the job directory
itself to evaluate an older run.

Best EMA is primary; `--which latest` uses the same protocol and is supplementary.
The JSON and adjacent NPZ retain individual errors, group means, checkpoint
steps, validation loss, source hashes and reference comparisons.
Keep the saved configurations and `train_script.log` files with the checkpoints;
the logs contain the validation curve. Terminal validation is disabled in these
presets, so the best checkpoint can precede the final completed training step.
The three-point C-short comparison margins are not extended to the four larger
points. Larger-point outputs report differences from the stored PRISM/SBMI
references for review, without an automatic reproduction verdict. PRISM
references are digitized from the repository SVG, not original raw errors.
Combined plotting of the seven points is not packaged.

A run trained from saved files (`prism simulate`, then `prism train` on the
generated `training.yaml`) is rejected by `--fig4b-point`: the check compares
all scientific settings with the preset, including the data loader, and the
file loader differs from the preset's `StaticDataset`. Omit `--fig4b-point` to
evaluate such a run with the same sample count and grouping, labelled
diagnostic. The recipe's optional
[saved-files section](https://github.com/mackelab/prism/tree/main/experiments/symbolic/simulation_efficiency#training-from-saved-files-optional)
lists the commands.

### Reduced check

For a reduced implementation check, omit `--fig4b-point` and set
`--num-samples 2 --groups 1`. This still requires a compatible checkpoint and
strict EMA restoration, but is labelled diagnostic. Full evaluation belongs on
a GPU; `--slurm a100` submits it as a Slurm job.

## Source

This evaluator follows `figures/fig2/sbmi_comp_fig.ipynb` from symbolic source
`c8e118b8bd557f2b001812f00af982b136f0d793`. Training settings are listed in the
[preset catalog](../reference/experiments.md#fixed-empirical-prior-family).
The former `python examples/eval_fig4b_point.py ...` entry point has been
removed; use the module command in the recipe with the same arguments.
The metric differs from the [Figure 4c](fig4c.md) metric; see
[Evaluate the figure metrics](../guides/symbolic.md#evaluate-the-figure-metrics).
