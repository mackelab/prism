# Figure 4c: model-space scaling

Evaluate the conditional parameter-prediction metric used for Figure 4c.
The generating model mask is supplied to parameter inference. The evaluator
reports mean predictive RMSE minus a separately simulated ground-truth noise
baseline; this does not evaluate model-selection error.

The commands are in the
[Figure 4c recipe](https://github.com/mackelab/prism/tree/main/experiments/symbolic/model_space_scaling#reproduce-figure-4c).
This page describes the training settings and evaluation protocol behind them.

## Training

Run each of `symbolic15`, `symbolic30` and `symbolic50` as a separate
experiment. These presets use online simulation, training batches of 4,096, a
2,097,152-example buffer, and a 24-hour training limit with no step cap. They
select the `a100` partition and submit a Slurm job, requesting 26 hours so the
24-hour training limit is reached, and they start fresh with tracking disabled.
Use a new `run.name` for an independent run. See the
[training guide](../guides/training.md#choose-how-the-run-executes)
for Slurm, W&B, and resume, and the
[preset catalog](../reference/experiments.md#variable-prior-scaling-family)
for full settings.

## Evaluation

K=15/30/50 checkpoints are supported. Defaults retain latest EMA, 64 diffusion
steps, the 500-point grid, and 1,048,576 evaluation datasets. The source's
five-checkpoint random-key split is preserved when evaluating a single model.
JSON and NPZ outputs include predictive and baseline errors, bootstrap results,
checkpoint information, configuration differences and source hashes.
Both predictive and ground-truth signals include fresh noise. The bootstrap
interval holds the separately measured ground-truth mean RMSE fixed.
`--batch-size` changes computational chunk size; `--num-samples` changes the
evaluation sample count. Choose a new output path for each evaluation.

The run directory can be given as `results/<run.name>`. A Slurm run keeps its
checkpoints in `results/<run.name>/<timestamp>/<job>/`; the evaluator then uses
the newest timestamp and prints the directory it selected. It refuses a sweep
with several jobs, or a newest job without a saved step. Pass the job directory
itself to evaluate an older run.

`--slurm a100` submits the evaluation as a Slurm job with the resources of
`conf/infrastructure/partition/a100.yaml` and prints the job ID; logs go to
`results/slurm/`. Run the module with `python -m` for this to work.
The complete model-space/width matrix and control modes of the figure are not
packaged.

### Reduced check

For a reduced CPU check, omit `--slurm` and use
`--num-samples 2 --batch-size 1`. A compatible trained checkpoint is still
required. Reduced runs check execution, not the full paper result.

## Source

The implementation follows `figures/fig2/eval_pred_error.py` and the aggregation
in `figures/fig2/scaling.ipynb` from symbolic source
`c8e118b8bd557f2b001812f00af982b136f0d793`. See the
[symbolic guide](../guides/symbolic.md). The former
`python examples/eval_symbolic_scaling.py ...` entry point has been removed;
use the module command in the recipe with the same arguments.
