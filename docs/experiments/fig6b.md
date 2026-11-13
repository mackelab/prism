# Figure 6b: condition-wise dMRI calibration

Run the Figure 6b SBC condition allocation and aggregation with an explicit B3S
checkpoint and its matching saved simulator. The scientific statistics, random
key order and condition list come from the frozen coverage scripts.

The commands are in the
[Figure 6b recipe](https://github.com/mackelab/prism/tree/main/experiments/dmri/calibration#reproduce-figure-6b).
This page describes the checkpoint requirements, conditions, statistics,
resume rules and plotting options behind them.

## Checkpoint

The checkpoint directory must contain `checkpoints/` and a saved training config
readable by `prism.train.utils.load_cfg`. Both 11-parameter B3S and 12-parameter
multishell B3S are supported. The loader restores finite EMA weights and records
the selected step, latest saved step, best step, validation loss and weight hash.
`--which latest` is the default; use `--which best` for a separate evaluation.
`--expected-step` checks the selected checkpoint, not the completed training budget.

A training run can be given as `results/<run.name>`. A Slurm run keeps its
checkpoints in `results/<run.name>/<timestamp>/<job>/`; the evaluator then uses
the newest timestamp and prints the directory it selected. It refuses a sweep
with several jobs, or a newest job without a saved step. The protocol records
the selected directory, so `--resume` works with either path form as long as it
selects the same directory. Pass the job directory itself to evaluate an older
run.

The public `msb3s_2_4_6_128` checkpoint can exercise this workflow, but it has not
been identified as the historical Figure 6 checkpoint. The separate
[synthetic evaluation](fig6c.md) workflow provides overall-sample SBC and
synthetic rKSD; real-voxel evaluation is not included here.

## Run

A small CPU execution check, from the repository root:

```bash
JAX_PLATFORMS=cpu uv run python -m experiments.dmri.calibration.evaluate \
  --checkpoint /path/to/checkpoint --which best \
  --mode model --output results/calibration_model_smoke \
  --trials 2 --batch-size 2 --chunk-size 2 --mask-samples 4

JAX_PLATFORMS=cpu uv run python -m experiments.dmri.calibration.evaluate \
  --checkpoint /path/to/checkpoint --which best \
  --mode theta --output results/calibration_theta_smoke \
  --trials 2 --batch-size 2 --chunk-size 2 --theta-samples 4 --ode-steps 4
```

For the full allocation, omit the reduced counts and add `--slurm a100` to submit
each mode as a Slurm job; logs go to `results/slurm/`. The job checks that JAX
uses the GPU. These are evaluation commands; they do not train a model. Tracking is off by default.
Enable W&B with an explicit `--wandb --wandb-entity ENTITY --wandb-project PROJECT
--run-id ID`; only the protocol, summaries and plots are uploaded.

## Conditions and statistics

| Mode | Conditions | Default datasets per condition | Total datasets |
| --- | --- | --- | --- |
| `model` | Short acquisition: overall plus 10 fixed prior values | 100,000 | 1,100,000 |
| `theta` | 15 fixed model masks for each of short and long acquisition | 10,000 | 300,000 |

The fixed prior values are `0, .1, .2, .3, .4, .5, .6, .8, .9, 1`, following the
source launcher. Defaults use seed 0, 200 model draws or 50 parameter draws, and
64 ODE steps with `t_min=1e-4` and no final Euler step. Trial keys preserve the
source launcher's sequential splits and `random.split(key, N)` order. Changing
`N` changes those keys; a reduced check is not a subset of the full experiment.

Model SBC uses randomized ranks of posterior model log probabilities. Parameter
SBC uses randomized ranks of the **sum of active latent parameters**. Coordinate
ranks are also saved as a supplementary diagnostic. They are not the statistic
used for the paper-style aggregate parameter curve.

Model curves are averaged equally over the 11 conditions. Parameter curves are
averaged over both acquisition schemes and then over the 15 fixed masks. The
right-hand panel reports each selected mask's calibration error separately for
the two schemes. The evaluator does not generate the gray expected-error band;
the original plotting workflow below provides its separate calculation.

## Results and resuming

Each condition has immutable trial chunks, raw arrays, a summary and plots.
`results.json` collects all conditions and the aggregate curve; `figure6b-model`
or `figure6b-theta` PNG/PDF files show the aggregate panels. `protocol.json`
records the checkpoint, source hashes, counts, sampling settings and condition
order. Environment and invocation details are recorded for each attempt.

Repeat the same command with `--resume` to verify completed chunks and continue
from the first missing trial. Changed settings, code, checkpoint contents or
damaged chunks are rejected (`--batch-size` only sets the computational chunk
and may differ). Because the source hashes cover every Python file
under `prism/` and `experiments/dmri/`, any code change between attempts
rejects a resume. Use a separate output directory for each mode and
checkpoint selection. Historical cluster outputs are reference results and are
not automatically imported into this resume format.

## Plot Figure 6b panels

`plot.py` reads stored calibration results and uses
`experiments/dmri/_shared/paper.mplstyle` for the panel style. Use evaluations
of the same checkpoint when combining panels. Either mode can also be plotted
alone. NPZ inputs are supported via `--model mask.npz`
and `--theta-short short.npz --theta-long long.npz`.

Plots use equal averaging, a fixed mask subset and step curves. Condition
summaries are reordered by mask bits into the panel order. The default labels
follow `cov_fig.ipynb`, which labels the short curve `HCP-like` and the
long curve `UKB-like`; those are the CLI defaults. The explicit override
`--curve-labels UKB-like HCP-like` in the recipe labels the matching short
UKB-like and long HCP-like generators used here.

For a gray reference band, run the Monte Carlo calculation
separately (`--make-band`), then pass its `reference_band.json` with
`--theta-band`. The recipe uses the paper's 10,000 datasets per parameter
condition. This defaults to 100,000 Monte Carlo repetitions with seed 0 and can
be expensive; add `--slurm a100` to submit it as a Slurm job. `--band-trials`
and `--band-batch-size` allow a small execution check. The saved counts must
match the parameter evaluation. Without a supplied band, it is omitted. Outputs
are SVG/PNG/PDF panels and `plot.json` with input and source hashes.

## Checkpoint compatibility

Shared loaders and acquisition/model adapters live in `experiments/dmri/_shared/`.
Existing saved `experiments.dmri.calibration.model` and `.acquisition` targets
continue to resolve to the same classes and functions at their new locations.

Saved imports from the earlier `experiments.dmri_paper.runtime` and
`experiments.dmri_paper.acquisition` modules are mapped in memory to
`_shared/model.py` and `_shared/acquisition.py`. Their numerical behavior
is retained; checkpoint
files and core model code are not rewritten. The acquisition adapter's 295-point
HCP template is reconstructed from the frozen 297-point template; the exact
295-point template used in the paper is unavailable. Assumptions and hashes are recorded when
that adapter is used. Other checkpoints retain their own saved generators.
