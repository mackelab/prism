# Experiments

Scripts that reproduce the packaged paper experiments. Run experiment commands
from the repository root after installing the project. The shared models,
simulators and training code live in `prism/`; experiment scripts specify
sampling, metrics and result collection. Store generated data, checkpoints and
results under `results/` or outside the checkout.

| Folder | Paper result | Recipe | Details |
| --- | --- | --- | --- |
| `symbolic/simulation_efficiency/` | Figure 4b | [Reproduce Figure 4b](symbolic/simulation_efficiency/README.md#reproduce-figure-4b) | [docs](../docs/experiments/fig4b.md) |
| `symbolic/illustrations/` | Figures 2, 4a, A.1 | [Illustrations](symbolic/illustrations/README.md#run-it) | posterior draws and panels for one observation |
| `symbolic/model_space_scaling/` | Figure 4c | [Reproduce Figure 4c](symbolic/model_space_scaling/README.md#reproduce-figure-4c) | [docs](../docs/experiments/fig4c.md) |
| `dmri/diffusion_steps/` | Figure A.4 | [Diffusion steps](dmri/diffusion_steps/README.md) | ESS and run time sweep (untested) |
| `symbolic/scaling_appendix/` | Figure A.2 | [Scaling metrics](symbolic/scaling_appendix/README.md) | plot and aggregation from stored archives |
| `dmri/calibration/` | Figure 6b | [Reproduce Figure 6b](dmri/calibration/README.md#reproduce-figure-6b) | [docs](../docs/experiments/fig6b.md) |
| `dmri/synthetic_evaluation/` | Figure 6c (synthetic) | [Reproduce Figure 6c](dmri/synthetic_evaluation/README.md#reproduce-figure-6c-synthetic) | [docs](../docs/experiments/fig6c.md) |
| `symbolic/classification/` | Table 1, Figure A.3 | [Classification metrics](symbolic/classification/README.md) | model-selection confusion matrix and metrics |
| `dmri/sbi_baseline/` | SBI joint baseline (Tables 2–4) | [B3S SBI baseline](dmri/sbi_baseline/README.md) | needs optional `torch` and `sbi` |
| `profiling/` | Run-time comparison | [Run-time profiling](profiling/README.md) | per-operation time and memory of a saved run |

Each recipe lists every command for its figure. The
[paper workflow map](../docs/paper_experiments.md) records which figures are
packaged, what "reproduce" means for new checkpoints, and the original
checkpoint and result records. No W&B account or Slurm installation is needed
for the reduced local checks described in the documentation pages.

Every script prints its options with `--help`, for example
`uv run python -m experiments.dmri.calibration.plot --help`.

## Shared dMRI code

`dmri/_shared/` contains internal checkpoint loading, acquisition adapters,
statistics, file utilities and plotting support. `paper.mplstyle` defines the
figure style. The shared checkpoint module has separate loaders for the pinned
public reference and saved training runs, both requiring EMA weights.

The calibration-specific condition schedule and resumable chunk format remain
in `dmri/calibration/protocol.py` and `dmri/calibration/chunks.py`. General
models, simulators, training and inference are implemented in `prism/`.
