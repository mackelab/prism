# PRISM

PRISM jointly infers model structure and continuous parameters from observations
using simulation-based inference. An observation encoder feeds an autoregressive
model-selection network and a diffusion model for parameter inference.
This repository supports diffusion MRI and symbolic regression.

[Paper](https://arxiv.org/abs/2603.15292) · [Documentation](docs/index.md) · [Run the experiments](#run-the-experiments) · [Reproduce the paper](#reproduce-the-paper)

**Responsible person:** [@shanzi-bao](https://github.com/shanzi-bao)

## Install

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
git clone https://github.com/mackelab/prism.git
cd prism
uv sync --locked --extra dev                  # CPU
uv sync --locked --extra dev --extra cuda12   # NVIDIA GPU (or cuda13)
```

Python 3.11+ is supported; `.python-version` and `uv.lock` select the pinned
environment. Without a CUDA extra, JAX runs on the CPU; pick `cuda12` or
`cuda13` to match your system's CUDA version. The Python package and command-line interface are named `prism`. Run commands
with `uv run`; each `uv sync` removes extras it is not given.

## Quick start

```bash
uv run python examples/smoke_polynomial.py
```

This small CPU example generates synthetic polynomial observations, trains for
20 steps, restores an EMA checkpoint, and samples model structures and parameters.
Outputs are saved under `results/smoke/`. More examples are in [examples/](examples/README.md).

## Examples

- Symbolic: [minimal example](examples/symbolic/minimal.ipynb), covering simulation, training, EMA checkpoints, and posterior inference.
- [dMRI notebooks](examples/README.md#dmri-notebooks): simulators, training, and pretrained inference.

## Diffusion MRI

`prism.dmri` contains the diffusion MRI application used in the paper:
simulators, encoders, and the `prism predict` and `prism eval` commands.
Pretrained models are published in
[manugloeck/dmri-pretrained](https://huggingface.co/manugloeck/dmri-pretrained).
To map a scan in FSL layout (`data.nii.gz`, `nodif_brain_mask.nii.gz`, `bvals`,
`bvecs`) with the default model:

```bash
uv run prism predict FOLDER --model msb3s_2_4_6_128 --quality balanced --non-interactive
```

See the [prediction guide](docs/guides/prediction.md) for options and outputs.
For extended dMRI-specific use cases, refer to the
[dmri package](https://github.com/mackelab/dmri).

## Run the experiments

Each experiment is a preset in `conf/experiment/train/`; configurations in
`conf/` select the simulator, network, data and optimizer. Where a command runs
is fixed as follows:

- **Training** (`prism train`) runs where the preset says. The paper-scale
  presets below set `infrastructure/launcher: slurm` and a partition, so
  `prism train` **submits a Slurm job** with those resources; nothing on the
  command line selects Slurm.
- **Evaluation** scripts run **in your current shell**. The `--slurm a100` at
  the end of each evaluation command below submits it as a Slurm job instead;
  drop it to evaluate in the current shell, for example inside a GPU allocation.

To check your installation first, train the toy polynomial example (preset
`symbolic`, a linear + quadratic model for 100 steps). It is not one of the
paper experiments below: it trains in the current shell on a CPU in about a
minute and saves checkpoints to `results/symbolic/checkpoints/`:

```bash
uv run prism train +experiment/train=symbolic
```

The paper-scale presets need an Ampere or newer NVIDIA GPU (install with
`--extra cuda12`); their partitions are those of the Tübingen cluster. Training
writes to `results/<preset>/`, and the evaluators accept that directory. To
train a paper preset in the current shell instead of submitting it, add
`hydra.mode=RUN`; the
[training guide](docs/guides/training.md#run-at-another-site-or-without-slurm)
covers other clusters, W&B and resume.

**dMRI ball-and-sticks**: streaming simulation. The other dMRI presets are
listed in the [preset catalog](docs/reference/experiments.md#dmri-training).

```bash
# Train: Slurm job on h100_long (70 training hours), set by the preset
uv run prism train +experiment/train=msb3s_2_4_6_128
# Evaluate: Slurm job on a100, because of --slurm a100
uv run python -m experiments.dmri.calibration.evaluate --checkpoint results/msb3s_2_4_6_128 --mode model --output results/msb3s_calibration_model --slurm a100
```

**Symbolic model-space scaling (Figure 4c)**: streaming simulation; also
`symbolic30` and `symbolic50`.

```bash
# Train: Slurm job on a100 (24 training hours), set by the preset
uv run prism train +experiment/train=symbolic15
# Evaluate: Slurm job on a100, because of --slurm a100
uv run python -m experiments.symbolic.model_space_scaling.evaluate results/symbolic15 --output results/symbolic15_evaluation.json --slurm a100
```

**Symbolic simulation budget (Figure 4b)**: a fixed dataset simulated once at
startup; also `symbolic_sbmi_25k` and `symbolic_sbmi_50k`. Larger datasets
train through the
[Figure 4b recipe](experiments/symbolic/simulation_efficiency/README.md#reproduce-figure-4b).

```bash
# Train: Slurm job on a100 (50,000 steps), set by the preset
uv run prism train +experiment/train=symbolic_sbmi_10k
# Evaluate: Slurm job on a100, because of --slurm a100
uv run python -m experiments.symbolic.simulation_efficiency.evaluate results/symbolic_sbmi_10k --fig4b-point 10k --which best --slurm a100 --output results/fig4b_10k_best.json
```

**Symbolic simulation budget from saved files (Figure 4b, offline)**: the same
10k point, but the dataset is simulated once into files and training reads only
those files. Training writes to `results/symbolic_sbmi_10k_offline/`. The strict
`--fig4b-point` evaluation accepts only the preset-trained run above, so a
file-trained run is evaluated without it, as a diagnostic.

```bash
# Simulate: runs in the current shell (CPU, under a minute), writes results/fig4b_10k_data/
uv run prism simulate +experiment/train=symbolic_sbmi_10k --output results/fig4b_10k_data --train-size 10000 --validation-size 1024 --simulation-batch-size 10000
# Train: Slurm job on a100, with the preset's settings saved in results/fig4b_10k_data/training.yaml
uv run prism train --config-path "$PWD/results/fig4b_10k_data" --config-name training
# Evaluate (diagnostic): Slurm job on a100, because of --slurm a100
uv run python -m experiments.symbolic.simulation_efficiency.evaluate results/symbolic_sbmi_10k_offline --which best --slurm a100 --output results/fig4b_10k_offline_best.json
```

The [training guide](docs/guides/training.md#simulate-once-then-train-from-files)
describes the saved files, their hashes and moving the data to another machine.

## Load and infer

Given an observation `x`, its `mask_prior`, and JAX random keys `mask_key` and
`theta_key`:

```python
from flax import nnx
from prism.train.utils import load_checkpoint

checkpoint, model, simulator_class = load_checkpoint("results/smoke", which="best")
nnx.update(model, checkpoint["params_ema"])
model.eval()

mask = model.sample_mask(mask_key, x=x, mask_prior=mask_prior)
theta = model.sample_theta(theta_key, x=x, model_mask=mask)
```

`which` accepts `"best"`, `"latest"`, or a saved step number.
See the [minimal symbolic notebook](examples/symbolic/minimal.ipynb) for the complete workflow.

## Reproduce the paper

| Paper result | Recipe | Details |
| --- | --- | --- |
| Figure 4b: symbolic simulation efficiency | [Recipe](experiments/symbolic/simulation_efficiency/README.md#reproduce-figure-4b) | [docs](docs/experiments/fig4b.md) |
| Figure 4c: symbolic model-space scaling | [Recipe](experiments/symbolic/model_space_scaling/README.md#reproduce-figure-4c) | [docs](docs/experiments/fig4c.md) |
| Figure 6b: dMRI condition-wise calibration | [Recipe](experiments/dmri/calibration/README.md#reproduce-figure-6b) | [docs](docs/experiments/fig6b.md) |
| Figure 6c: synthetic dMRI posterior quality | [Recipe](experiments/dmri/synthetic_evaluation/README.md#reproduce-figure-6c-synthetic) | [docs](docs/experiments/fig6c.md) |

These workflows cover selected paper experiments. Each recipe lists every
command for its figure. The [paper workflow map](docs/paper_experiments.md)
records what is packaged, what counts as reproduction, and the original
checkpoint records; [experiments/](experiments/README.md) maps the code.

## Code layout

```text
prism/             The method: model (nn), training, inference, simulator interface
prism/dmri/        Diffusion MRI application: simulators, encoders, `prism eval`, `prism predict`
prism/symbolic/    Symbolic regression application, incl. the frozen Figure 4b architecture
conf/              Training and model configurations
experiments/       Scientific evaluation and plotting workflows (dmri/, symbolic/)
examples/          Small runnable examples (dmri/, symbolic/)
tests/             Tests
docs/              Guides and API reference
```

The core `prism` packages never import `prism.dmri` or `prism.symbolic`;
`tests/test_package_boundary.py` enforces this.

To use a new simulator, prior or dataset, start with the
[symbolic guide](docs/guides/symbolic.md#adapt-the-simulator-prior-or-training-data)
and [simulator API](docs/reference/simulators.md).
For development and test commands, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Sources

Built from [mackelab/dmri](https://github.com/mackelab/dmri) and
[mackelab/sbmi_transformer_symbolic](https://github.com/mackelab/sbmi_transformer_symbolic).
Released under the [MIT license](LICENCE).
