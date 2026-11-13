# Installation and Quickstart

## Requirements

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Git and a clone of this repository
- CPU support for small examples; a supported NVIDIA GPU is recommended for
  larger training and prediction workloads

## Install the runtime

Create an environment and install from the repository. The Python package and
command are named `prism`:

```bash
git clone https://github.com/mackelab/prism.git
cd prism
uv sync --locked
```

`uv.lock` fixes dependency versions. `.python-version` selects the validated
Python 3.12.14 environment; the package still supports Python 3.11 or newer.
Run commands with `uv run`.

The default installation uses CPU JAX. On a supported NVIDIA Linux machine,
select the extra that matches your driver and GPU:

```bash
uv sync --locked --extra cuda12
# Or, for a CUDA 13 compatible system:
uv sync --locked --extra cuda13
```

Use one CUDA extra. The extras install JAX's bundled CUDA runtime. Check the
[JAX installation requirements](https://docs.jax.dev/en/latest/installation.html)
for supported hardware and driver versions. The package currently requires
JAX `0.8.1` and Flax `0.12.0`; retain these constraints when configuring
the environment.

Inspect the devices JAX can use:

```bash
uv run python -c 'import jax; print(jax.devices())'
```

Development tools are not part of the default installation. Add `--extra dev`
when developing, or `--extra docs` to build the documentation. Each sync should
include all extras you want to retain, for example
`uv sync --locked --extra dev --extra cuda12`.

## Train a small symbolic model

Run a 100-step polynomial example on a small fixed simulated dataset:

```bash
uv run prism train +experiment/train=symbolic run.name=symbolic_example
```

Checkpoints are saved below `results/symbolic_example/`. Use a new `run.name`
for an independent run. The [training guide](guides/training.md) covers data,
checkpoint loading, and resume; the [quickstart notebooks](examples/index.md)
walk through posterior sampling and prediction in Python.

## Predict from diffusion MRI data

Prepare a folder containing `data.nii.gz`, `nodif_brain_mask.nii.gz`, `bvals`,
and `bvecs`, as described in the [prediction guide](guides/prediction.md#input).

> **Check the acquisition.** A pretrained checkpoint is valid only for
> compatible acquisition schemes. The four required filenames do not establish
> compatibility. PRISM is research software and has not been clinically
> validated.

Specify the model and quality preset so the command does not prompt:

```bash
uv run prism predict FOLDER \
  --model msb3s_2_4_6_128 \
  --quality balanced \
  --non-interactive
```

The checkpoint is downloaded from Hugging Face and cached locally. Open
`FOLDER/dmri_output/view_results.html` to inspect the exported maps. CPU
prediction can be slow for thousands of voxels; use a supported GPU for larger
datasets. See the [prediction guide](guides/prediction.md) for local checkpoints
and sampling settings, the [CLI reference](cli/predict.md) for options, and
the [outputs reference](reference/outputs.md#prediction-output) for file details.

## Continue

- [Train a model](guides/training.md) with `prism train`.
- [Evaluate a trained model](guides/evaluation.md) with `prism eval`.
- [Run the examples](examples/index.md) for Python workflows.
- [Run the experiments](https://github.com/mackelab/prism#run-the-experiments)
  with one command per task, or [reproduce a paper figure](paper_experiments.md).
