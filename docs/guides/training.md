# Training

`prism train` composes `conf/train.yaml`, generates training and validation data
from the configured simulator, builds the model, and runs the JAX training
loop. Use Hydra overrides to change a run. Retain the saved Hydra config,
source revision, environment, and checkpoints when recording an experiment.
The [experiment catalog](../reference/experiments.md) describes the named
presets and their scientific settings. For polynomial and figure workflows,
see the [symbolic regression guide](symbolic.md).

## Small symbolic example

```bash
uv run prism train +experiment/train=symbolic run.name=symbolic_example
```

The preset selects `SimplePolynomial`, a transformer observation embedding,
and `CompartmentTokenizer` for additive components. It simulates 128 training
examples and 32 validation examples on a grid of 32 points in `[-1, 1]`.
Training uses fixed data, batch size 8, EMA weights, a 100-step cap, and
validation/checkpoint intervals of 25 steps.

Use a distinct `run.name` for each independent experiment. Checkpoints for this
example live in `results/symbolic_example/checkpoints/`; timestamped directories
hold each launch's configuration and log.

## Choose how the run executes

Every training preset that runs longer than an hour submits a Slurm job: the
dMRI presets select the `h100_long` partition (`2h100_long` for the two-GPU
presets) and the symbolic presets select `a100`. Only the small `symbolic`
preset runs locally. The base config without a preset also runs locally, with
no cluster partition. Tracking is disabled and `training.continue_training=false`
by default; select both explicitly.

### One local process

```bash
uv run prism train +experiment/train=b3s_2_4_6_128 hydra.mode=RUN \
  run.name=local_test training.max_train_hours=1
```

With `hydra.mode=RUN`, training runs directly in the calling process and the
launcher config is not invoked, even for a preset that normally submits a job.
This is the simplest mode for development, for one-machine training, and for
training inside an existing GPU allocation.

Named presets store their scientific settings in ordinary configuration
fields, so they apply equally to single runs and multiruns. For example,
`b3s_2_4_6_128` and `msb3s_2_4_6_128` both select width 128 and 2 observation,
4 model-selection, and 6 parameter-inference layers. Inspect the composed
configuration with `--cfg job --resolve`.

### Local Hydra multirun

```bash
uv run prism train --multirun +experiment/train=b3s_2_4_6_128 \
  infrastructure/launcher=local infrastructure/partition=none run.seed=0,1,2
```

Hydra expands the requested seed values and Submitit starts each job locally.
Reset the partition together with the launcher: partition presets set
Slurm-only fields.
Comma-separated overrides add sweep dimensions; `--multirun` by itself runs
one job with the preset's settings. Each job receives a numbered output
directory containing its own configuration, logs, and checkpoints.

### SLURM

```bash
uv run prism train +experiment/train=b3s_2_4_6_128
uv run prism train +experiment/train=symbolic infrastructure/launcher=slurm \
  infrastructure/partition=a100
```

The first command submits with the preset's own launcher and partition. The
second selects them for a configuration that runs locally by default. The
`slurm` launcher sets `hydra.mode=MULTIRUN` and maps `hydra.launcher` values to
a submitted job; partition presets (`a100`, `h100_long`, `2h100_long`) only set
queue, time, CPU, memory, and GPU fields. They name partitions of the Tübingen
cluster; inspect and select resources available at your site, for example with
`infrastructure/partition=h100_long` or `hydra.launcher.partition=my-queue`.
`hydra.launcher.timeout_min` is the scheduler limit, while
`training.max_train_hours` and `training.max_train_steps` are enforced inside
training. They are independent: allow enough scheduler and training time to
reach a requested step count.

### Run at another site or without Slurm

The partition presets name partitions of the Tübingen cluster. For another
Slurm cluster, add a preset such as `conf/infrastructure/partition/mysite.yaml`:

```yaml
# @package _global_

hydra:
  launcher:
    timeout_min: 1560
    gres: gpu
    partition: my-gpu-queue
    cpus_per_task: 8
    mem_gb: 32
```

Then select it with `infrastructure/partition=mysite` for `prism train`, or with
`--slurm mysite` for the experiment scripts; both read the same file. Other
`hydra.launcher.*` overrides, such as `hydra.launcher.account=YOUR_ACCOUNT`,
apply to `prism train` only. Scripts also accept `--slurm-timeout-min`.

Without Slurm, add `hydra.mode=RUN` to `prism train` and omit `--slurm` from the
experiment scripts: both then run in the current process, for example on a
workstation or inside an interactive GPU allocation. The paper presets need an
Ampere or newer NVIDIA GPU; CPU runs are only practical for the small
`symbolic` preset.

A Slurm job started by `prism train` logs to
`results/<run.name>/<timestamp>/.submitit/<job id>/`; a job started with
`--slurm` logs to `results/slurm/<job id>/`. Check queued and running jobs with
`squeue --me`.

### Short test run of a preset

Before submitting a multi-hour job, check that a paper preset starts, trains,
validates and checkpoints in your environment. Run it in the current process on
one GPU with a separate `run.name`, so the check never mixes with the real
results in `results/<preset>/`:

```bash
uv run prism train +experiment/train=symbolic_sbmi_10k hydra.mode=RUN \
  run.name=symbolic_sbmi_10k_check \
  training.max_train_steps=10 training.inner_steps=5 \
  training.checkpoint_freq=5 training.eval_freq=5 \
  training.eval.nnl_mask.iters=1 training.eval.nnl_theta.iters=1 \
  training.dataloader.dataset.dataset_size=256 \
  training.dataloader.dataset.simulation_batch_size=256 \
  training.dataloader.val_dataset.dataset_size=128
```

Lower `training.inner_steps` together with `training.max_train_steps`: the loop
advances in blocks of `inner_steps` (500 in the paper presets). Fixed-dataset
presets simulate their whole dataset at startup, so shrink `dataset_size`.
Streaming presets (`symbolic15|30|50`, dMRI) fill their buffer at startup;
replace the dataset overrides with
`training.dataloader.dataset.buffer_size=4096
training.dataloader.dataset.simulation_batch_size=512
training.dataloader.train_loader.batch_size=64`, and for dMRI also add
`training.eval.ksd.iters=0`. The paper presets use flash attention and need an
Ampere or newer GPU; a CPU run of them compiles for a long time. The evaluators'
reduced checks, described on each [paper workflow](../paper_experiments.md)
page, then run against `results/<preset>_check`.

### Weights & Biases tracking

To record a run in W&B, authenticate in the execution environment and set the
destination explicitly:

```bash
uv run prism train +experiment/train=b3s_2_4_6_128 run.name=tracked_run \
  tracking.enabled=true \
  tracking.wandb.entity=my-entity tracking.wandb.project=my-project
```

Replace `my-entity` and `my-project` with your W&B workspace and project.
The same overrides apply to local and Slurm runs. Tracking remains disabled
when `tracking.enabled=false`.

## Compose a training config

The base config selects these groups:

```text
conf/
|-- train.yaml
|-- experiment/train/
|-- simulator/
|   `-- acquisition/
|-- model/
|   |-- embedding_net/
|   |-- model_selection_net/
|   `-- inference_net/
|-- training/
|   |-- dataloader/
|   `-- optimizer/
`-- infrastructure/
    |-- launcher/
    `-- partition/
```

The base config selects no application: `simulator` and `model` have no
default. `experiment/train` presets select groups and set model, simulator, and
training values. The groups can also be selected directly; dMRI simulators
default to the `multi` acquisition:

```bash
uv run prism train simulator=ball3stick_shared simulator/acquisition=multi \
  model=dmri training/dataloader=datastream \
  training/optimizer=adamw infrastructure/launcher=local \
  infrastructure/partition=none
```

Common scalar overrides are:

```bash
uv run prism train +experiment/train=symbolic run.seed=7 run.name=my_run \
  tracking.enabled=false \
  training.dataloader.train_loader.batch_size=64 \
  training.optimizer.learning_rate=3e-4 \
  training.max_train_hours=12
```

The `symbolic` preset has a 128-sample static dataset; a larger batch is
rejected unless `training.dataloader.dataset.dataset_size` is raised too.

The current simulator schema separates the structural signal model from data
generation:

```yaml
simulator:
  model_class: my_package.models.MyModel
  posterior_score: false
  mask_prior: {}
  acquisitions:
    - _target_: my_package.acquisition.random_acquisition
      _partial_: true
      num_acquisitions: 128
```

For diffusion MRI, `model_class` resolves to a `MultiCompartment` subclass.
Every acquisition entry must instantiate to a callable, which is why function
targets use `_partial_: true`. A supplied `simulator/acquisition` group writes
the same `simulator.acquisitions` list.

Additive symbolic models select an `AdditiveModel` subclass and a coordinate
grid instead of an acquisition factory. The small `symbolic` preset uses
`simulator.num_points`, `simulator.x_min`, and `simulator.x_max`, with
observations shaped `(num_points, 2)` for coordinate and noisy signal columns.
The Figure 4b and K=15/30/50 presets use `observation_format=signal_blocks`:
500 signal values on the fixed grid become `(50, 10)` inputs with implicit
coordinates. Each preset selects a compatible tokenizer and observation
embedding. See the [symbolic guide](symbolic.md) for the input conventions and
extension points.

The main extension points are the simulator class and its prior, the
observation embedding under `model.embedding_net`, and the parameter/mask
tokenizer selected by `model.tokenizer_cls`. Model and embedding configuration
classes can be provided as fully qualified import paths. Start from the
matching dMRI or symbolic preset and keep their input and parameter conventions
consistent.

## Streaming or fixed training data

`training/dataloader=datastream` uses `SimulationDataset`: a finite buffer
whose samples are refreshed in the background as training consumes them.

`training/dataloader=static` uses `StaticDataset`: training and validation
samples are simulated once at startup, then reused. For example:

```bash
uv run prism train +experiment/train=symbolic run.name=symbolic_10k \
  training.dataloader.dataset.dataset_size=10000 \
  training.dataloader.val_dataset.dataset_size=1000 \
  training.max_train_steps=50000
```

This changes the example's dataset size and step cap; it does not select a
paper experiment. `dataset.simulation_batch_size` controls generation batches,
while `train_loader.batch_size` controls optimizer batches.

### Simulate once, then train from files

For a runnable walkthrough of generation, file loading, training, and EMA
restoration, see the [offline training notebook](../examples/symbolic/offline_training.md).

Use `prism simulate` to save symbolic data before training. It builds the
`AdditiveModel` class selected by `simulator.model_class` and uses the configured
prior and observation grid. This example uses the existing polynomial preset:

```bash
uv run prism simulate +experiment/train=symbolic \
  --output results/polynomial_data \
  --train-size 128 --validation-size 32 --simulation-batch-size 32

uv run prism train \
  --config-path "$PWD/results/polynomial_data" --config-name training \
  run.name=polynomial_offline training.max_train_steps=20
```

The first command writes:

```text
results/polynomial_data/
├── train.npz        # Fixed training examples
├── validation.npz   # Independently generated validation examples
├── simulation.yaml  # Configuration used to generate the data
└── training.yaml    # Matching configuration for file-only training
```

The second command uses `FileDataset` to load those files and the existing
trainer to optimize the model. It does not construct simulation generators or
generate new training or validation examples. Training and validation loss,
model and parameter NLL, optimizer state, EMA, best/latest checkpoints, W&B,
and resume use the same
training machinery. Symbolic `signal_blocks` presets also retain predictive RMSE:
the evaluator passes posterior parameter samples through the simulator on the
configured grid and compares predictions with saved observations. It does not
generate new training examples. KSD remains disabled in this mode; run it
separately with the experiment evaluator.

Each file contains `x` (observations), `model_mask`, `theta`, and `mask_prior`
(lambda, or `None` for a fixed prior), plus optional `target_score` values.
The files also record the simulator configuration and random keys. They contain
numeric arrays and JSON, without pickled Python objects. The current loader holds
the fixed dataset in host memory; choose a corpus that fits RAM.

Choose another symbolic preset, such as `+experiment/train=symbolic15`, to change
the model and prior. `run.seed` and any explicit dataset seed overrides determine
the saved samples. Training and validation use separate random keys.
Generation runs on CPU by default; `--device gpu` selects an already available
GPU. `prism simulate` does not submit cluster jobs.

The generated training configuration retains the selected network, batch sizes,
optimizer, stopping settings, and Hydra launcher settings. Both loaders use
`drop_last: false` to keep every saved sample, including a partial final batch.
You can override `training.dataloader.train_loader.batch_size` and
`training.dataloader.val_loader.batch_size` at training time, even above the split
size; a smaller split is returned as one batch. A partial batch may require an
additional JAX compilation for its shape. Changing `run.seed` for a subsequent
training run changes training randomness, not the saved data.

For example, save the Figure 4b 10k dataset, then submit training with the
preset's A100 partition and resource limits:

```bash
uv run prism simulate +experiment/train=symbolic_sbmi_10k \
  --output results/fig4b_10k_data \
  --train-size 10000 --validation-size 1024 --simulation-batch-size 10000

uv run prism train \
  --config-path "$PWD/results/fig4b_10k_data" --config-name training
```

Generation appends `_offline` to `run.name` in `training.yaml`. With the default
output-directory templates, the example above writes training results under
`results/symbolic_sbmi_10k_offline/`. To choose a different name, pass
`run.name=YOUR_RUN_NAME` to `prism train`.

Run the second command on a machine with Slurm access. It submits to
`a100-galvani`; no handwritten Slurm script is needed. Launcher overrides such as
`hydra.launcher.account=YOUR_ACCOUNT` can be supplied at either step. If already
inside a GPU allocation, add `hydra.mode=RUN` to the training command to train in
that allocation. The polynomial preset remains local by default.

These are saved settings, so later changes to a preset do not alter an existing
`training.yaml`. Output-directory templates are evaluated when training starts.
When calling `generate_datasets` from Python, use
`compose(..., return_hydra_config=True)` to include the preset's launch settings.
The strict Figure 4b `--fig4b-point` evaluator currently expects the original
`StaticDataset` configuration and does not yet accept these file-backed runs.

The generated `training.yaml` records a SHA-256 hash for each data file. These
hashes and the simulator configuration are checked before training, including the
first run, so pointing a 100k configuration at a 10k dataset is rejected.
To move the dataset to another machine, copy the whole directory and override
`training.dataloader.dataset.directory` with its new absolute path. The hashes
remain valid when the file contents are unchanged. The run root also records
file paths and hashes in `datasets.json`. Resume with the same dataset files and
`training.continue_training=true` as described below. Resume verifies the recorded
file hashes and stream order; file paths may change when moving the data.
The existing resume path restores optimizer and EMA states, but does not preserve
the data-loader cursor.

### Use your own symbolic model

Define your components and combine them in an `AdditiveModel` subclass. For
example, an importable `my_project.models` module can reuse existing components:

```python
from prism.symbolic.simulators import AdditiveModel, GaussianNoise, Linear, Quadratic


class MyModel(AdditiveModel):
    model_types = [Linear, Quadratic]
    noise_types = [GaussianNoise]
```

Replace those component classes with your own implementations as described in
the [symbolic guide](symbolic.md#adapt-the-simulator-prior-or-training-data).
Select the class in YAML with `simulator.model_class`, or override it when
creating the dataset:

```bash
uv run prism simulate +experiment/train=symbolic \
  simulator.model_class=my_project.models.MyModel \
  --output results/my_model_data --train-size 128 --validation-size 32

uv run prism train --config-path "$PWD/results/my_model_data" --config-name training
```

Keep the model module importable during generation, training, and checkpoint
loading. Component classes define the calculations and parameter transforms;
YAML selects the class, prior, and observation grid. No changes to the file
loader or training loop are needed. Keep the tokenizer and observation encoder
compatible with the selected model and input format.

In Python, `StaticDataset(data)` also accepts a precomputed PyTree of arrays
whose leaves have the same nonempty leading sample dimension. The supplied
training data must include observations and the targets required by the loss
(such as `x`, `theta`, `model_mask`, and `mask_prior`). The CLI static preset
generates data from the simulator; it does not load a dataset file.

## Steps, validation, and stopping

One optimizer update increments the training step by one. The loop performs
`training.inner_steps` updates before logging, checking elapsed time, validating,
or saving by cadence. `training.checkpoint_freq`, `training.eval_freq`, and the
optional `training.restart_every` are rounded down to a multiple of
`inner_steps`; a positive value smaller than `inner_steps` becomes one full
block. Add the last setting with `+training.restart_every=<steps>` because it is
absent from the base config.

`training.max_train_steps` sets an exact, cumulative optimizer-step cap; `null`
disables it. The final block is shortened when needed, so the cap need not be
a multiple of `inner_steps`. On resume, a cap of 50,000 means step 50,000 in
total, not another 50,000 updates. If the restored step already meets the cap,
no updates are performed.

Training stops at the step cap or `training.max_train_hours`, whichever is
reached first. Elapsed time is checked after each block. A normal stop runs
validation when enabled and `training.eval_at_end=true`, and writes a regular
checkpoint even when the final step is off the normal cadence. The symbolic
figure presets set `eval_at_end=false`. The loop can also stop when a
non-finite loss cannot be recovered from a regular checkpoint. Signals and
scheduler termination do not cause a final save.

Periodic validation computes mask and parameter negative log likelihood and,
when configured, KSD. With EMA enabled, validation uses EMA parameters. The sum
of the two negative log likelihood values selects the best checkpoint; KSD does
not select it.

## Outputs

With the default Hydra paths, a named run writes checkpoints and per-launch
metadata in different locations:

```text
results/<run.name>/
|-- checkpoints/
|   |-- <step>/
|   `-- best/
|       `-- <step>/
`-- <timestamp>/
    |-- .hydra/
    |   |-- config.yaml
    |   |-- hydra.yaml
    |   `-- overrides.yaml
    |-- artifact.yaml
    `-- train_script.log
```

A multirun instead gives every job its own root:
`results/<run.name>/<timestamp>/<hydra.job.num>/`. That directory contains
`checkpoints/`, `.hydra/`, `artifact.yaml`, and the training log, so different
jobs do not share checkpoints. `artifact.yaml` contains the model config and
simulator model class needed to rebuild the model. The full
`.hydra/config.yaml` also contains data generation, training, and tracking
settings. Keep the training log: it records validation mask and parameter NLL.

For a custom single-run Hydra output path, the checkpoint root is the nearest
enclosing directory named exactly `run.name`, or the output directory itself
if no such directory exists. In multirun mode, each job's Hydra output
directory is always its checkpoint root. The startup log prints both the Hydra
output directory and the checkpoint directory.

Regular checkpoints are numeric directories immediately under `checkpoints/`.
Orbax retains three by default; set
`+training.checkpoint.max_checkpoints=<count>` to change this. The latest
checkpoint is the highest retained regular step. When best-checkpoint tracking
is enabled, `checkpoints/best/` retains one validation-selected step.

Every regular or best checkpoint contains raw parameters, optimizer state,
model state, step, loss, and RNG state. When `training.track_ema=true`, it also
contains `params_ema` and EMA state. EMA does not replace raw parameters. A best
checkpoint also stores `val_loss`, the validation selection metric. Record both
the final completed training step and the selected checkpoint step when
reporting an evaluation; they can differ.

## Resume training

Continuation is disabled by default. Resume restores the latest regular
checkpoint from the run's checkpoint root. For a single run with default
paths, reuse `run.name` and enable continuation:

```bash
uv run prism train +experiment/train=symbolic run.name=symbolic_example \
  training.continue_training=true training.max_train_steps=200
```

This resumes from `results/symbolic_example/checkpoints/` and creates a new
timestamped log/config directory. Reuse the original simulator, architecture,
and compatible training settings. For custom paths, preserve the same
checkpoint root as described above. A fresh run needs a new name: with
`training.continue_training=false`, training stops at start-up when the run's
checkpoint directory already holds checkpoints, instead of mixing two runs.

For multiruns, set `hydra.sweep.dir` to the existing timestamped sweep directory
and preserve the original `hydra.sweep.subdir` and parameter-to-job mapping.
Otherwise a new launch creates new per-job checkpoint directories and cannot
resume the previous jobs merely by reusing `run.name`.

Normal resume restores parameters, model state, optimizer state, EMA state, RNG,
and step when present. `training.restart_optimizer=true` rebuilds optimizer
state after loading. `training.partial_restore=true` requests an Orbax partial
parameter restore and also rebuilds optimizer and EMA state. Simulation buffers,
loader position, and background producer state are not checkpointed, so a
resumed run should not be described as an exact continuation of sample order.

## Load a checkpoint in Python

```python
from flax import nnx
from prism.train.utils import load_checkpoint, select_checkpoint_params

checkpoint, model, simulator_model = load_checkpoint(
    "/absolute/path/to/results/my_run",
    which="best",
)
nnx.update(model, select_checkpoint_params(checkpoint, "params_ema"))
model.eval()
```

`which` accepts `"latest"`, `"best"`, or an integer regular step. Loading
returns the checkpoint payload and a rebuilt model; it does not apply the
returned parameters to that model. The example explicitly selects EMA weights.
Pass `"params"` to `select_checkpoint_params` to select raw weights. Missing
requested weights raise `ValueError`; the selector never substitutes another
parameter set.

Pass the directory that contains `checkpoints/`, not its timestamped metadata
directory. `load_cfg` reads a root `config.yaml` or `.hydra/config.yaml` when
present; otherwise it selects the newest timestamped config (or job `0` within
it). Keep configurations consistent across resumptions, and use a checkpoint
bundle to preserve a specific config alongside selected weights.

## Hand off to evaluation

For a dMRI run at `/work/results/my_run`, evaluation expects the parent as
`checkpoint.path` and the run name as `checkpoint.model_name`:

```bash
uv run prism eval \
  checkpoint.path=/work/results \
  checkpoint.model_name=my_run \
  checkpoint.which=best \
  evaluation.input.path=/work/data/subject
```

A Slurm or other multirun launch keeps its checkpoints in
`results/<run.name>/<timestamp>/<job>/`. `prism eval` and the experiment
evaluators accept `results/<run.name>` for such a run: they use the newest
timestamp, print the selected directory, and refuse a sweep with several jobs
or a newest job without a saved step. To choose a job yourself, use the numbered
job directory: its parent is `checkpoint.path` and its number is
`checkpoint.model_name`. For custom layouts, use the directory containing
`checkpoints/` in the same way.
The dMRI evaluation presets expect acquisition data and dMRI export
settings; they are not a symbolic evaluation preset. Evaluation uses
`checkpoint.params_name=params_ema` by default. Set `checkpoint.params_name=params`
to select raw weights; missing requested weights raise `ValueError`. The chosen
weights and restored step are logged. See the
[evaluation guide](evaluation.md) for input, sampling, metrics, and export
settings, and the [symbolic guide](symbolic.md#evaluate-the-figure-metrics) for
the figure evaluation scripts.

## Bundle or publish a checkpoint

```python
from prism.train.utils import bundle_checkpoint, upload_checkpoint_to_hub

bundle_checkpoint(run_dir, "bundles", model_name="my_model")
upload_checkpoint_to_hub(
    run_dir,
    "owner/models",
    model_name="my_model",
    private=True,
)
```

Both functions include best and latest by default. `which` also accepts
`"best"`, `"latest"`, `"all"`, or an integer regular step. A bundle stores the
model under `<output_dir>/<model_name>/` with `config.yaml`, `artifact.yaml`, and
the selected checkpoint tree.
