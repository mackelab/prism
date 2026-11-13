# Configuration

`prism train` and `prism eval` use Hydra. Configuration files are under `conf/`.
Command-line overrides use dotted keys, and config groups use `group=name`.
The [experiment catalog](experiments.md) describes each supported training and
evaluation preset. See the [training guide](../guides/training.md) for execution
and checkpoint workflows.

## Configuration sources and compatibility

The dMRI configuration tree comes from
[`mackelab/dmri` at `24c6373`](https://github.com/mackelab/dmri/tree/24c6373ea0088719159a6e200c642ba5149b7367/conf).
PRISM Clean adapted its execution defaults and fixed preset composition. The
Figure 4b and K=15/30/50 symbolic presets were ported from
[`sbmi_transformer_symbolic` at `c8e118b`](https://github.com/mackelab/sbmi_transformer_symbolic/tree/c8e118b8bd557f2b001812f00af982b136f0d793/conf);
the small `symbolic` polynomial preset is an integration example.

Configs name classes by import path: core classes under `prism.nn` and
`prism.simulators`, application classes under `prism.dmri` and
`prism.symbolic`. Group names such as `model=dmri` are unchanged.
Configuration tests cover the named training and evaluation presets, their
composition, and migration of older layouts.

For a new run, choose the matching model family and set data paths, compute
resources, and training budget. Named partition presets contain site-specific
queues; convolved `2gpus` presets request two GPUs. To evaluate an
existing checkpoint, let the loader rebuild its architecture and simulator
from the saved configuration. Preserve that configuration when resuming.

An upstream-derived preset does not identify the weights or complete settings
used for a paper figure. See the [B3S preset notes](experiments.md#ball-and-sticks)
and [paper checkpoint and result records](../paper_experiments.md#checkpoints-and-saved-source-results)
before selecting a reproduction recipe.

## Inspect a configuration

Print a composed configuration without running it:

```bash
uv run prism train +experiment/train=b3s_2_4_6_128 --cfg job --resolve
uv run prism eval --cfg job --resolve
```

List Hydra help and available groups:

```bash
uv run prism train --help
uv run prism eval --help
```

Use `+experiment/train=NAME` or `+experiment/eval=NAME` because experiment
groups are not selected by the base defaults.

## Override syntax

Set scalar values with dotted keys:

```bash
uv run prism train +experiment/train=symbolic run.name=test training.max_train_hours=1 tracking.enabled=false
uv run prism eval checkpoint.model_name=my_run
```

Select a group:

```bash
uv run prism train simulator=ball3stick_shared simulator/acquisition=multi model=dmri
uv run prism eval evaluation/selection=ball3stick_best
```

Replace a nested group at a package path:

```bash
uv run prism eval \
  evaluation/theta/corrector@evaluation.sampling.theta.corrector=none
```

Quote list values and removal overrides in a shell:

```bash
uv run prism eval \
  evaluation.pipeline.default_mask='[true,true,true,false,true]' \
  '~evaluation.export.theta.metrics'
```

Hydra rejects keys that are not present in a structured config. Prefix a new
optional key with `+`, for example `+training.restart_every=100000`.

## Training tree

The base training config is `conf/train.yaml`:

```text
run
tracking
simulator
model
training
infrastructure
```

Important groups are:

- `conf/simulator/`: importable simulator classes that implement the
  [simulator interface](simulators.md#simulator-interface) (the dMRI
  `MultiCompartment` or symbolic `AdditiveModel` families) and their
  data-generation settings. There is no default; choose one or a preset.
- `conf/simulator/acquisition/`: dMRI acquisition generator partials; dMRI
  simulator presets default to `multi`.
- `conf/model/`: network and embedding configuration.
- `conf/training/`: loop, dataloader, and optimizer settings.
- `conf/infrastructure/`: Hydra launcher and resource settings.
- `conf/experiment/train/`: named combinations and scientific settings.

Training and evaluation default to `infrastructure/launcher=local` and
`infrastructure/partition=none`; training also defaults to
`tracking.enabled=false` and `training.continue_training=false`. Every training
preset that runs longer than an hour selects the `slurm` launcher and a partition:
`h100_long` for the dMRI presets, `2h100_long` for the two-GPU presets, and
`a100` for the symbolic presets. Only `symbolic` stays local. The `slurm`
launcher sets `hydra.mode=MULTIRUN`; partition presets set only resources. To
submit a configuration that runs locally by default:

```bash
uv run prism eval +experiment/eval=eval_test \
  infrastructure/launcher=slurm infrastructure/partition=a100
```

Add `hydra.mode=RUN` to run a submitting preset in the current process instead.
The partition presets name queues of the Tübingen cluster; override
`hydra.launcher.partition` or add a preset for your site. W&B tracking requires `tracking.enabled=true`; set
`tracking.wandb.entity` and `tracking.wandb.project` explicitly as shown in the
[tracking example](../guides/training.md#weights-biases-tracking).

The default output directories are:

```yaml
hydra:
  run:
    dir: results/${run.name}/${now:%Y-%m-%d_%H-%M-%S}
  sweep:
    dir: results/${run.name}/${now:%Y-%m-%d_%H-%M-%S}
    subdir: ${hydra.job.num}
```

Named presets set ordinary model and training fields, so their scientific
settings apply to both single runs and multiruns. In particular,
`b3s_2_4_6_128` and `msb3s_2_4_6_128` select width 128 and 2/4/6 observation,
model-selection, and parameter-inference layers in either mode.

Comma-separated overrides such as `run.seed=0,1,2` define a sweep with
`--multirun`. Custom values under `hydra.sweeper.params` are also multirun-only;
they do not change a single run. Each multirun job owns its numbered output
and checkpoint directory. Single runs keep checkpoints under the named run
root and per-launch metadata under its timestamped directory; see
[training outputs](../guides/training.md#outputs).

Symbolic simulator input formats depend on the preset: the polynomial example
uses coordinate/signal pairs, while the figure presets use signal-only blocks.
The [symbolic guide](../guides/symbolic.md) describes their shapes and priors.

## Evaluation tree

The base evaluation config is `conf/eval.yaml`:

```text
run
checkpoint
evaluation
  input
  pipeline
  sampling.mask
  sampling.theta
  selection
  export.theta
  export.model_selection
```

Important groups are:

- `conf/evaluation/input/`: file or synthetic data.
- `conf/evaluation/mask/`: model-mask sampling.
- `conf/evaluation/theta/`: theta sampling and correctors.
- `conf/evaluation/selection/`: average, best, or no model selection.
- `conf/evaluation/export/`: exporters and metrics.
- `conf/experiment/eval/`: named evaluation configurations.

The default evaluation uses file input, samples masks and theta, uses no
selection transform, and exports Ball3Stick theta and model-selection results.

## Prediction tree

`prism predict` composes `conf/predict.yaml`, which lists `eval` in its defaults —
so it inherits the whole evaluation tree above — and then layers the
prediction-specific policy:

```text
conf/predict.yaml
conf/predict/
  quality/       very-fast, fast, balanced, high
  model_mode/    per-sample, best, fixed
  fixed_model/   B1S, B2S, B3S
  viewer/        default, none
```

The CLI is a front end over this: `--quality fast` becomes
`predict/quality=fast`, `--fixed-model B2S` becomes `predict/fixed_model=B2S`,
and `--set key=value` passes anything else straight through. Because each preset
is a group choice rather than a set of inlined numbers, the run's
`.hydra/hydra.yaml` records *which* preset was used, not only its resolved
values.

A quality option sets the sampling keys and selects a corrector. It needs
`override` on the corrector because `conf/evaluation/theta/default.yaml` already
chooses one:

```yaml
# conf/predict/quality/fast.yaml
# @package _global_
defaults:
  - override /evaluation/theta/corrector@evaluation.sampling.theta.corrector: none

evaluation:
  precision: fp16
  sampling:
    theta: {num_samples: 25, params: {num_steps: 20}}
    mask: {n_samples: 25}
```

Prediction also selects `evaluation/export/theta: ball3stick_predict`, which is
`ball3stick` without the metrics exporters — there is no ground truth at
prediction time to score against.

To run the same presets through `prism eval` instead, compose `predict` and add
what the `prism predict` command fills in itself: the checkpoint, the input
folder and the output directory.

```bash
uv run prism eval --config-name predict predict/quality=fast --cfg job   # inspect
uv run prism eval --config-name predict predict/quality=fast \
  checkpoint.pretrained.repo_id=manugloeck/dmri-pretrained \
  checkpoint.model_name=msb3s_2_4_6_128 \
  evaluation.input.path=/data/sub-001/dwi run.output_dir=/data/sub-001/dwi/dmri_output
```

## Saved configuration

Hydra writes these files in each run directory:

```text
.hydra/
|-- config.yaml
|-- hydra.yaml
`-- overrides.yaml
```

They record the composed application configuration, Hydra configuration, and
command-line overrides; saved values may retain interpolations. Reproducing a
run can still depend on package versions,
source revisions, devices, external data, and random seeds that are not embedded
in those files.

Training also writes `artifact.yaml`. This is a small model-construction
manifest used by portable checkpoints. It is not a replacement for the full
saved training configuration.

## Legacy configurations

The loader normalizes older schema layouts when reading saved runs. Class
names and paths written by earlier versions, such as
`DMRIInferenceModelConfigMaskPriorAmortizedPP` or
`prism.simulators.dmri.models.Ball3StickSharedDiffusivity`, resolve through
`prism.config.aliases`, so older checkpoints and the published pretrained
models keep loading. New configs
should use the current schema shown above, including `simulator.model_class`,
`simulator.acquisitions`, `simulator.mask_prior`, and
`simulator.posterior_score`.
