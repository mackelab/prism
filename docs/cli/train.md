# `prism train`

Train a model from the Hydra configuration in `conf/train.yaml`. See the
[training guide](../guides/training.md) for datasets, launchers, validation,
tracking, and checkpoint handling.

## Syntax

```text
uv run prism train [HYDRA_OPTIONS] [OVERRIDE ...]
uv run prism train --multirun [HYDRA_OPTIONS] [OVERRIDE ...]
```

| Option | Effect |
| --- | --- |
| `--cfg job` | Print the composed application config without training. |
| `--cfg all` | Include Hydra settings, including preset sweep parameters. |
| `--resolve` | Resolve interpolations when printing config. |
| `--info defaults` | Show the selected config groups. |
| `-m`, `--multirun` | Apply sweep parameters and execute through the configured launcher. |

For example, inspect a preset without starting it:

```bash
uv run prism train +experiment/train=symbolic --cfg job --resolve
```

## Common overrides

| Purpose | Override |
| --- | --- |
| Training preset | `+experiment/train=symbolic` |
| Run name and seed | `run.name=my_run run.seed=0` |
| Simulator class | `simulator.model_class=my_package.models.MyModel` |
| dMRI acquisition group | `simulator/acquisition=multi` |
| Batch size | `training.dataloader.train_loader.batch_size=1024` |
| Learning rate | `training.optimizer.learning_rate=3e-4` |
| EMA tracking | `training.track_ema=true` |
| Exact cumulative step cap | `training.max_train_steps=50000` |
| Training-loop time limit | `training.max_train_hours=1` |
| Resume from latest regular checkpoint | `training.continue_training=true` |
| Retained regular checkpoints | `+training.checkpoint.max_checkpoints=5` |
| W&B tracking | `tracking.enabled=true tracking.wandb.entity=ENTITY tracking.wandb.project=PROJECT` |

See [Configuration](../reference/configuration.md) for config groups and
[Experiment presets](../reference/experiments.md) for available recipes.

## Execution

A single run executes in the current process; presets that select the Slurm
launcher switch Hydra to multirun and submit a job instead (add `hydra.mode=RUN`
to run them in the current process). Multirun uses the configured
Submitit launcher and applies `hydra.sweeper.params`. See the training guide
for [local runs, sweeps, and Slurm](../guides/training.md#choose-how-the-run-executes).

## Stopping

Training stops at `training.max_train_steps` or `training.max_train_hours`,
whichever is reached first. The step cap is cumulative across resumptions;
`null` disables it. Time is checked after each block of `training.inner_steps`.
See [steps, validation, and stopping](../guides/training.md#steps-validation-and-stopping)
for final validation, checkpoint saves, and scheduler termination.

## Output

Default single runs keep checkpoints under `results/<run.name>/checkpoints/`
and launch metadata under `results/<run.name>/<timestamp>/`. Multiruns give each
job its own checkpoint and metadata directory. See the
[output layouts](../reference/outputs.md#training-run) and
[resume instructions](../guides/training.md#resume-training).
