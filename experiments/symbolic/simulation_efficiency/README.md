# Symbolic simulation efficiency (Figure 4b)

Train PRISM on fixed datasets of 10k–1000k symbolic simulations and report
predictive RMSE minus the historical noise floor. The
[Figure 4b documentation](../../../docs/experiments/fig4b.md) describes the
protocol, completion evidence, reduced checks and all options.

| File | Role |
| --- | --- |
| `train.py` | Completion recorder: trains a 100k–1000k point and writes `training-complete.json` |
| `evaluate.py` | Figure 4b evaluator |
| `completion.py` | Checks completion records |

## Reproduce Figure 4b

Run from the repository root. `prism train` submits a Slurm job with the
resources set in its preset. The scripts run in the current shell unless the
command ends in `--slurm a100`, which submits them as a Slurm job instead. To
run elsewhere, see
[another site or without Slurm](../../../docs/guides/training.md#run-at-another-site-or-without-slurm).

### Training

Train 10k, 25k and 50k with their presets:

```bash
uv run prism train +experiment/train=symbolic_sbmi_10k
uv run prism train +experiment/train=symbolic_sbmi_25k
uv run prism train +experiment/train=symbolic_sbmi_50k
```

Train 100k, 200k, 500k and 1000k through the completion recorder; the
evaluator requires its record for these time-limited points:

```bash
uv run python -m experiments.symbolic.simulation_efficiency.train --point 100k --output results/symbolic_sbmi_100k --slurm a100
uv run python -m experiments.symbolic.simulation_efficiency.train --point 200k --output results/symbolic_sbmi_200k --slurm a100
uv run python -m experiments.symbolic.simulation_efficiency.train --point 500k --output results/symbolic_sbmi_500k --slurm a100
uv run python -m experiments.symbolic.simulation_efficiency.train --point 1000k --output results/symbolic_sbmi_1000k --slurm a100
```

### Evaluation

Evaluate each point with its best EMA weights. The commands for 10k and 200k
are shown; the other points follow the same pattern:

```bash
uv run python -m experiments.symbolic.simulation_efficiency.evaluate results/symbolic_sbmi_10k --fig4b-point 10k --which best --slurm a100 --output results/fig4b_10k_best.json
uv run python -m experiments.symbolic.simulation_efficiency.evaluate results/symbolic_sbmi_200k --fig4b-point 200k --which best --slurm a100 --output results/fig4b_200k_best.json
```

Each evaluation writes a JSON summary and an adjacent NPZ. Combined plotting of
the seven points is not packaged.

To evaluate the published best EMA weights of a point
(`manugloeck/prism-symbolic-pretrained`) instead of a training run, pass the
preset instead of the run directory; `--weights-dir DIR` reads a local copy:

```bash
uv run python -m experiments.symbolic.simulation_efficiency.evaluate --preset symbolic_sbmi_10k --fig4b-point 10k --which best --output results/fig4b_10k_published.json
```

With `--preset`, `--fig4b-point` enforces the evaluation protocol and the
matching preset, but no training evidence exists to check; the JSON records the
weights' hash and sets `fig4b_protocol_checks_passed` to false.

### Training from saved files (optional)

To simulate a point's dataset once into files and train only from those files,
shown for 10k. The strict `--fig4b-point` check accepts only preset-trained
runs, so evaluate a file-trained run without it, as a diagnostic:

```bash
# Simulate: runs in the current shell (CPU, under a minute), writes results/fig4b_10k_data/
uv run prism simulate +experiment/train=symbolic_sbmi_10k --output results/fig4b_10k_data --train-size 10000 --validation-size 1024 --simulation-batch-size 10000
# Train: Slurm job on a100, with the preset's settings saved in results/fig4b_10k_data/training.yaml
uv run prism train --config-path "$PWD/results/fig4b_10k_data" --config-name training
# Evaluate (diagnostic): Slurm job on a100, because of --slurm a100
uv run python -m experiments.symbolic.simulation_efficiency.evaluate results/symbolic_sbmi_10k_offline --which best --slurm a100 --output results/fig4b_10k_offline_best.json
```

The [training guide](../../../docs/guides/training.md#simulate-once-then-train-from-files)
describes the saved files, their hashes and moving the data.
