# Symbolic model-space scaling (Figure 4c)

Train prior-conditioned PRISM models with 15, 30 or 50 expressions and report
conditional parameter-prediction error. The
[Figure 4c documentation](../../../docs/experiments/fig4c.md) describes the
protocol, outputs, reduced checks and all options.

| File | Role |
| --- | --- |
| `evaluate.py` | Figure 4c evaluator |

## Reproduce Figure 4c

Run from the repository root. `prism train` submits a Slurm job with the
resources set in its preset. The scripts run in the current shell unless the
command ends in `--slurm a100`, which submits them as a Slurm job instead. To
run elsewhere, see
[another site or without Slurm](../../../docs/guides/training.md#run-at-another-site-or-without-slurm).

### Training

Each preset trains for 24 hours with online simulation:

```bash
uv run prism train +experiment/train=symbolic15
uv run prism train +experiment/train=symbolic30
uv run prism train +experiment/train=symbolic50
```

### Evaluation

```bash
uv run python -m experiments.symbolic.model_space_scaling.evaluate results/symbolic15 --output results/symbolic15_evaluation.json --slurm a100
uv run python -m experiments.symbolic.model_space_scaling.evaluate results/symbolic30 --output results/symbolic30_evaluation.json --slurm a100
uv run python -m experiments.symbolic.model_space_scaling.evaluate results/symbolic50 --output results/symbolic50_evaluation.json --slurm a100
```

Each evaluation writes a JSON report and an adjacent NPZ. The figure's complete
model-space/width matrix and control modes are not packaged.

To evaluate the published EMA weights instead of a training run, pass the
preset (`symbolic15`, `symbolic30`, `symbolic50`, also with `_small`/`_tiny`)
instead of the run directory; `--weights-dir DIR` reads a local copy. The
published configuration has no training section, so only the simulator and
model settings are checked:

```bash
uv run python -m experiments.symbolic.model_space_scaling.evaluate --preset symbolic15 --which latest --output results/symbolic15_published.json
```
