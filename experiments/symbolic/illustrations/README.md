# Symbolic posterior illustrations (Figures 2, 4a, A.1)

One simulated observation, queried through a trained model: which components does
the model-selection network include as the mask-prior hyperparameter λ grows, what
functions do complete posterior draws describe, and how is the parameter posterior of
the simplest explanation shaped?

| File | Role |
| --- | --- |
| `illustrate.py` | simulates the observation, runs the queries, writes `arrays.npz` and `summary.json` |
| `plot.py` | the panels: model fits and median component count (Figure 2), model posterior across λ and parameter posterior (Figure 4a, A.1), the equations |

## Run it

From the repository root. Add `--slurm a100` to submit a Slurm job instead.

**Figures 2 and 4a** use a K=50 model trained on ten observations (`x = 0, 1, …, 9`):

```bash
uv run prism train +experiment/train=symbolic50_illustrative   # 72 h, seed 1, batch 8192
uv run python -m experiments.symbolic.illustrations.illustrate results/symbolic50_illustrative \
    --output-dir results/symbolic50_illustrative/illustrations
uv run python -m experiments.symbolic.illustrations.plot \
    results/symbolic50_illustrative/illustrations --output-dir results/symbolic50_illustrative/panels
```

The paper's panels come from the checkpoint at step 1,396,000 of that run (published as
`symbolic50_illustrative/figure`); `--which latest` selects the run's last step instead.

**Figure A.1** uses the standard K=50 model (`symbolic50`, latest step 362,000) on the
full observation grid (`x = 0, 0.02, …, 9.98`, 500 observations; the caption's "1000
observations" does not match the original analysis) and the same simulated observation as
Figure 2 (seed 69, ground truth `4.3·log(x + 1.5) + 6.7`), without the Figure 2 edit.
The notebook shows fixed posterior draws instead of scoring them (the draw at an index
depends on the seed and on `--num-samples`, so use `--num-samples 1000` with the stated
indices): draw 627 at λ = 0.5, draw 12 at λ = 0.1, and the parameter posterior of the
model of draw 12. With the published weights
(`manugloeck/prism-symbolic-pretrained`, needs the safetensors loader of
`prism.symbolic.pretrained`):

```bash
uv run python -m experiments.symbolic.illustrations.illustrate --preset symbolic50 --which latest \
    --seed 69 --no-edit-observation --num-samples 1000 \
    --display-index 0.5=627 --display-index 0.1=12 --posterior-index 12 \
    --output-dir results/symbolic50/illustrations
```

This reproduces the figure's equations exactly: `2.7·√(x + 0.4) + 8.3` at λ = 0.1 and
`1.4·10⁻⁵·x⁵ + 10.7 + 4.5·exp(−2.9·exp(−0.8x)) − 3.5·exp(−1.6x)` at λ = 0.5. These settings
come from the original analysis that produced the A.1 panels; an earlier version of
this recipe (seed 147, 5,000 draws) followed a later exploration that saved no
figure panels.

`--weights-dir DIR` reads a local copy (`DIR/config.yaml`, `DIR/latest/params_ema.safetensors`)
instead of the Hub. The Figures 2 and 4a command above also works with
`--preset symbolic50_illustrative --which figure`.

## What is computed

* The observation: generator draw with `--seed` (default 69) and mask-prior hyperparameter 0.02.
  For Figures 2 and 4a the Figure 2 notebook first overwrites the first three
  values (7.1, 13.567, 9.0002313); `--no-edit-observation` keeps the simulated ones.
* `model_space`: inclusion probability of each component for 100 values of λ in
  [0.01, 0.99], from 100 mask samples each.
* `component_quantiles`: median and 95% interval of the number of active signal
  components for 100 values of λ in [0, 1] (500 samples each).
* Complete posterior draws (mask, then parameters) at λ = 0.9, 0.5, 0.1
  (`--num-samples`, default 50,000 each). For λ = 0.5 and 0.1 the displayed
  function is the draw given by `--display-index`, or else the one minimising `error**20 + 0.1 * components` against the
  observations, drawn with its 95% predictive band (noisy predictions, smoothed).
* `parameter_posterior`: `--num-posterior` (default 100,000) parameter draws for the
  mask of the λ = 0.1 draw, in the components' natural parameters.
* `summary.json`: the true and the displayed equations (SymPy and LaTeX), their
  component counts, and the run's step.

Component counts exclude the noise-component flags in the mask.
