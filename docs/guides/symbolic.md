# Symbolic regression

PRISM infers which expressions contribute to a signal and the continuous
parameters of those expressions. The symbolic workflows share the training
and checkpoint machinery with dMRI, with their own simulators, observation
formats, and network configurations.

Follow [installation](../getting-started.md#install-the-runtime) first. Run the commands
below from the repository root with the environment active. The
[experiment catalog](../reference/experiments.md) lists the supported presets;
the [training guide](training.md) covers output paths, Slurm, tracking, and
continuation.

## Choose an experiment

| Workflow | Training preset | Simulator and prior | Observation shape |
| --- | --- | --- | --- |
| Small polynomial example | `symbolic` | `SimplePolynomial`: linear and quadratic terms, one Gaussian noise component; Beta-Bernoulli prior | `(32, 2)`, coordinate and signal columns |
| Figure 4b comparison | `symbolic_sbmi_*k` (seven sizes, 10k–1000k) | `SymbolicModelV0`: 15 expressions and 5 noise alternatives; fixed empirical mask prior | `(50, 10)`, signal blocks |
| Figure 4c scaling | `symbolic15`, `symbolic30`, `symbolic50`, `symbolic80`, `symbolic100` (and `_small`/`_tiny` widths) | `SymbolicModelV15`, `V30`, `V50`, `V80`, `V100`: 15/30/50/80/100 expressions and 5/8/8/8/8 noise alternatives; variable Beta-Bernoulli prior | `(50, 10)`, signal blocks |

Both figure workflows evaluate 500 coordinates from 0 to 10, excluding 10,
at spacing 0.02. The network receives the signal reshaped into 50 blocks of
10 consecutive values; coordinates are implicit in that fixed grid. A model
mask contains expression bits followed by noise bits. The full mask/parameter
dimensions are 20/28 for K=15, 38/73 for K=30, and 58/115 for K=50.

Figure 4b samples its fixed prior from the packaged empirical mask tables in
`prism/symbolic/paper/data/`. Its model-selection head has no prior
input. Figure 4c samples `lambda` from `Beta(1, 1)` during training and passes
it to a prior-conditioned head. Expressions are sampled with Bernoulli
probability `lambda`, with one uniformly chosen expression forced active;
exactly one noise alternative is selected. Larger `lambda` therefore favors
more expressions. The K=15 scaling preset has 15 expressions and is exposed as `symbolic15`.

## Train

For a small fixed dataset and a short checkpointed run:

```bash
uv run prism train +experiment/train=symbolic run.name=polynomial_example
```

See the [training guide](training.md#small-symbolic-example) for this preset
and the [quickstart notebooks](../examples/index.md) for a complete Python
workflow from simulation through posterior prediction.

Full symbolic experiments require larger datasets and run as Slurm GPU jobs.
Their training and evaluation commands are kept in a recipe with each task,
and the protocol on its documentation page:

- Simulation efficiency: fixed datasets of 10k–1000k simulations (Figure 4b).
  [Recipe](https://github.com/mackelab/prism/tree/main/experiments/symbolic/simulation_efficiency#reproduce-figure-4b)
  · [details](../experiments/fig4b.md).
- Model-space scaling: online simulation with 15, 30, or 50 expressions
  (Figure 4c).
  [Recipe](https://github.com/mackelab/prism/tree/main/experiments/symbolic/model_space_scaling#reproduce-figure-4c)
  · [details](../experiments/fig4c.md).

The [experiment catalog](../reference/experiments.md#symbolic-training) lists
architecture, prior, batch, and stopping settings. The
[training guide](training.md#choose-how-the-run-executes) covers launchers,
W&B, and resume.

## Load EMA weights and infer

The checkpoint loader rebuilds the model and returns its saved state. Apply
the selected weights before inference. This example uses a trained K=15 run:

```python
import jax
import jax.numpy as jnp
from flax import nnx

from prism.train.build_simulator import build_simulator
from prism.train.utils import load_cfg, load_checkpoint, select_checkpoint_params

run_dir = "results/symbolic15"
checkpoint, model, simulator_class = load_checkpoint(run_dir, which="latest")
nnx.update(model, select_checkpoint_params(checkpoint, "params_ema"))
model.eval()

cfg = load_cfg(run_dir)
_, generators = build_simulator(cfg)
data = generators[0](jax.random.key(1))
prior = jnp.array([0.2])

mask = model.sample_mask(jax.random.key(2), x=data["x"], mask_prior=prior)
log_probability = model.log_prob_mask(mask, x=data["x"], mask_prior=prior)
theta = model.sample_theta(
    jax.random.key(3),
    x=data["x"],
    model_mask=mask,
    sample_method="ode",
    num_steps=64,
)

grid = jnp.arange(cfg.simulator.x_min, cfg.simulator.x_max, cfg.simulator.x_step)
sampled_model = simulator_class.create_from_theta(theta, model_mask=mask)
mean_signal = sampled_model.evaluate(grid)
noisy_signal = sampled_model.evaluate(grid, rng=jax.random.key(4))
```

`select_checkpoint_params` raises if EMA weights are absent; it never silently
uses raw weights. `which` accepts `"latest"`, `"best"`, or an integer saved
step. For Figure 4b, load the matching run and use `mask_prior=None`. For the
polynomial example, the grid comes from the first column of `data["x"]`.

To analyze your own signal, replace `data["x"]` with a floating-point array in
the checkpoint's observation format. Figure checkpoints expect `(50, 10)`
signal blocks on the training grid; the polynomial model expects coordinate
and signal pairs. Supply the same `prior` to `sample_mask` and `log_prob_mask`.
Changing that value queries the trained prior-conditioned posterior; it does
not retrain the network. Parameter inference conditions on the chosen mask.
`theta` is the simulator's unconstrained parameter vector;
`create_from_theta` transforms it to the bounded expression and noise parameters.

## Evaluate the figure metrics

The two tasks measure different parts of inference:

| Task | Metric and checkpoint selection |
| --- | --- |
| [Simulation efficiency](../experiments/fig4b.md#evaluation) | Joint mask and parameter prediction; RMSE minus the stored noise floor. Best EMA is primary, latest EMA supplementary. |
| [Model-space scaling](../experiments/fig4c.md#evaluation) | Parameter prediction conditioned on the true mask; RMSE minus a separately simulated noise baseline. Uses latest EMA. |

Each task page defines its evaluation grid, sample count, aggregation and
outputs; its recipe lists the run commands. Use that protocol when comparing results; the two metrics
are not interchangeable.

## Adapt the simulator, prior, or training data

Define expression components with `SymbolicExpressionCompartment` and noise
components with `SymbolicNoiseCompartment`. Each declares its SymPy expression,
parameter symbols, `theta_dim`, and parameter bounds. Compose them in an
`AdditiveModel` subclass through `model_types` and `noise_types`, then select
its import path with `simulator.model_class`. The
[symbolic simulator API](../reference/simulators.md#symbolic-models) documents
these interfaces. Preserve component order: it determines mask entries and
the layout of `theta`.

An additive model selects its prior through `mask_prior_cls` and
`mask_prior_kwargs`; `simulator.mask_prior` supplies configuration overrides.
For the variable-prior presets, `alpha`, `beta`, and `min_active_models` control
the training distribution. The builder passes these settings to both the
simulator and the prior-conditioned model-selection head. A different prior
family can also require a matching head and prior input dimension. Select
compatible observation embeddings and tokenizers through
`model.embedding_net` and `model.tokenizer_cls`; see the
[neural network API](../reference/nn.md).

`build_simulator(cfg)` resolves the simulator class and returns the generators
from its `training_generators` hook. A simulator family outside `AdditiveModel`
can plug in the same way by implementing the
[simulator interface](../reference/simulators.md#simulator-interface). Each
symbolic generator accepts a JAX key and returns `x`, `theta`, `model_mask`,
and `mask_prior`. `StaticDataset(data)` accepts a precomputed PyTree with those
fields and a common leading sample dimension; `SimulationDataset` maintains
a buffer refreshed by simulation. The CLI's static presets generate their
data from the configured simulator. For supplied training arrays, use the
[Python dataset interfaces](../reference/train.md#simulation-data) and
[minimal symbolic notebook](../examples/symbolic/minimal.md).
