# Examples

## Symbolic notebooks

| Notebook | Content |
| --- | --- |
| [minimal.ipynb](symbolic/minimal.ipynb) | A standalone polynomial example: simulation, training, EMA checkpoint save/restore, posterior sampling, and predictions |
| [offline_training.ipynb](symbolic/offline_training.ipynb) | The same polynomial simulator: save fixed datasets, inspect FileDataset, train from files, and restore EMA for inference |

```bash
uv run python -m jupyter lab examples/symbolic/minimal.ipynb
```

The minimal notebook uses the repository environment, with no downloaded training data
or pretrained weights. It runs 1,000 training steps with batch size 32 and 100
observation points. Visualizations explain observation features, component tokens,
lambda conditioning, autoregressive masks, and parameter and predictive samples.
The short training run illustrates the workflow; it does not establish posterior
accuracy or calibration.

Additional sections display saved Figure 4b comparison results and contrast
symbolic and dMRI simulator inputs. The Figure 4b values are embedded for display;
running this notebook does not rerun those experiments.

## dMRI notebooks

The dMRI tutorials retain their original filenames:

| Notebook | Content |
| --- | --- |
| [01_dmri_simulator_components.ipynb](dmri/01_dmri_simulator_components.ipynb) | Signals from individual compartments |
| [02_dmri_multicompartment_model.ipynb](dmri/02_dmri_multicompartment_model.ipynb) | Constructing and simulating mixtures |
| [03_pretrained_models.ipynb](dmri/03_pretrained_models.ipynb) | Loading a public checkpoint and inspecting posteriors |
| [04_train_standalone.ipynb](dmri/04_train_standalone.ipynb) | Training a small model with a standalone loop |
| [05_example_data.ipynb](dmri/05_example_data.ipynb) | Downloading data, preparing FSL inputs, and running prediction |

```bash
uv run python -m jupyter lab examples/dmri
```

Notebook 03 downloads a public checkpoint on first use. Notebook 05 includes a
174 MB data download and full-volume prediction; run it separately when needed.
The documentation build executes notebooks 01–04 and uses saved outputs for 05.

Each notebook has one source under `examples/`. Reading pages and downloads in
`docs/examples/` are generated; see the
[rendering instructions](../docs/development.md#documentation).

## Command-line check

[`smoke_polynomial.py`](smoke_polynomial.py) checks a small polynomial workflow from the
command line. Run it with a fresh output directory:

```bash
uv run python examples/smoke_polynomial.py --output-dir results/polynomial_check
```

It uses 20 training steps and 32 observation points on CPU, saving checkpoints,
`smoke.json`, and `samples.npz`. See the [symbolic guide](../docs/guides/symbolic.md)
for configuration and API details. Scientific evaluations live in
[`experiments/`](../experiments/README.md).

For a runnable guide to fixed symbolic datasets, open
[`offline_training.ipynb`](symbolic/offline_training.ipynb). It uses 128 training
samples, 32 validation samples, and 20 CPU training steps. See also the
[offline training commands](../docs/guides/training.md#simulate-once-then-train-from-files).
