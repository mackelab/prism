# Examples

## Symbolic

[Minimal example](symbolic/minimal.md): a standalone polynomial workflow covering
simulation, training, EMA checkpoint save/restore, posterior inference, and predictions.
No downloaded training data or pretrained checkpoint is required.

[Offline training](symbolic/offline_training.md): save polynomial simulations,
inspect the files, train through `FileDataset`, and restore an EMA checkpoint.
The notebook uses 20 training steps on CPU.

Source: [minimal.ipynb](https://github.com/mackelab/prism/blob/main/examples/symbolic/minimal.ipynb).
Offline training source: [offline_training.ipynb](https://github.com/mackelab/prism/blob/main/examples/symbolic/offline_training.ipynb).
See [examples/README.md](https://github.com/mackelab/prism/blob/main/examples/README.md)
for execution commands and the command-line smoke check.

## dMRI

1. [Simulator components](dmri/01_dmri_simulator_components.md): simulate individual compartments.
2. [Multi-compartment model](dmri/02_dmri_multicompartment_model.md): combine components into a mixture.
3. [Using pretrained models](dmri/03_pretrained_models.md): load a public checkpoint and inspect posteriors.
4. [Train a small model](dmri/04_train_standalone.md): train with a standalone loop.
5. [Getting example data](dmri/05_example_data.md): download data, prepare FSL inputs, and run prediction.

Sources are in
[examples/dmri/](https://github.com/mackelab/prism/tree/main/examples/dmri).
Notebook 05 includes a 174 MB download and full-volume prediction; its reading
page uses saved outputs, and it is not executed in normal CI.

All notebook pages, figures, and downloads are generated from `examples/`.
Edit the source notebook and follow the [rendering instructions](../development.md#documentation).
Scientific evaluation commands are in the figure recipes listed under
[paper workflows](../paper_experiments.md).
