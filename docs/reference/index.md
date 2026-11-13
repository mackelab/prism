# Reference

Use the reference pages for exact options, configuration keys, output files,
and Python signatures. The Python API pages are generated from the source
docstrings for the checked-out version of PRISM.

## User reference

- [Configuration](configuration.md): configuration files, Hydra groups, and
  command-line overrides.
- [Experiment presets](experiments.md): simulator, architecture, and training settings.
- [Paper workflows](../paper_experiments.md): figure recipes and protocol pages,
  required checkpoints, and available source results.
- [Outputs](outputs.md): prediction and evaluation output directories and files.
- [Troubleshooting](troubleshooting.md): installation, device, memory, and input
  problems.

## Command-line interface

- [`prism predict`](../cli/predict.md)
- [`prism train`](../cli/train.md)
- [`prism eval`](../cli/eval.md)

## Package layout

The method is the core `prism` package: `prism.nn` (model), `prism.train`,
`prism.inference`, `prism.simulators` (simulator interface and mask priors)
and `prism.config`. Applications build on it and are never imported by it:

- `prism.dmri`: diffusion MRI simulators, encoders and tokenizers, the
  volumetric `prism eval` pipeline and `prism predict`.
- `prism.symbolic`: additive symbolic simulators and, in `prism.symbolic.paper`,
  the frozen Figure 4b architecture used by the paper's symbolic experiments.

Saved runs that name classes by their former paths keep loading through
`prism.config.aliases`.

## Python API

- [Prediction and pretrained models](hub.md)
- [Simulators](simulators.md)
- [Neural networks](nn.md)
- [Training](train.md)
- [Evaluation](eval.md)
- [Utilities](utils.md)
