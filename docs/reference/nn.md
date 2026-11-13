# Neural Nets API

The method's network building blocks live in `prism.nn`: `PrismModel`
combines an observation encoder, an autoregressive model-selection decoder and
a diffusion parameter-inference head. Applications contribute encoders,
tokenizers and configuration presets: `prism.dmri.nn` for diffusion MRI and
`prism.symbolic.paper` for the frozen Figure 4b architecture.
See the [symbolic guide](../guides/symbolic.md) for configuration and inference.

For a small training and inference example, see the
[minimal symbolic notebook](../examples/symbolic/minimal.md) or the
[standalone dMRI training notebook](../examples/dmri/04_train_standalone.md). The
[training guide](../guides/training.md) covers checkpointed runs.

## PRISM model and selection

::: prism.nn.model
    options:
      show_root_heading: true
      show_root_full_path: false

::: prism.nn.autoregressive
    options:
      show_root_heading: true
      show_root_full_path: false

## Embeddings and tokenization

`TransformerEmbedding` encodes observations given as token sequences. The
`CompartmentTokenizer` embeds model masks and parameters for any simulator
that implements the [simulator interface](simulators.md#simulator-interface).

::: prism.nn.embedding
    options:
      show_root_heading: true
      show_root_full_path: false

::: prism.nn.tokenizer
    options:
      show_root_heading: true
      show_root_full_path: false

::: prism.nn.compartment_tokenizer
    options:
      show_root_heading: true
      show_root_full_path: false

## Transformer backbones

::: prism.nn.simformer
    options:
      show_root_heading: true
      show_root_full_path: false

## Diffusion MRI

`DMRIModelConfig*` presets select the dMRI tokenizers and the b-value/b-vector
signal encoder; the model class is still `PrismModel`.

::: prism.dmri.nn.model
    options:
      show_root_heading: true
      show_root_full_path: false

::: prism.dmri.nn.embedding
    options:
      show_root_heading: true
      show_root_full_path: false

::: prism.dmri.nn.tokenizer
    options:
      show_root_heading: true
      show_root_full_path: false

## Symbolic paper architecture

The figure presets use `SymbolicPaperModel` with either a fixed-prior or a
prior-amortized configuration. The latter supplies `mask_prior` to mask
sampling and scoring; parameter inference takes the selected `model_mask`.
These modules reproduce the paper's parameter trees and initialization; new
models should use the core components above.

::: prism.symbolic.paper.model
    options:
      show_root_heading: true
      show_root_full_path: false

::: prism.symbolic.paper.autoregressive
    options:
      show_root_heading: true
      show_root_full_path: false

Figure presets use `SymbolicPaperTokenizerPP`, which retains the source's
shifted mask embeddings and initialization conventions.

::: prism.symbolic.paper.tokenizer
    options:
      show_root_heading: true
      show_root_full_path: false

::: prism.symbolic.paper.embedding
    options:
      show_root_heading: true
      show_root_full_path: false
