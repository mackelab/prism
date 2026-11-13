# Simulators API

Exact signatures for acquisition schemes, compartments, mixtures, priors, and
noise models. Read the [dMRI simulator guide](../guides/simulators.md) or the
[symbolic guide](../guides/symbolic.md) for the corresponding workflow.

## Acquisition schemes

::: prism.dmri.simulators.acquisition_scheme.acquisition_scheme
    options:
      show_root_heading: true
      show_root_full_path: false
      separate_signature: true

::: prism.dmri.simulators.acquisition_scheme.ssfp_acquisition_scheme
    options:
      show_root_heading: true
      show_root_full_path: false
      separate_signature: true

## Simulator interface

PRISM's training and inference code relies only on these members of a
simulator class. `resolve_simulator_model` checks them when a configuration
names a simulator. The dMRI `MultiCompartment` and symbolic `AdditiveModel`
families implement them, including `training_generators`, which builds the
application's training samples.

::: prism.simulators.base.Simulator
    options:
      show_root_heading: true
      show_root_full_path: false

## Abstract simulator classes

`create_from_theta` constructs a compartment by calling its `from_theta`
implementation. Existing `from_theta` calls remain supported.

::: prism.simulators.base.Compartment
    options:
      show_root_heading: true
      show_root_full_path: false
      members:
        - to_theta
        - to_params
        - create_from_theta
        - from_theta
        - tree_flatten
        - tree_unflatten

::: prism.dmri.simulators.base.SignalCompartment
    options:
      show_root_heading: true
      show_root_full_path: false
      members:
        - signal_fn
        - log_signal_fn
        - signal
        - log_signal
        - fit
        - to_fod

::: prism.dmri.simulators.base.NoiseCompartment
    options:
      show_root_heading: true
      show_root_full_path: false
      members:
        - noise
        - log_likelihood


::: prism.dmri.simulators.local_signal_models.signal_response_kernels
    options:
        show_root_heading: true
        show_root_full_path: false


::: prism.dmri.simulators.multi_compartment.MultiCompartment
    options:
      show_root_heading: true
      show_root_full_path: false
      filters:
        - "^signal_fn$"
        - "^create_from_theta$"
        - "^from_theta$"
        - "^to_theta$"
        - "^num_compartments$"
        - "^get_all_params$"

## Local signal models

A selection of voxel-wise dMRI signal compartments is implemented below.

### Isotropic compartments

::: prism.dmri.simulators.local_signal_models.ball.Ball
    options:
        show_root_heading: true
        show_root_full_path: false

::: prism.dmri.simulators.local_signal_models.ball.StaticBall
    options:
        show_root_heading: true
        show_root_full_path: false

::: prism.dmri.simulators.local_signal_models.ball.MultiShellBall
    options:
        show_root_heading: true
        show_root_full_path: false

::: prism.dmri.simulators.local_signal_models.ball.MultiShellStaticBall
    options:
        show_root_heading: true
        show_root_full_path: false

::: prism.dmri.simulators.local_signal_models.ball.SSFPBall
    options:
        show_root_heading: true
        show_root_full_path: false

::: prism.dmri.simulators.local_signal_models.ball.SSFPStaticBall
    options:
        show_root_heading: true
        show_root_full_path: false


::: prism.dmri.simulators.local_signal_models.sphere.Sphere
    options:
        show_root_heading: true
        show_root_full_path: false

### Anisotropic compartments

::: prism.dmri.simulators.local_signal_models.zeppelin.Zeppelin
    options:
        show_root_heading: true
        show_root_full_path: false

::: prism.dmri.simulators.local_signal_models.zeppelin.StaticZeppelin
    options:
        show_root_heading: true
        show_root_full_path: false


::: prism.dmri.simulators.local_signal_models.cylinder.Cylinder
    options:
        show_root_heading: true
        show_root_full_path: false

::: prism.dmri.simulators.local_signal_models.convolved_zeppelin.WatsonZeppelin
    options:
        show_root_heading: true
        show_root_full_path: false

::: prism.dmri.simulators.local_signal_models.convolved_zeppelin.BinghamZeppelin
    options:
        show_root_heading: true
        show_root_full_path: false

::: prism.dmri.simulators.local_signal_models.stick.Stick
    options:
        show_root_heading: true
        show_root_full_path: false

::: prism.dmri.simulators.local_signal_models.stick.StaticStick
    options:
        show_root_heading: true
        show_root_full_path: false

::: prism.dmri.simulators.local_signal_models.stick.MultiShellStick
    options:
        show_root_heading: true
        show_root_full_path: false

::: prism.dmri.simulators.local_signal_models.stick.MultiShellStaticStick
    options:
        show_root_heading: true
        show_root_full_path: false

::: prism.dmri.simulators.local_signal_models.stick.SSFPStick
    options:
        show_root_heading: true
        show_root_full_path: false

::: prism.dmri.simulators.local_signal_models.stick.SSFPStaticStick
    options:
        show_root_heading: true
        show_root_full_path: false

::: prism.dmri.simulators.local_signal_models.convolved_stick.WatsonStick
    options:
        show_root_heading: true
        show_root_full_path: false

::: prism.dmri.simulators.local_signal_models.convolved_stick.BinghamStick
    options:
        show_root_heading: true
        show_root_full_path: false

::: prism.dmri.simulators.local_signal_models.dot.Dot
    options:
        show_root_heading: true
        show_root_full_path: false

::: prism.dmri.simulators.local_signal_models.dti.Dti
    options:
        show_root_heading: true
        show_root_full_path: false

### Spherical convolution based simulators

These compartments convolve a kernel with a fiber orientation distribution (FOD)
on the sphere. The shared base classes live in `prism.dmri.simulators.convolved_models`.

::: prism.dmri.simulators.convolved_models
    options:
        show_root_heading: true
        show_root_full_path: false

::: prism.dmri.simulators.local_signal_models.noddi.NoddiW
    options:
        show_root_heading: true
        show_root_full_path: false

::: prism.dmri.simulators.local_signal_models.noddi.NoddiB
    options:
        show_root_heading: true
        show_root_full_path: false

::: prism.dmri.simulators.local_signal_models.sandi.SandiW
    options:
        show_root_heading: true
        show_root_full_path: false

::: prism.dmri.simulators.local_signal_models.sandi.SandiB
    options:
        show_root_heading: true
        show_root_full_path: false

## Spherical distributions

Orientation distributions (FODs, Watson, Bingham, ...) used by convolved
compartments and as orientation priors.

::: prism.dmri.simulators.spherical_distributions
    options:
        show_root_heading: true
        show_root_full_path: false

## Predefined model collections

Ready-made `MultiCompartment` families mirroring common literature mixtures
(Ball–Stick, Ball–3-Stick, multi-shell variants, ...).

::: prism.dmri.simulators.models
    options:
        show_root_heading: true
        show_root_full_path: false

## Noise models

::: prism.dmri.simulators.noise_compartments
    options:
      show_root_heading: true
      show_root_full_path: false

## Mask priors

::: prism.simulators.mask_prior
    options:
      show_root_heading: true
      show_root_full_path: false

## Symbolic models

`AdditiveModel` combines selected expression components and noise alternatives.
Its mask orders expression bits before noise bits, and its full `theta` vector
uses unconstrained standard-normal coordinates for all component parameters.
Calling `evaluate(grid)` gives the noise-free signal; supplying `rng` adds
the selected noise.

::: prism.symbolic.simulators.base.AdditiveModel
    options:
      show_root_heading: true
      show_root_full_path: false

::: prism.symbolic.simulators.model.SymbolicExpressionCompartment
    options:
      show_root_heading: true
      show_root_full_path: false

::: prism.symbolic.simulators.model.SymbolicNoiseCompartment
    options:
      show_root_heading: true
      show_root_full_path: false

### Preset simulators

`SimplePolynomial` is the small API example. `SymbolicModelV0` uses the fixed
empirical Figure 4b prior; `SymbolicModelV15`, `V30`, `V50`, `V80` and
`V100` use the Beta-Bernoulli prior for the scaling experiments. See the
[experiment catalog](experiments.md) for their configurations.

::: prism.symbolic.simulators.SimplePolynomial
    options:
      show_root_heading: true
      show_root_full_path: false

::: prism.symbolic.paper.model_specs
    options:
      show_root_heading: true
      show_root_full_path: false
      members:
        - SymbolicModelV0
        - SymbolicModelV15
        - SymbolicModelV30
        - SymbolicModelV50
        - SymbolicModelV80
        - SymbolicModelV100

### Empirical mask prior

Sampling preserves the order and multiplicity of the packaged mask-table rows.
The legacy lookup score assigns `1/N` to a supported mask without aggregating
duplicate rows; Figure 4b uses this prior for sampling.

::: prism.symbolic.paper.mask_prior.SymbolicMaskPrior
    options:
      show_root_heading: true
      show_root_full_path: false
