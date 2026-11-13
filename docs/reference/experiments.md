# Experiment presets

This catalog covers the entry presets in `conf/experiment/train/` and
`conf/experiment/eval/`. Select them with `+experiment/train=NAME` or
`+experiment/eval=NAME`, without the `.yaml` extension. Inspect the composed
settings before an expensive run:

```bash
uv run prism train +experiment/train=b3s_2_4_6_128 --cfg job --resolve
uv run prism eval +experiment/eval=eval_b3s_best_model_selection --cfg job --resolve
```

Training presets that run longer than an hour submit a Slurm job: the dMRI
presets use the `h100_long` partition (`2h100_long` for the two-GPU presets)
with a 70-hour training limit inside the 72-hour scheduler limit, and the
symbolic presets use `a100`. `symbolic` is the exception and runs locally.
The [training guide](../guides/training.md#choose-how-the-run-executes) shows
how to train in the current process, at another site, or without Slurm; the
[experiment commands](https://github.com/mackelab/prism#run-the-experiments)
list one command per task. Tracking is disabled and
`training.continue_training=false`; enable tracking and request continuation
explicitly. A preset's historical
`run.name` identifies an output/checkpoint directory; it does not select a
published checkpoint or prove the provenance of weights found there. Use a new
name for a new experiment. See [training](../guides/training.md) for resume paths
and [evaluation](../guides/evaluation.md) for checkpoint selection.

Below, **E/M/T** gives the number of observation-embedding, model-selection,
and parameter-inference layers; **width** is `model.model_dim`. A simulator's
**mask dimension** counts signal and noise components. Its **theta dimension**
is the full parameter vector, including inactive components, not the parameter
count of an individual selected model. Training stops when either its time or
step limit is reached; a time budget is not a guarantee of a particular step.

## dMRI training

All dMRI entry presets jointly train model selection and parameter inference.
They use a b-value/b-vector/signal embedding, an autoregressive mask decoder,
and a Simformer parameter network with `v` loss. All use the `multi` acquisition
group: HCP-style generators with 105 and 297 measurements. The respective
typical/random acquisition probabilities are 0.5/0.5 and 0.8/0.2.

Unless a difference is listed below, they use online `SimulationDataset` data,
CPU simulation in batches of 2,048, a 524,288-example buffer, training batches
of 2,048, RAdam at `1e-4`, no learning-rate scheduler, EMA decay 0.999, and a
70-hour limit with no step cap (72 hours originally; lowered so training stops
before the 72-hour scheduler limit). Checkpoints and validation are scheduled every
20,000 and 40,000 steps. Both loss weights are 1; loss masking is enabled,
complexity weighting and component permutation are disabled.

### Ball and sticks

| Preset | E/M/T; width | Purpose and distinguishing settings |
| --- | --- | --- |
| `b3s_2_4_6_64` | 2/4/6; 64 | Smaller-width shared-diffusivity Ball3Stick run; EMA 0.9999. |
| `b3s_2_4_6_128` | 2/4/6; 128 | Wider version of the same task; EMA 0.9999. |
| `msb3s_2_4_6_128` | 2/4/6; 128 | Multi-shell shared-diffusivity task; `posterior_score=true`; EMA 0.999. |

The first two select
`prism.dmri.simulators.models.Ball3StickSharedDiffusivity` (mask 5, theta 11);
the third selects `prism.dmri.simulators.models.MultiShellBall3StickSharedDiffusivity`
(mask 5, theta 12). Each has one ball, three sticks, and one bounded Gaussian
noise component, with fraction-prior concentrations `[3.5, 1, 0.3, 0.1]`.
The multi-shell model has two shared diffusivity parameters, compared with one
in the other model. Only the multi-shell preset generates posterior-score
targets.

All three use `prism.dmri.nn.model.DMRIModelConfigMaskPriorAmortizedPP`, including
`DMRITokenizerPP`, and the variable Beta–Bernoulli mask prior described below.
Their `run.name` equals the preset name. Earlier versions stored the named
architecture settings only in `hydra.sweeper.params`; those settings now apply
to both single runs and multiruns. Old saved single-run configs may therefore
contain 3/3/6 layers and width 128 despite their experiment filename. Check the
saved config when resuming or evaluating historical runs.

The frozen Figure 6 scripts instead reference `updated5_b3s_2_4_6_128`, whose
source training preset selects the **PPP** architecture. That exact training
preset is not included here. The current `b3s_2_4_6_128` preset is a PP recipe;
`msb3s_2_4_6_128` additionally selects a different simulator. Use a separately
identified paper recipe for retraining, and the saved configuration for loading
historical weights. Matching layer counts or similar run names do not establish
checkpoint compatibility. See [published weights](../paper_experiments.md#published-weights).

### Gaussian compartment collection

`all_3_4_8_128_model_prior` and
`updated_all_3_6_8_128_model_prior` select
`prism.dmri.simulators.models.AllGaussianModelsParamCountPrior`: one ball, three
sticks, three zeppelins, three diffusion tensors, and two alternative noise
components (bounded Gaussian and bounded Rician). The mask dimension is 12;
theta dimension is 51. Fractions have unit Dirichlet concentrations. Both use
`TotalParamPenalizedPrior`, and neither generates posterior-score targets.

| Setting | Original `all_…` | `updated_all_…` |
| --- | --- | --- |
| E/M/T; width | 3/4/8; 128 | 3/6/8; 128 |
| Model config suffix | `MaskPriorAmortizedPP` | `MaskPriorAmortizedPPP` |
| Mask-head output | Layer normalization, linear layer | No output normalization, MLP |
| RAdam learning rate | `1e-4` | `5e-4` |
| Adaptive gradient clipping value | 2 | 1 |
| `cut_off_tsm`; label smoothing | 0.1; 0.01 | 0; 0 |
| Validation KSD | Disabled | 1 iteration, 16 samples |

Both use `DMRIModelConfig` presets from `prism.dmri.nn.model`; the suffix change
selects `DMRITokenizerPPP` instead of `DMRITokenizerPP`. It is an architecture
change, so “updated” is not a checkpoint-compatible resume instruction.

The retained historical run names are
`all_3_4_8_128_model_prior_rician_fixed` and
`updated_all_3_6_8_128_model_prior_rician_fixed_continued`.

### Gaussian and convolved compartment collection

`all_3_6_8_128_model_prior_with_convs_2gpus` and
`updated_all_3_6_8_128_model_prior_with_convs_2gpus` select
`prism.dmri.simulators.models.AllGaussianAndConvolvedModels`.
They extend the ten Gaussian signal components with Dot, WatsonStick,
WatsonZeppelin, BinghamStick, BinghamZeppelin, NoddiB, NoddiW, SandiB, and SandiW.
With two alternative bounded Gaussian/Rician noise components, this gives
mask dimension 21 and theta dimension 114. Both use unit Dirichlet fraction
concentrations, `TotalParamPenalizedPrior`, no posterior-score targets, and
3/6/8 layers of width 128.

The original uses the PP config, normalized linear mask output, clipping value
2, `cut_off_tsm=0.1`, label smoothing 0.01, and no validation KSD. The updated
version uses PPP, an unnormalized MLP mask output, clipping value 1, zero
cutoff and smoothing, and the same validation KSD settings as the updated
Gaussian-only preset. **Both convolved presets retain learning rate `1e-4`.**

These presets use `datastream_multigpu`: GPU 1 generates simulations, while
GPU 0 handles training and validation. They require **two visible GPUs**, even
for a local run; the name does not imply training is split across two GPUs.
Both train with batches of 1,024. The original simulates batches of 4,096 into
a 786,432-example buffer; the updated version uses 2,048 and 4,194,304.

Their historical run names are
`all_3_6_8_128_model_prior_with_convs_2gpus_rician_fixed` and
`updated_all_3_6_8_128_model_prior_with_convs_2gpus_rician_fixed_continued`.
For both updated dMRI presets, `scheduler_params.transition_steps=2000000`
is stored but inactive because `scheduler=null`. The `_rician_fixed` and
`_continued` suffixes are historical labels: both noise alternatives remain
in the simulator, and continuation must be enabled explicitly.

### What the dMRI model priors mean

`BetaBernoulliMaskPrior` draws a shared `p ~ Beta(1, 1)`, draws signal bits
with Bernoulli probability `p`, then forces one uniformly chosen signal
component active. This is a forced-component mixture, rather than rejection
sampling conditioned on a nonempty mask. It selects exactly one noise
component uniformly. The model receives the one-dimensional prior input `p`.

`TotalParamPenalizedPrior` draws `u ~ Beta(1, 1)`. Its actual default is
**linear** penalty mode, with base inclusion probability 0.5 and maximum
logit penalty 5. For signal component size `s_i`, inclusion logit is
`-5 * (1-u) * s_i / max(s)`: larger components are penalized more, and `u=1`
removes the penalty. Here `s_i` counts that signal compartment's theta
parameters, excluding mixture fractions and noise parameters. No signal is
forced active. Exactly one noise alternative is selected uniformly. The
one-dimensional prior input is `u`, so an evaluation key called `p_mask`
means `u` for these simulators, not a shared inclusion probability.

## Symbolic training

### Small working example

`symbolic` uses `prism.symbolic.simulators.SimplePolynomial`: linear and
quadratic terms plus Gaussian noise (mask 3, theta 3), with the same variable
Beta–Bernoulli prior. It selects
`PrismModelConfigMaskPriorAmortized`, the generic `CompartmentTokenizer`,
and 1/1/1 layers of width 16. Observations contain coordinate/signal pairs at
32 points in `[-1, 1]`. This is a small training example: fixed datasets of
128 training and 32 validation examples, batches of 8, AdamW at `1e-4`,
EMA 0.999, a 100-step cap (also inheriting 72 hours), and checkpoint/validation
every 25 steps. Its run name is `symbolic`.

### Fixed empirical-prior family

The `symbolic_sbmi` family uses
`prism.symbolic.paper.model_specs.SymbolicModelV0`: 15 functions
and five noise components (mask 20, theta 28). `SymbolicMaskPrior` samples
uniformly from one million stored 20-bit rows; repeated rows retain their
sampling multiplicity. It has no prior-hyperparameter input. The model is
`prism.symbolic.paper.model.SymbolicPaperModelConfig`, with
`prism.symbolic.paper.tokenizer.SymbolicPaperTokenizerPP` and width 64. The 10k, 25k and 50k
points use 2/4/4 layers, the 100k point 2/6/4 and 200k/500k/1000k 2/6/6 (the
layer counts follow the stored sweep configurations).

| Preset / run name | Fixed training examples | Step cap |
| --- | ---: | ---: |
| `symbolic_sbmi` | 10,000 | None |
| `symbolic_sbmi_10k` | 10,000 | 50,000 |
| `symbolic_sbmi_25k` | 25,000 | 100,000 |
| `symbolic_sbmi_50k` | 50,000 | 200,000 |
| `symbolic_sbmi_100k` | 100,000 | 500,000 |
| `symbolic_sbmi_200k` | 200,000 | None (8 training hours) |
| `symbolic_sbmi_500k` | 500,000 | None (8 training hours) |
| `symbolic_sbmi_1000k` | 1,000,000 | None (8 training hours) |

All have seed 0, 1,024 fixed validation examples, training batches of 128,
RAdam at `1e-4`, EMA 0.999, and checkpoint/validation intervals of 2,000 steps. The
internal time ceiling is 72 hours through 50k, 24 hours for 100k, and 8 hours
for the last three points; the step caps stop the three smallest points first.
Scheduler limits are 700 minutes, and 1,560 minutes for 50k and 100k: 200k
steps take about 10 hours, and the 100k preset's 24-hour ceiling must be
reachable. Model-selection loss has a 15,000-step warmup and
15,000-step ramp through 100k; both are zero for the last three points.
Label smoothing and KSD validation are disabled. The terminal checkpoint
is saved without a final validation pass. These are separate dataset-size and
training-budget presets, not a chain of resumed runs.

The last three retain 100 regular checkpoints; other presets retain three.
The original nested-config bug ignored its requested retention of 100.
The original eight-hour H100 allocation and 72-hour internal ceiling are
replaced with a graceful eight-hour training-loop stop for those runs. This
retains the existing A100 reproduction policy, not an equal compute budget:
eight A100 hours are not compute-equivalent to eight H100 hours.
The [Figure 4b page](../experiments/fig4b.md) documents completion evidence
and source hashes.

This family implements the Figure 4b setup; no checkpoint is selected by
these presets.

### Variable-prior scaling family

`symbolic15`, `symbolic30`, `symbolic50`, `symbolic80` and `symbolic100` use the symbolic
`SymbolicPaperModelConfigMaskPriorAmortized` class with the same symbolic PP
tokenizer, **2/6/6 layers and width 128**. The mask decoder takes a scalar
prior input and enables `precondition_with_mask_prior`. All use the variable
Beta–Bernoulli prior (`alpha=beta=1`, one signal forced active), rather than
the fixed empirical prior above.

| Preset / run name | Simulator class in `…symbolic.model_specs` | Mask / theta dimensions |
| --- | --- | --- |
| `symbolic15` | `SymbolicModelV15`: 15 functions, 5 noises | 20 / 28 |
| `symbolic30` | `SymbolicModelV30`: 30 functions, 8 noises | 38 / 73 |
| `symbolic50` | `SymbolicModelV50`: 50 functions, 8 noises | 58 / 115 |
| `symbolic80` | `SymbolicModelV80`: 80 functions, 8 noises | 88 / 187 |
| `symbolic100` | `SymbolicModelV100`: 100 functions, 8 noises | 108 / 223 |

These use online simulations with batches of 8,192, a 2,097,152-example buffer,
training batches of 4,096, RAdam at `5e-4`, EMA **0.99**, seed 0, a
**24-hour** cap and no step cap. They request 1,560 minutes of scheduler time so
the 24-hour training limit is reached. There is no model-selection warmup or ramp. Checkpoints occur
every 2,000 steps and validation every 4,000; terminal validation and KSD are
disabled. `symbolic15`, `symbolic30`, `symbolic80` and `symbolic100` inherit `symbolic50` and replace the
simulator and run name. `symbolic80` and `symbolic100` start training fresh; the original runs' continuation
is not reproduced. The expression
palette repeats cyclically up to K, as in the original. For every K in 15, 30, 50, 80 and 100,
`symbolic<K>_small` and `symbolic<K>_tiny` set the model width to 64 and 32. Preset names count signal functions,
excluding noise.

`symbolic50_illustrative` is the run behind Figures 2 and 4a: the K=50 model
observed at ten inputs (`x = 0, 1, …, 9`, simulator `symbolic_50_illustrative`),
seed 1, training batches of 8,192, a 72-hour cap and 4,320 minutes of scheduler
time. The figures use its step 1,396,000, published as
`symbolic50_illustrative/figure`.

Both larger symbolic families observe 500 signal values on `[0, 10)` with
spacing 0.02, arranged as 50 blocks of 10. They do not use dMRI acquisitions.
The fixed and variable prior families
have different model configurations and scientific settings; sharing symbolic
components does not make their checkpoints interchangeable.

## Evaluation presets

Evaluation presets load the simulator and network architecture from the
selected checkpoint; they do not define a new training model. None selects
a particular checkpoint. They default to EMA weights from the latest regular
checkpoint, float32 evaluation, and 50 theta samples per voxel. There is no
evaluation wall-time cap in these presets. File inputs and checkpoint paths
must be supplied for a useful run.

The Ball3Stick masks below follow `[ball, stick1, stick2, stick3, noise]`.
The five-bit masks and Ball3Stick exporters require a compatible simulator;
these are not ready-made evaluation presets for the larger dMRI collections
or the symbolic families.

### Model-selection comparisons

All three use file input, the default theta sampler (40 ODE steps plus the
final Euler step), and the `auto` corrector. `auto` selects SMC for a fixed
model per voxel and MCMC when models vary between posterior draws. They export
Ball3Stick parameter maps and model-selection results.

| Preset | Model treatment |
| --- | --- |
| `eval_b3s_average_model_selection` | Draws 50 masks at `p_mask=0.35`; parameter draws use individual sampled masks. KSD export is disabled. |
| `eval_b3s_best_model_selection` | Uses `p_mask=0.35`; scores B1S/B2S/B3S and conditions theta on the best mask per voxel. Also draws mask samples for export. |
| `eval_b3s_no_selection` | Disables mask sampling and applies no selection transform; with file input and no explicit default mask, theta uses all components (B3S). |

“Average” means averaging over sampled models through parameter draws; it does
not round average mask bits into one model. The corresponding theta export
directory names end in `_average_model_selection`, `_best_model_selection`,
and `_no_selection`. These presets retain the inherited feasible-model
probability export, including when mask sampling is disabled.

### Fixed-model reference and diagnostic runs

These presets use seed `42424242`, disable mask sampling and model selection,
and export raw theta samples. File-input presets select spatial slice `z=52`;
`eval_test_syn` evaluates all 512 generated voxels without a spatial slice.
“Ground truth” is a historical preset label for a costly reference-sampling
configuration, not a claim of exact posterior samples or known parameters for
real data.

| Preset | Conditioning and sampling |
| --- | --- |
| `eval_ground_truth_b1s` | Fixed `[true,true,false,false,true]`; `gt` sampler. |
| `eval_ground_truth_b2s` | Fixed `[true,true,true,false,true]`; `gt` sampler. |
| `eval_ground_truth_b3s` | Fixed all-true B3S mask; `gt` sampler. |
| `eval_ground_truth_b3s_only_smc` | Fixed all-true B3S mask; zero network ODE steps and SMC from scratch. |
| `eval_test` | Same B3S mask and `gt` sampler on real slice 52; theta export name `test_real`. |
| `eval_test_syn` | 512 synthetic voxels, HCP acquisition, `gt` sampler; generated true masks override the configured B3S fallback. Theta export name `test`. |

`gt` uses 64 ODE steps, theta batches of 1,024, and `smc_high`: start
`lambda=0.9`, 10 correction steps, 2 inner steps, 10 integration steps, and
step size 0.002. `only_smc` uses theta batches of 3,841 and
`smc_from_scratch`: start `lambda=0`, 100 correction steps, 3 inner steps,
10 integration steps, and step size 0.002. Both produce 50 theta samples.
The presets named `eval_test` and `eval_test_syn` use `gt`.

The four ground-truth theta export names match their preset names without
`eval_`, with `inference_results` inserted before `b1s`, `b2s`, or `b3s`.
Raw export includes reconstruction error, posterior NLL, KSD, and SBC metrics;
SBC needs synthetic truth. The inherited model-selection exporter may still
compute feasible B1S/B2S/B3S probabilities even with both selection stages
disabled. See the [evaluation guide](../guides/evaluation.md) for metric
prerequisites and overrides.
