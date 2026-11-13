# Prediction

<video controls muted loop playsinline preload="none"
       poster="../../assets/predict-demo-poster.jpg"
       style="width: 100%; border-radius: 4px;">
  <source src="../assets/predict-demo.mp4" type="video/mp4">
  <a href="../assets/predict-demo.mp4">Download the recording</a>.
</video>

*An interactive run: model, quality, model mode, and fixed model are selected,
then sampling writes NIfTI maps.*

`prism predict` applies a pretrained checkpoint to diffusion MRI data in the
standard FSL folder layout. It downloads or opens a checkpoint, samples model
masks and parameters, and writes NIfTI maps. Use [`prism eval`](evaluation.md)
when you need custom Hydra configuration, synthetic data, or metrics.

## Input

The input folder must contain:

```text
FOLDER/
|-- data.nii.gz
|-- nodif_brain_mask.nii.gz
|-- bvals
`-- bvecs
```

The image and mask dimensions must agree. The signal volume must have one
measurement per b-value and b-vector.

See [Getting example data](../examples/dmri/05_example_data.md) for download
and input-preparation commands.

> **Check acquisition compatibility.** A checkpoint was trained for a
> particular acquisition distribution. Matching filenames and image dimensions
> does not establish compatibility.

## Run prediction

For an interactive terminal:

```bash
uv run prism predict FOLDER
```

The selector asks for a model, quality, model mode, and a fixed model when that
mode is selected. Use arrow keys or `j`/`k`, then Enter. Number keys select an
entry directly. Command-line values are not prompted again.

For scripts:

```bash
uv run prism predict FOLDER \
  --model msb3s_2_4_6_128 \
  --quality balanced \
  --model-mode best \
  --non-interactive
```

The default repository is `manugloeck/dmri-pretrained`, and the default model is
`msb3s_2_4_6_128`. Hugging Face downloads are cached locally.

## Quality

| Quality | ODE steps | Samples | Network precision | Corrector |
| --- | ---: | ---: | --- | --- |
| `very-fast` | 8 | 10 | fp16 on supported accelerators | none |
| `fast` | 20 | 25 | fp16 on supported accelerators | none |
| `balanced` | 40 | 50 | fp32 | auto |
| `high` | 60 | 100 | fp32 | auto |

An explicit `--num-steps`, `--theta-samples`, `--precision`, or `--corrector`
overrides the selected preset. Runtime grows approximately with voxels x theta
samples x network evaluations. Validate lower-cost settings on representative
data before using them for a study.

On CPU, fp16 and bf16 fall back to fp32. CPU prediction is supported but can be very
slow even for a few thousand voxels. Use a supported GPU accelerator for normal
volume-sized inputs.

## Model mode

`per-sample` retains model uncertainty. Each parameter sample is conditioned on
a sampled model mask:

```bash
uv run prism predict FOLDER --model-mode per-sample
```

`best` selects one highest-probability feasible model per voxel:

```bash
uv run prism predict FOLDER --model-mode best
```

`fixed` uses one Ball-and-Stick model throughout the mask:

```bash
uv run prism predict FOLDER --fixed-model B2S
```

The current Ball3Stick checkpoints support `B1S`, `B2S`, and `B3S`. These names
are checkpoint-family specific.

In `per-sample` mode, `--mask-samples` must be at least
`--theta-samples`.

## Checkpoint source

List models or choose another Hub revision:

```bash
uv run prism predict --list-models
uv run prism predict FOLDER --model MODEL --repo-id OWNER/REPOSITORY
uv run prism predict FOLDER --revision COMMIT_OR_TAG
```

Use cached files without a network request:

```bash
uv run prism predict FOLDER --local-files-only
```

Use a local portable bundle:

```bash
uv run prism predict FOLDER --local-checkpoint /models/my_model
```

See [Checkpoints and outputs](../reference/outputs.md) for the bundle layout.

Prediction uses EMA weights (`params_ema`) by default for both local and
Hugging Face checkpoints. To select raw weights explicitly:

```bash
uv run prism predict FOLDER --set checkpoint.params_name=params
```

Only `params_ema` and `params` are accepted; missing requested weights raise
`ValueError`. Use `--verbose` to see the selected weights and restored step in
the evaluation logs.

## Output

The default output is `FOLDER/dmri_output/`:

```text
dmri_output/
|-- .hydra/                                   run configuration
|-- ball3stick_inference_results/
|-- ball3stick_model_selection_results/       not in fixed mode
|-- ball3stick_model_selection_mask_samples/  per-sample mode only
`-- view_results.html
```

The HTML viewer contains selected scalar maps. It is written after successful
inference; use `--no-viewer` to skip it. The NIfTI files remain the complete
output and can be opened in another viewer.

Common inference maps include:

- `mean_f0samples.nii.gz`: mean isotropic fraction.
- `mean_f1samples.nii.gz` through `mean_f3samples.nii.gz`: mean stick fractions.
- `mean_fsumsamples.nii.gz`: mean total anisotropic fraction, conditioned on
  samples that retain the ball when that export option is enabled.
- `mean_fsumsamples_all.nii.gz`: unconditioned mean total anisotropic fraction.
- `frac_ball_active.nii.gz`: fraction of posterior masks containing the ball.
- `mean_dsamples.nii.gz`: mean diffusivity summary.
- `mean_num_fib_predsamples.nii.gz`: mean predicted number of fibers.

Outside-mask voxels are zero. See [Checkpoints and outputs](../reference/outputs.md)
for directories and naming conventions.

Choose a different subdirectory or replace an earlier result:

```bash
uv run prism predict FOLDER --output-subdir analysis --overwrite
```

## Batching and logs

If `--batch-size` is omitted, PRISM probes stage-specific batch sizes and caches
them in `.jax_cache/prism_batch_size.json` by default. Set
`JAX_COMPILATION_CACHE_DIR` to relocate both compilation and batch-size caches.
A first run on a device includes compilation and probing time. Later runs can
reuse both batch and JAX
compilation caches. Runtime out-of-memory errors reduce the current batch and
retry it.

Use `--batch-size` to set a fixed voxel batch. Use `--memory-fraction` before
backend initialization when sharing a GPU. All visible supported accelerators
are used.

Normal output shows a summary and progress. `--verbose` adds evaluation logs.
Set `PRISM_SHOW_NATIVE_LOGS=1` only when debugging JAX/XLA diagnostics.

## Research use

PRISM has not been clinically validated and must not be used for clinical
decisions.
