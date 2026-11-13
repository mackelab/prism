# Troubleshooting

## Prediction is running on CPU

`prism predict --verbose` logs the selected JAX devices (`Eval devices: ...`); a
CPU run also prints a warning without it. CPU prediction can be very slow
even for a few thousand voxels. Install the appropriate accelerated JAX build
and confirm that `jax.devices()` lists the expected GPU or TPU.

On CPU, requested fp16 or bf16 network precision falls back to fp32 because
half-precision dot products with fp32 accumulation are not reliable there.

## First run is slow

JAX compiles model operations on first use. Automatic batch sizing also probes
each evaluation stage. Compilation and batch-size results are cached under
`.jax_cache/` by default; later runs with the same model, device, and sampling
settings can reuse them.

Set `JAX_COMPILATION_CACHE_DIR` before starting Python to choose another cache
directory. A path set through `jax.config.update("jax_compilation_cache_dir", path)`
before importing the evaluation module is also preserved. Both caches use the
selected directory.

## Out of memory

- Close other accelerator processes.
- Lower `--memory-fraction` before JAX initializes.
- Set a smaller `--batch-size` for prediction.
- For evaluation, set `evaluation.batch_size` or stage-specific
  `eval_batch_size` values.
- Reduce theta samples or ODE steps when scientifically acceptable.

Evaluation halves a batch and retries after recognized runtime OOM errors.

## Input folder is rejected

Prediction requires these exact default names:

```text
data.nii.gz
nodif_brain_mask.nii.gz
bvals
bvecs
```

Check that:

- the mask and image spatial shapes agree;
- the image measurement dimension matches b-values and b-vectors;
- the mask contains in-brain voxels;
- the acquisition includes a b=0 shell;
- b-vectors have the expected orientation and shape.

`prism eval` permits filename overrides under `evaluation.input`.

## Output directory already exists

Prediction refuses to replace an existing output directory. Pass `--overwrite`
only after confirming that the existing result can be removed, or choose another
`--output-subdir`.

## Hub download or authentication fails

- Run `uv run hf auth login` for private repositories.
- Check `--repo-id`, model name, and revision.
- Use `--cache-dir` when the default cache is unavailable.
- Use `--local-files-only` only after all required files are cached.
- Use `--local-checkpoint` with a portable bundle to avoid Hub access.

## Checkpoint cannot be restored

Confirm whether the requested checkpoint is `latest`, `best`, or a numeric
regular step. A run may not contain `best/` if validation-selected checkpointing
was disabled or no best checkpoint was saved.

For portable bundles, `config.yaml` and `checkpoints/` must be inside the model
directory; `artifact.yaml` is used when present.

## Training did not resume

Resume reads `results/<run.name>/checkpoints/`, so repeat the original
`run.name` with `training.continue_training=true`; the new command gets its own
timestamped log directory. For a multirun, also point `hydra.sweep.dir` at the
original sweep. Resume uses the latest regular checkpoint, not `best/`. Without
`continue_training`, a run whose checkpoint directory already holds checkpoints
stops at start-up.

Simulation buffers and loader position are not checkpointed.

## A documented command does not compose

Print the config first:

```bash
prism train --cfg job --resolve ...
prism eval --cfg job --resolve ...
```

Remember that experiment groups need a leading `+`, list/removal overrides
should be quoted in a shell, and sweep parameters require `--multirun`.

## Results contain non-finite samples

Rerun with `fp32`, check the input normalization and acquisition compatibility,
and compare with an uncorrected run. The evaluator reports the affected fraction
when it detects non-finite theta samples.

## Acquisition compatibility

A checkpoint can load successfully and still be inappropriate for the input
acquisition. Confirm shell distribution, b-values, direction distribution,
signal normalization, and any model-specific acquisition assumptions before
interpreting output.
