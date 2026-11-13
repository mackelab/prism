# Evaluation CLI

`prism eval` runs the Hydra-configured evaluation pipeline. It can read a local
training run, a local portable checkpoint bundle, or a checkpoint bundle from
Hugging Face; process real or generated data; sample model masks and parameters;
compute configured metrics; and export NIfTI and JSON files.

Use [`prism predict`](predict.md) for a simpler command that validates a standard
input folder, downloads a pretrained model by default, creates an output folder,
and disables evaluation metrics. Use `prism eval` when you need synthetic data,
metrics, non-default exporters, or direct Hydra overrides.

## Synopsis

```bash
uv run prism eval [HYDRA_OVERRIDES...]
```

Useful inspection commands:

```bash
uv run prism eval --help
uv run prism eval +experiment/eval=eval_b3s_best_model_selection --cfg job --resolve
```

## Minimal Real-Data Run

```bash
uv run prism eval +experiment/eval=eval_b3s_best_model_selection \
  checkpoint.model_name=my_run checkpoint.which=best \
  evaluation.input.path=/data/sub-001/dwi
```

The input directory must contain `data.nii.gz`, `nodif_brain_mask.nii.gz`,
`bvals`, and `bvecs`, unless their names are overridden under
`evaluation.input`.

## Common Overrides

| Purpose | Override |
| --- | --- |
| Local run below `results/` | `checkpoint.model_name=<run>` (newest job), or `<run>/<timestamp>/<job>` for one multirun job |
| Local portable bundle | `checkpoint.path=<parent> checkpoint.model_name=<bundle-name>` |
| Hugging Face bundle | `checkpoint.pretrained.repo_id=<owner/repo> checkpoint.model_name=<name>` |
| Best or latest checkpoint | `checkpoint.which=best` or `checkpoint.which=latest` (default `latest`; the Hub `b3s_*` bundles have only `best`) |
| Direct input directory | `evaluation.input.path=/absolute/path` |
| Input below a data root | `evaluation.input.data_root=/data evaluation.input.data_folder=sub-001/dwi` |
| Synthetic input | `evaluation/input=synthetic` |
| Parameter samples | `evaluation.sampling.theta.num_samples=50` |
| Mask samples | `evaluation.sampling.mask.n_samples=50` |
| Corrector | `evaluation/theta/corrector@evaluation.sampling.theta.corrector=auto` |
| Precision | `evaluation.precision=fp32` |
| Fixed batch size | `evaluation.batch_size=1024` |
| Explicit export root | `run.output_dir=/absolute/output` |
| Disable theta metrics | `~evaluation.export.theta.metrics` |

`evaluation.input.path` takes precedence over `data_folder`. Without `path`,
the input is `<data_root>/<data_folder>`; `data_root` defaults to `data/` below
the directory from which the command was started.

## Outputs

By default, exports are written below the selected checkpoint location. For a
normal run or a Hub checkpoint, that is:

```text
<checkpoint.results_root>/<checkpoint.results_folder>/<checkpoint.model_name>/
```

The optional `results_folder` segment is omitted when it is `null`. With
`checkpoint.path=<parent>`, the default is instead
`<parent>/<checkpoint.model_name>/`.

For example, the best-model preset writes parameter maps to
`ball3stick_inference_results_best_model_selection/` and model-selection maps to
`ball3stick_model_selection_results/`. Set `run.output_dir` to put all export
subdirectories elsewhere. `hydra.run.dir` controls Hydra's own run metadata; it
does not control these exports.

See the [evaluation guide](../guides/evaluation.md) for checkpoint layouts,
pipeline modes, preprocessing, metrics, and hardware settings.
