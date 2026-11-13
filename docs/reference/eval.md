# Evaluation Python API

`prism eval` is the diffusion MRI volumetric evaluation pipeline
(`prism.dmri.eval`). It builds on the method-level inference helpers in
`prism.inference`: batched sampling and correction, model selection,
precision, batch-size tuning and the kernel Stein discrepancy.

Full evaluation is CLI-driven. `prism eval` composes the Hydra configuration,
loads the checkpoint and input, chooses devices and batch sizes, runs the
pipeline, and dispatches exporters. There is no supported Python function that
accepts an eval config and replaces the CLI entry point; use subprocess or call
the `prism` command for a complete run.

The package-level `prism.dmri.eval` API exposes the metric configuration types and
metric runner. The additional functions below are selected module-level helpers
for code that already owns the required arrays, model, or configuration.

## Configured Metrics

`run_configured_metrics` evaluates a metric set after inference. Callers must
construct a complete `MetricContext`, including spatial metadata used to embed
voxel values into NIfTI volumes. The CLI normally does this work.

::: prism.dmri.eval.export_metrics
    options:
      show_root_heading: true
      show_root_full_path: false
      members:
        - MetricAggregation
        - MetricSpec
        - MetricResult
        - MetricContext
        - run_configured_metrics

These names can also be imported directly from `prism.dmri.eval`:

```python
from prism.dmri.eval import MetricContext, MetricSpec, run_configured_metrics
```

## File Input And Preprocessing

::: prism.dmri.eval.load_data
    options:
      show_root_heading: true
      show_root_full_path: false
      members:
        - load_and_process_data
        - load_brain_mask
        - read_in_brain
        - process_bvals
        - sort_by_bvals
        - normalize_in_brain

## Synthetic Input

::: prism.dmri.eval.data_sources
    options:
      show_root_heading: true
      show_root_full_path: false
      members:
        - generate_synthetic_data

## Precision

::: prism.inference.precision
    options:
      show_root_heading: true
      show_root_full_path: false
      members:
        - normalize_precision
        - resolve_precision_for_backend
        - apply_precision_to_cfg

## Kernel Stein Discrepancy

::: prism.inference.ksd
    options:
      show_root_heading: true
      show_root_full_path: false

## Batching And Sampling Cost

`eval_in_batches` applies a prepared function over voxel batches. It does not
load a model or build an evaluation pipeline.

::: prism.inference.sampling_methods
    options:
      show_root_heading: true
      show_root_full_path: false
      members:
        - eval_in_batches
        - network_evaluations_per_sample

## Array And Export Helpers

::: prism.dmri.eval.export_theta
    options:
      show_root_heading: true
      show_root_full_path: false
      members:
        - spherical_to_cartesian
        - map_over_voxels
        - embed_in_full_brain_array

::: prism.dmri.eval.export_models
    options:
      show_root_heading: true
      show_root_full_path: false
      members:
        - get_model_selection_exporter
