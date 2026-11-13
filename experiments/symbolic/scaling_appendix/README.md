# rRMSE, rKSD and calibration error against model-space size (Figure A.2)

Three panels for the symbolic models of three widths (~5M, ~2M, ~1M parameters):
rRMSE (predictive error minus ground-truth noise, bootstrap interval over the
evaluation samples), rKSD (kernel Stein discrepancy divided by the mean prior
KSD, after dropping the outer 5% of values) and calibration error
(mean |coverage − α|) for the model and the parameter posterior, with the range of
the same error for perfectly uniform PITs shown in gray.

| File | Role |
| --- | --- |
| `plot.py` | aggregates the stored evaluation archives and draws the figure |

Source: `figures/fig2/scaling.ipynb` in the private symbolic source repository
(`mackelab/sbmi_transformer_symbolic`, commit `c8e118b8`).

**Status.** Only plotting and aggregation are packaged. The evaluation archives it
reads (`eval_pred_error_*`, `eval_ksd_*`, `eval_coverage_*`; about 200 MB) came from
evaluators that are not ported here, and the archives themselves are not published.
The Fig 4c evaluator in [`model_space_scaling`](../model_space_scaling/README.md)
computes the rRMSE quantity for K=15/30/50 only.

```bash
uv run python -m experiments.symbolic.scaling_appendix.plot RESULTS_DIR \
    --output results/scaling_metrics_appendix.svg
```

`RESULTS_DIR` must contain `eval_pred_error_{predictive,gt}.npz`, the `_small` and
`_tiny` predictive variants, and `eval_ksd_updated2_symbolic{K}[_small|_tiny].npz`
and `eval_coverage_{mask,theta}_updated2_symbolic{K}[_small|_tiny].npz` for
K = 20, 30, 50, 80, 100. The run takes a few minutes, mostly the rRMSE bootstrap
(`--n-boot` lowers it; the rKSD intervals always use 1000 repetitions). The figure is
written with a transparent background. The script stops with an error if a width is
missing one of those checkpoints. The gray band is the 2.5–97.5% range of the
calibration error of 32,768 uniform PITs (the number of evaluation datasets in the
coverage archives), computed on the 101-point α grid.

Run once on the stored archives, the curves and values look like the notebook's saved
figure (rRMSE about 20 for the smallest width at 2^100, rKSD about 0.26, calibration
errors between 0.001 and 0.008). That comparison was by eye, not numerical, and the
bootstrap random numbers differ.

**Known inconsistency, kept from the notebook.** The first checkpoint is
`updated2_symbolic20` (K=20) but is drawn at 2^15 in the rRMSE and calibration
panels and at 2^20 in the rKSD panel. Which model space size the paper intends there
is unresolved.
