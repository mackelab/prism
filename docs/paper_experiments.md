# Paper workflows

This map distinguishes runnable experiment code from model support and stored
figures. Each packaged figure has a recipe next to its code, listing every
command, and a documentation page with its protocol, options and caveats.

**Before you run a recipe.** Install with a CUDA extra
(`uv sync --locked --extra cuda12`, or `cuda13`) on an Ampere or newer NVIDIA
GPU. The [folder map](https://github.com/mackelab/prism/blob/main/experiments/README.md)
lists the experiment code. Recipes
submit Slurm jobs using the Tübingen partition presets; see
[another site or without Slurm](guides/training.md#run-at-another-site-or-without-slurm).
Running a recipe trains or evaluates **new** checkpoints: see
[what counts as reproduction](#what-counts-as-reproduction).

| Task | Paper result | Run | Repository coverage |
| --- | --- | --- | --- |
| Symbolic simulation efficiency | Figure 4b | [Recipe](https://github.com/mackelab/prism/tree/main/experiments/symbolic/simulation_efficiency#reproduce-figure-4b) · [details](experiments/fig4b.md) | [Simulation-efficiency evaluation](https://github.com/mackelab/prism/tree/main/experiments/symbolic/simulation_efficiency) and all seven fixed-data training presets are packaged. Larger runs retain the declared A100 stopping policy. The alternate batch-1024 training recipe and combined plotting still need integration. |
| Prior-conditioned symbolic scaling | Figure 4c | [Recipe](https://github.com/mackelab/prism/tree/main/experiments/symbolic/model_space_scaling#reproduce-figure-4c) · [details](experiments/fig4c.md) | K=15/30/50 training and [conditional predictive evaluation](https://github.com/mackelab/prism/tree/main/experiments/symbolic/model_space_scaling) are packaged. The complete model-space/width matrix and control modes are not. |
| Symbolic posterior illustrations | Figures 2, 4a, A.1 | [Recipe](https://github.com/mackelab/prism/tree/main/experiments/symbolic/illustrations#run-it) | [`illustrations`](https://github.com/mackelab/prism/tree/main/experiments/symbolic/illustrations) packages the observation, λ sweep, posterior draws and panels, with a `symbolic50_illustrative` preset for the ten-observation run. Figure 2/4a were run on the source checkpoint (step 1,396,000); the paper's exact panel values were not compared. |
| Symbolic model-selection classification | Table 1; Figure A.3 | [Recipe](https://github.com/mackelab/prism/tree/main/experiments/symbolic/classification#run-it) | [`classification`](https://github.com/mackelab/prism/tree/main/experiments/symbolic/classification) packages the confusion-matrix evaluator, Table 1 layout and Figure A.3 panel. On the published weights it reproduces the stored metrics; the paper's Top-1 (K=30, 80) and Top-5 (K=50, 80) entries differ from them, see the README. |
| Symbolic scaling metrics | Figure A.2 | [Recipe](https://github.com/mackelab/prism/tree/main/experiments/symbolic/scaling_appendix) | [`scaling_appendix`](https://github.com/mackelab/prism/tree/main/experiments/symbolic/scaling_appendix) ports the plotting and aggregation of rRMSE, rKSD and calibration error from stored archives; the evaluators that produce the archives are not packaged. |
| Condition-wise dMRI calibration | Figure 6b | [Recipe](https://github.com/mackelab/prism/tree/main/experiments/dmri/calibration#reproduce-figure-6b) · [details](experiments/fig6b.md) | [`calibration`](https://github.com/mackelab/prism/tree/main/experiments/dmri/calibration) packages the frozen condition allocation, model/parameter SBC and resumable execution. Original notebook panel plotting and a separate reference-band calculation are included; historical checkpoint identity remains unconfirmed. |
| Synthetic dMRI posterior quality | Synthetic Figure 6c | [Recipe](https://github.com/mackelab/prism/tree/main/experiments/dmri/synthetic_evaluation#reproduce-figure-6c-synthetic) · [details](experiments/fig6c.md) | [`synthetic_evaluation`](https://github.com/mackelab/prism/tree/main/experiments/dmri/synthetic_evaluation) runs public-checkpoint SBC/rKSD with matching synthetic inputs and includes the original notebook violin plotting code. It does not identify the historical paper checkpoint. |
| dMRI SBI baseline | Baseline in Tables 2–4 | [Recipe](https://github.com/mackelab/prism/tree/main/experiments/dmri/sbi_baseline#b3s-sbi-baseline) | [`sbi_baseline`](https://github.com/mackelab/prism/tree/main/experiments/dmri/sbi_baseline) trains and evaluates a joint neural-spline-flow baseline on simulated B3S data (optional `torch` and `sbi`). It is a cleaned port; it has not been checked against the paper's baseline numbers, and real-data evaluation is not included. |
| ESS and run time against diffusion steps | Figure A.4 | [Recipe](https://github.com/mackelab/prism/tree/main/experiments/dmri/diffusion_steps) | [`diffusion_steps`](https://github.com/mackelab/prism/tree/main/experiments/dmri/diffusion_steps) ports the sweep and plot from the source notebook. The sweep script is untested (needs real data, a B3S checkpoint and the original GPU); only the plot is tested. |
| Real-voxel posterior and posterior quality | Figure 6a; real Figure 6c | — | Requires the real-data selection and matching checkpoint; the complete workflow is not packaged. |
| Run time and memory | Run-time comparison | [Profiling guide](https://github.com/mackelab/prism/tree/main/experiments/profiling#run-time-profiling) | [`profiling`](https://github.com/mackelab/prism/tree/main/experiments/profiling) times mask/parameter sampling, log probabilities and the training step of any saved run. The recorded source timings are tabulated for reference; this port was not re-measured. |
| Tractography | Figure 7 | — | Not packaged; real data and external tractography software are required. Stored figure inputs alone do not regenerate the tractography. |
| Extended dMRI model spaces, generalization and FOD | Figure 8 | — | Model support and source evaluation scripts exist; the full experiment workflows are not packaged. |

## What counts as reproduction

Frozen source references: symbolic
[`c8e118b`](https://github.com/mackelab/sbmi_transformer_symbolic/tree/c8e118b8bd557f2b001812f00af982b136f0d793)
and dMRI
[`f363a66`](https://github.com/mackelab/dmri/tree/f363a66b1a002c316024bf5c5d11233a90c5319f).
Implemented code and reduced execution checks do not by themselves establish
quantitative reproduction of a paper panel. A recipe run produces a new
checkpoint or evaluates the public dMRI checkpoint, which has not been
identified as the paper's; report such outputs as new results.

## Checkpoints and saved source results

The following records belong to the frozen upstream trees linked above.
The symbolic tree tracks no original training checkpoints. Converted EMA weights of
the Figure 4b and 4c runs are published separately as safetensors; see
[Symbolic checkpoints](reference/hub.md#symbolic-checkpoints-safetensors). Its evaluators name
run directories, and some notebooks record a step, but those references do not
provide the weights. The dMRI tree also refers to external result directories;
the separate [published dMRI bundles](reference/hub.md#published-checkpoint-selections)
have not been identified as the exact figure-used checkpoints.

| Paper result | Original checkpoint status | Saved upstream records |
| --- | --- | --- |
| Figures 2, 4a, A.1: symbolic illustrations | Converted EMA weights are published in `manugloeck/prism-symbolic-pretrained`: `symbolic50_illustrative/figure` (step 1,396,000, the step named by the illustrative notebook; Figures 2 and 4a) and `symbolic50` latest, step 362,000 (Figure A.1). The A.1 panels come from `figures/fig2/illustrative.ipynb` (seed 69, 1,000 draws, draws 627 and 12); the recipe reproduces the figure's equations exactly. | Notebook outputs, run references and SVGs. Figure 2/4a panel values were not compared. |
| Figure 4b: PRISM learning curve | Seven run directories are referenced. Best EMA weights of all seven points are published in `manugloeck/prism-symbolic-pretrained`, and with the evaluator's `--preset` option every point reproduces the figure's excess RMSE to within 0.003 (largest difference 0.0029 at 200k; 100k: 0.665 against 0.667). | Training presets, evaluation notebook and curve SVG. |
| Figure 4b: SBMI comparator | No comparator weights located in the inspected source. | `figures/fig2/rmse_n_training_big_v2.pkl` contains aggregate comparison values and noise/prior references. |
| Figures 4c, A.2: symbolic scaling | K=15/30/50/80/100 at three widths (best and latest EMA) are published in `manugloeck/prism-symbolic-pretrained`, matched to runs by experiment name. The 15 latest checkpoints agree with the stored 2^20-sample conditional-RMSE means (largest |z| 1.83 at 16,384 datasets). | Prediction-error, coverage and KSD NPZ archives; CSV summaries and figures for the scaling families. |
| Table 1, Figure A.3: symbolic classification | Original weights not supplied. | Five `confussion_matrices/metrics/metrics_*.json` summaries and confusion-matrix figure assets. |
| Figure 6, Figures A.4–A.7: B3S posteriors and diagnostics | Scripts reference `updated5_b3s_2_4_6_128`; an exact exported checkpoint is unidentified. | Notebook outputs and figures, `dmri_illustrative_evidences.npz`, and classification JSONs. The complete B3S coverage/KSD input arrays are not supplied. |
| Figure 7: tractography | Inference checkpoint and full downstream provenance remain unresolved. | Plotting notebook, figures, tract names and 42 saved tract correlations per comparison in TXT files. |
| Figure 8, Figures A.8–A.12: extended dMRI spaces | The referenced broad and convolved model weights are not supplied by the frozen tree or the public Hub revision above. | Some coverage/KSD NPZ archives, classification JSONs, model-ranking CSVs, notebook outputs and figures. These do not provide every panel's input arrays. |
| Tables 2–4: dMRI metrics | Matching historical checkpoints remain unidentified or unavailable in the inspected artifacts. | Saved metric summaries in notebooks; complete input volumes and metric maps are not supplied. |
| Table 5: dMRI classification | Matching historical checkpoints remain unidentified or unavailable in the inspected artifacts. | `b3s.json`, `all.json`, `all_conv.json` and related baseline summaries. |

The symbolic artifacts are under upstream `figures/fig2/`; dMRI artifacts are
under `figures/b3s_fig/`, `figures/all_fig/`, and `figures/all_conv_figs/`.
Source directory numbering predates the paper's final figure numbering.

Saved arrays can support replotting or numerical comparison where the required
inputs are present. Figures and aggregate metrics alone cannot reconstruct
network weights or a full evaluation dataset. Retain the source revision,
checkpoint/configuration identity, selected step and EMA/raw choice, evaluation
inputs and protocol with any new run. Newly trained PRISM Clean checkpoints
and their evaluations should be identified as new results.
