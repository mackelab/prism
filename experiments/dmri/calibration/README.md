# Condition-wise dMRI calibration (Figure 6b)

Model and parameter SBC over the frozen Figure 6b condition allocation for a
Ball3Stick (B3S) checkpoint. The
[Figure 6b documentation](../../../docs/experiments/fig6b.md) describes the
checkpoint requirements, conditions, statistics, resume rules, reduced checks
and plotting options.

| File | Role |
| --- | --- |
| `evaluate.py` | Runs one mode (`model` or `theta`); resumable |
| `protocol.py` | Frozen condition schedule and statistics |
| `chunks.py` | Resumable trial-chunk format |
| `plot.py` | Figure 6b panels and the reference band |

## Reproduce Figure 6b

Run from the repository root. `prism train` submits a Slurm job with the
resources set in its preset. The scripts run in the current shell unless the
command ends in `--slurm a100`, which submits them as a Slurm job instead. To
run elsewhere, see
[another site or without Slurm](../../../docs/guides/training.md#run-at-another-site-or-without-slurm).

### Checkpoint

Train the multishell B3S model (70 training hours on one H100):

```bash
uv run prism train +experiment/train=msb3s_2_4_6_128
```

To use the public checkpoint instead, download it as in the
[Figure 6c recipe](../synthetic_evaluation/README.md#checkpoint) and replace
`results/msb3s_2_4_6_128` below with `results/pretrained/msb3s_2_4_6_128`.
Neither checkpoint has been identified as the paper's.

### Evaluation

Run both modes, each into its own output directory:

```bash
uv run python -m experiments.dmri.calibration.evaluate --checkpoint results/msb3s_2_4_6_128 --mode model --output results/msb3s_calibration_model --slurm a100
uv run python -m experiments.dmri.calibration.evaluate --checkpoint results/msb3s_2_4_6_128 --mode theta --output results/msb3s_calibration_theta --slurm a100
```

If a job stops early, repeat the same command with `--resume`.

### Panels

Compute the reference band once, then plot both panels:

```bash
uv run python -m experiments.dmri.calibration.plot --make-band 10000 --output results/calibration_band --slurm a100
uv run python -m experiments.dmri.calibration.plot --model results/msb3s_calibration_model/results.json --theta-result results/msb3s_calibration_theta/results.json --theta-band results/calibration_band/reference_band.json --curve-labels UKB-like HCP-like --output results/calibration_panels
```
