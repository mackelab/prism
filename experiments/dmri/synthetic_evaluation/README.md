# Synthetic dMRI evaluation (Figure 6c, synthetic part)

Model/parameter SBC and relative KSD of the public `msb3s_2_4_6_128` model on
synthetic observations. The
[Figure 6c documentation](../../../docs/experiments/fig6c.md) describes the
input checks, metrics, outputs, reduced checks and plotting options.

| File | Role |
| --- | --- |
| `run.py` | SBC and rKSD evaluation |
| `plot.py` | Figure 6c posterior-quality violins |

## Reproduce Figure 6c (synthetic)

Run from the repository root. The commands run in the current shell unless
they end in `--slurm a100`, which submits them as a Slurm job instead. To run
elsewhere, see
[another site or without Slurm](../../../docs/guides/training.md#run-at-another-site-or-without-slurm).

### Checkpoint

This workflow accepts only the public checkpoint at the pinned revision. Stage
it once:

```bash
uv run hf download manugloeck/dmri-pretrained --revision 017561ffbf77b109c9fe6c3d9c7d48831780ddc3 --include "msb3s_2_4_6_128/**" --local-dir results/pretrained
```

### Evaluation

```bash
uv run python -m experiments.dmri.synthetic_evaluation.run --checkpoint results/pretrained/msb3s_2_4_6_128 --which best --output results/dmri_synthetic --slurm a100
```

### Panel

```bash
uv run python -m experiments.dmri.synthetic_evaluation.plot --synthetic results/dmri_synthetic/short_overall/samples.npz --measurements 105 --kernel frozen --output results/dmri_synthetic_panel
```

Real-voxel groups need real data and are not packaged.
