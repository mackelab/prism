# ESS and run time against diffusion steps (Figure A.4)

How many solver steps does the B3S posterior need? For one real voxel, 1000
posterior samples of the full three-stick model are drawn with 2 to 512 steps; each
setting is timed (after a compile run) and its effective sample size is computed from
importance weights against the network's own density evaluated at 1024 steps.

| File | Role |
| --- | --- |
| `run.py` | loads a B3S run and a voxel, sweeps the step counts, writes an npz |
| `plot.py` | the two-panel figure |

Source: [`figures/b3s_fig/dmri_illustrative.ipynb`](https://github.com/mackelab/dmri/blob/f363a66b1a002c316024bf5c5d11233a90c5319f/figures/b3s_fig/dmri_illustrative.ipynb)
in [`mackelab/dmri`](https://github.com/mackelab/dmri) at `f363a66`.

**Status: not tested.** `run.py` was ported from a notebook but has not been run:
it needs real HCP data (`dipy`) and a B3S checkpoint, and the run times are
specific to the hardware (the paper used an Nvidia H100). Only `plot.py` is covered
by a test.

```bash
uv run python -m experiments.dmri.diffusion_steps.run results/b3s \
    --data-dir data/hcp --voxel 50000 --output results/b3s/steps_ess.npz
uv run python -m experiments.dmri.diffusion_steps.plot results/b3s/steps_ess.npz \
    --output results/b3s/steps_ess.svg
```

The voxel is the in-brain voxel with 0-based index 50,000 of the dataset after the notebook's
preprocessing (b-values rounded to 0/1000/2000, normalisation by the mean b=0
signal, 99.9% quantile clipping, volumes sorted by b-value). Which checkpoint the
paper's panel used is unresolved (see the dMRI rows of the
[experiment index](../../../docs/paper_experiments.md)).

Not checked against the notebook: `run.py` reads the data with its own loader instead of
`prism.dmri.eval.load_data`, loads the run with the bare checkpoint loader (not the
`experiments/dmri/_shared` helpers that map legacy configs and fix the diffusion bounds),
and passes only `t_min=5e-4`, leaving `t_max` at the checkpoint's default. Whether this
matches the notebook is unverified.
