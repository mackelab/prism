# B3S SBI baseline

A joint neural-spline-flow baseline for the Ball-and-3-Stick parameters: a
transformer encodes the signal and its acquisition, and an `sbi` neural spline
flow models the 11 parameters given that encoding. It is the "SBI joint"
comparison for PRISM's parameter posterior (paper Tables 2–4). It trains on
simulated data only and, unlike PRISM, has no model selection.

It needs `torch` and `sbi`, which are **not** dependencies of `prism`. Install
them into the project environment before running the recipe (a later
`uv sync` removes them again):

```bash
uv pip install torch sbi
```

Run commands from the repository root. Each script takes `--slurm PARTITION` to
submit itself as a Slurm job (use a partition with a GPU; the job environment
needs `torch` and `sbi` too); without it the command runs in the current shell.

## Train

Every step simulates fresh data for the two random acquisition families
(105 and 297 measurements, `--batch-size` voxels each) and averages their
negative log-likelihoods. The best checkpoint on a fixed held-out batch is kept.

```bash
uv run python -m experiments.dmri.sbi_baseline.train --output results/sbi_b3s \
    --num-steps 30000000 --slurm h100_long
```

The source ran this for 30M steps (three days on one H100). `--resume` continues
from `results/sbi_b3s/checkpoints/latest.pt` with the network and simulator saved
in the run's `config.json`; architecture flags need not be repeated, and a
conflicting one is rejected. Use `--sim-type
MultiShellBall3StickSharedDiffusivity` for the multi-shell simulator.

## Evaluate

```bash
uv run python -m experiments.dmri.sbi_baseline.evaluate \
    results/sbi_b3s/checkpoints/best.pt --output results/sbi_b3s_eval.json --slurm h100_long
```

For each acquisition family, `--num-datasets` simulated problems (default 100)
with `--num-samples` posterior samples each (default 50) give the effective
sample size (as a fraction of the samples), the signal RMSE of posterior
predictions, and the multi-scale kernel Stein discrepancy with its p-value.

A quick local check of the whole path:

```bash
uv run python -m experiments.dmri.sbi_baseline.train --output results/sbi_smoke --num-steps 5 \
    --batch-size 8 --eval-batch-size 8 --eval-every 5 --model-dim 16 --output-dim 16 \
    --num-cls-tokens 2 --num-layers 1 --hidden-features 16 --num-transforms 2
uv run python -m experiments.dmri.sbi_baseline.evaluate results/sbi_smoke/checkpoints/best.pt \
    --regimes hcp --num-datasets 2 --num-samples 16
```

## Differences from the source

This is a cleaned-up port of `figures/b3s_baseline` at
[`f363a66`](https://github.com/mackelab/dmri/tree/f363a66b1a002c316024bf5c5d11233a90c5319f);
it has not been
checked against the paper's numbers or the original checkpoints.

- The condition is one packed tensor, so the unmodified `sbi` builder and the
  underlying `nflows` flow replace the source's dict-conditioned wrapper.
- Data are simulated online each step, without the source's replay buffer and
  prefetching; there is no W&B logging and only `best.pt` and `latest.pt` are kept.
- The held-out evaluation batch is fixed, not drawn from the data stream.
- ESS weights use the likelihood times the standard-normal prior, like PRISM's
  KSD and ESS. The source's baseline script used the likelihood alone.
- Real-data evaluation (UKB/HCP volumes) is not included.
