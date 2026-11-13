# Run-time profiling

Times the operations of a trained checkpoint, for the run-time comparison of
the symbolic and dMRI models.

```bash
python -m experiments.profiling.run_time results/symbolic50 \
    --slurm a100 --output results/symbolic50/run_time_gpu.json
JAX_PLATFORMS=cpu python -m experiments.profiling.run_time results/<dmri run> \
    --skip-training --output results/<dmri run>/run_time_cpu.json
```

`run_time` loads the run directory (`config.yaml` plus `checkpoints/`) with the
EMA weights, draws inputs from the run's own simulator (`--simulator-index`
chooses among several), and profiles for one observation

| Operation | What is timed |
| --- | --- |
| `sample_mask` | draw a model mask (autoregressive) |
| `log_prob_mask` | log probability of a mask |
| `sample_theta` | draw parameters given the mask (`--theta-num-steps`, default 64) |
| `log_prob_theta` | parameter log probability by ODE (`--theta-num-steps`) |

and, for a simulated batch of `--loss-batch-size` (default 2048) examples,

| Operation | What is timed |
| --- | --- |
| `loss_forward` | training loss |
| `loss_backward` | loss and gradients |

Each operation is compiled ahead of time (`compile_s`), called once
(`first_exec_s`), then called `--warmup` plus `--repeat` times; the summary
reports the steady-state mean, median, standard deviation, minimum and maximum
in seconds, XLA's memory estimate and, on GPU, the allocator peak (the device's cumulative high-water mark, not reset between operations and summed over devices, so later rows include earlier ones). The table is
printed and `--output` writes the complete JSON, including parameter counts per
component. Results depend on the device; the JSON records the device kind and
the JAX version. Use `--skip-training` on the CPU.

## Reference: results recorded by the original scripts

Steady-state means in milliseconds on one GPU (device model not recorded), batch
2048 for the loss, 64 diffusion steps, from the original run-time measurements: the dMRI rows from the run-time
analysis in [`mackelab/dmri`](https://github.com/mackelab/dmri), the symbolic
rows from the original symbolic code. These are for orientation, not a
reproduction target: this implementation was not re-measured.

| Run | Parameters | `sample_mask` | `log_prob_mask` | `sample_theta` | `log_prob_theta` | `loss_forward` | `loss_backward` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dMRI B3S (`b3s`) | 6.8 M | 1.05 | 0.34 | 22.2 | 138.7 | 10.6 | 35.1 |
| dMRI 10 models (`bszt`) | 9.4 M | 3.53 | 0.47 | 30.9 | 247.6 | 20.3 | 69.8 |
| dMRI 19 models (`bszt_conv`) | 9.5 M | 6.10 | 0.47 | 32.6 | 358.0 | 27.5 | 101.3 |
| Symbolic K=15 (`sym15`) | 5.8 M | 12.2 | 0.81 | 65.9 | 159.2 | 12.8 | 39.4 |
| Symbolic K=15 tiny | 1.3 M | 12.2 | 0.76 | 56.5 | 148.4 | 7.6 | 24.2 |
| Symbolic K=50 (`sym50`) | 5.8 M | 88.1 | 1.88 | 83.4 | 327.1 | 54.8 | 112.0 |
| Symbolic K=50 tiny | 1.3 M | 88.2 | 1.77 | 73.5 | 247.9 | 43.7 | 78.0 |
| Symbolic K=100 (`sym100`) | 5.8 M | 299.9 | 3.21 | 118.9 | 784.7 | 251.1 | 359.0 |
| Symbolic K=100 tiny | 1.3 M | 296.8 | 3.14 | 109.2 | 576.1 | 230.9 | 302.6 |
| dMRI B3S on CPU (`b3s_cpu`) | 6.8 M | 6.47 | 3.99 | 89.5 | 4316.3 | — | — |

## Differences from the original scripts

- One shared timing helper and one entry point replace the two near-identical
  1000-line scripts; the generic factory/demo commands, trace capture and
  memory-profile files are gone.
- Inputs come from the run's simulator. The dMRI source used random acquisitions
  with uniform random signals, so timings need not match exactly.
- EMA weights are required (no fallback to raw parameters).
- The device is chosen with `JAX_PLATFORMS`; the sampling temperature option is
  not offered.
