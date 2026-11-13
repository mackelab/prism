# Symbolic model-selection classification (Table 1, Figure A.3)

How often does the model-selection network recover the model that generated an
observation? A support set of 200 models (binary masks) is drawn from the run's
simulator, `--num-per-class` observations (default 1000) are simulated from each,
and the network scores all 200 masks for every observation. The arg-max is the
prediction; the result is the confusion matrix and the metrics of Table 1.

| File | Role |
| --- | --- |
| `evaluate.py` | draws the data, predicts, writes metrics JSON (and optionally the arrays) |
| `metrics.py` | confusion matrices, classification metrics, similarity ordering |
| `table.py` | per-K metrics files to the Table 1 layout |
| `plot.py` | Figure A.3 panel from the saved arrays |

## Run it

From the repository root, for a saved run of each model-space size (the K=15/30/50
presets are `symbolic15`, `symbolic30`, `symbolic50`; K=80/100 are `symbolic80`,
`symbolic100`). Add `--slurm a100` to submit a Slurm job instead.

```bash
uv run python -m experiments.symbolic.classification.evaluate results/symbolic50 \
    --which latest --save-arrays --output results/symbolic50/classification.json
uv run python -m experiments.symbolic.classification.table \
    15=results/symbolic15/classification.json 50=results/symbolic50/classification.json
uv run python -m experiments.symbolic.classification.plot results/symbolic50/classification \
    --output results/symbolic50/confusion.svg
```

To use the published EMA weights instead of a saved run, pass the preset in
place of the run directory (`--weights-dir DIR` reads a local copy):

```bash
uv run python -m experiments.symbolic.classification.evaluate --preset symbolic50 \
    --which latest --save-arrays --output results/symbolic50_published/classification.json
```

The full protocol (200 classes × 1000 observations, 200,000 predictions each
scoring 200 masks) is a GPU job; lower `--num-per-class` for a diagnostic. Use
`--pred-batch-size` to fit smaller GPUs (a batch of 100 ran out of memory on a
10 GB card for K≥30; 10 worked). The weights are the EMA parameters.

Reported metrics: `top1_micro` (overall accuracy), `top5_micro` (true class among the
five most probable, per sample), `top5_macro` (per class: its own samples' five most
frequent predictions contain it; ties are broken by class index as in the source, which can count
a class that was never predicted as a hit when its samples spread over fewer than five classes), macro precision, recall and F1. Macro recall equals
`top1_macro`.

## Comparison with the stored source results

The source repository stores `metrics_{20,30,50,80,100}.json` (K=15…100, EMA
weights; whether best or latest is not recorded). Evaluating the published
`latest` EMA weights with this port at 100 observations per class (about 20,000
predictions, standard error about 0.3 points on accuracy) against those files and
Table 1 of the paper:

| K | Top-1 here | Top-1 stored | Top-1 paper | Top-5 sample here / stored | Top-5 class here / stored | Top-5 paper |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 15 | 0.794 | 0.797 | 0.796 | 0.981 / 0.981 | 1.000 / 1.000 | 0.981 |
| 30 | 0.736 | 0.742 | 0.730 | 0.948 / 0.952 | 0.995 / 0.995 | 0.995 |
| 50 | 0.689 | 0.688 | 0.690 | 0.929 / 0.929 | 0.990 / 0.990 | 0.995 |
| 80 | 0.543 | 0.548 | 0.560 | 0.869 / 0.864 | 0.955 / 0.965 | 0.940 |
| 100 | 0.509 | 0.503 | 0.503 | 0.842 / 0.839 | 0.925 / 0.905 | 0.905 |

Accuracy, precision, recall and F1 agree with the stored files within sampling error
(the 200 support classes are the same: the class sizes match). Two things in the paper's
Table 1 are **not** reproduced by the stored files: the Top-1 row for K=30 (0.730) and
K=80 (0.560) differ from the stored 0.742 and 0.548, and the Top-5 row mixes the
per-sample value (K=15) and the per-class value (K=30, K=100) and has values for K=50 and
K=80 (0.995, 0.940) that match neither. The paper's macro precision, recall and F1 rows do match the
stored files to three decimals (this port is within about 0.01 of them at 20,000 predictions). The `best` weights agree with `latest` to within sampling
error at K=15 (checked; the other sizes were evaluated with `latest` only), so this check does not tell which
selection the source used.

## Differences from the source scripts

- EMA weights are required (no fall-back to raw parameters).
- Checkpoint loading, simulators and slurm submission go through `prism`;
  the simulator keyword is `mask_prior_hyperparameter`.
- `metrics.py` is checked against the frozen source functions on a seeded problem
  (`tests/fixtures/symbolic_classification_source.*`). The data set construction is kept,
  including that the 1000 initial prior draws stay in the evaluation set.
- `table.py` and `plot.py` replace notebook cells (similarity order and layout unchanged).
