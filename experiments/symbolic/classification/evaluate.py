"""Classify which model generated an observation (Table 1, Figure A.3 data).

A set of 200 support models (binary masks) is drawn from the run's simulator;
``--num-per-class`` observations are simulated from each with its own mask-prior
hyperparameter. The model-selection network scores every support mask for each
observation; the arg-max is the prediction. The result is the confusion matrix
and the classification metrics of Table 1.

Example: python -m experiments.symbolic.classification.evaluate results/symbolic50 \
    --which latest --slurm a100 --output results/symbolic50/classification.json

Use a small ``--num-per-class`` for pipeline diagnostics.
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from prism.utils.slurm import add_slurm_arguments, submit

from .._published import add_published_arguments, check_weights_source, load_published
from .metrics import classification_metrics, confusion_matrices

SEED = 123


def build_dataset(
    generator, num_classes, num_per_class, seed=SEED, sim_batch_size=10_000
):
    """Observations, true masks and mask-prior hyperparameters.

    As in the original evaluation, ``5 * num_classes`` prior draws provide the
    support masks (the first ``num_classes`` distinct masks in sorted order, each
    with the hyperparameter of its first draw) and are kept in the data set.
    """
    rng, rng_init = jax.random.split(jax.random.PRNGKey(seed))
    batch = jax.vmap(generator)(jax.random.split(rng_init, 5 * num_classes))
    masks, first = np.unique(np.asarray(batch["model_mask"]), axis=0, return_index=True)
    support = masks[:num_classes]
    priors = np.asarray(batch["mask_prior"])[first][:num_classes]

    rngs = jax.random.split(jax.random.split(rng)[1], num_per_class)
    xs, true_masks, mask_priors = (
        [batch["x"]],
        [batch["model_mask"]],
        [batch["mask_prior"]],
    )
    for mask, prior in zip(support, priors, strict=True):
        data = jax.lax.map(
            partial(
                generator,
                model_mask=jnp.asarray(mask),
                mask_prior_hyperparameter=jnp.asarray(prior),
            ),
            rngs,
            batch_size=sim_batch_size,
        )
        xs.append(data["x"])
        true_masks.append(data["model_mask"])
        mask_priors.append(data["mask_prior"])
    return (
        jnp.concatenate(xs),
        jnp.concatenate(true_masks),
        jnp.concatenate(mask_priors),
    )


def support_classes(true_masks, num_classes):
    """The ``num_classes`` most frequent masks and each sample's class (or -1)."""
    masks = np.asarray(true_masks)
    unique, counts = np.unique(masks, axis=0, return_counts=True)
    order = np.argsort(-counts)[:num_classes]
    classes, class_counts = unique[order], counts[order]
    index = {tuple(m.tolist()): i for i, m in enumerate(classes)}
    labels = np.array([index.get(tuple(m.tolist()), -1) for m in masks], dtype=int)
    return classes, class_counts, labels


def predict(model, classes, xs, mask_priors, topk, batch_size):
    """Arg-max and top-k class indices by model-selection log probability."""
    classes = jnp.asarray(classes)

    def top_k(args):
        x, prior = args
        scores = jax.vmap(lambda m: model.log_prob_mask(m, x=x, mask_prior=prior))(
            classes
        )
        best = jax.lax.top_k(scores, topk)[1]
        return best[0], best

    first, ranked = jax.lax.map(top_k, (xs, mask_priors), batch_size=batch_size)
    return np.asarray(first), np.asarray(ranked)


def evaluate_model(
    model,
    generator,
    *,
    num_classes=200,
    num_per_class=1000,
    seed=SEED,
    topk=5,
    sim_batch_size=10_000,
    pred_batch_size=200,
):
    """Return ``(metrics, counts, normalised, classes, class_counts)``."""
    xs, true_masks, mask_priors = build_dataset(
        generator, num_classes, num_per_class, seed, sim_batch_size
    )
    classes, class_counts, labels = support_classes(true_masks, num_classes)
    if topk > len(classes):
        raise ValueError(
            f"topk ({topk}) exceeds the number of classes ({len(classes)})"
        )
    keep = labels >= 0
    labels = labels[keep]
    first, ranked = predict(
        model, classes, xs[keep], mask_priors[keep], topk, pred_batch_size
    )
    counts, normalised = confusion_matrices(labels, first, len(classes))
    metrics = classification_metrics(counts, normalised, ranked, labels, topk)
    return metrics, counts, normalised, classes, class_counts


def load_run(path, which):
    from flax import nnx

    from prism.train.build_simulator import build_simulator
    from prism.train.utils import load_cfg, load_checkpoint, select_checkpoint_params

    checkpoint, model, _ = load_checkpoint(path, which=which)
    nnx.update(model, select_checkpoint_params(checkpoint, "params_ema"))
    model.eval()
    _, generators = build_simulator(load_cfg(path))
    return model, generators[0], int(checkpoint["step"])


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "checkpoint", type=Path, nargs="?", help="saved run directory (or --preset)"
    )
    parser.add_argument("--which", default="latest", choices=("latest", "best"))
    parser.add_argument("--num-classes", type=int, default=200)
    parser.add_argument("--num-per-class", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--sim-batch-size", type=int, default=10_000)
    parser.add_argument("--pred-batch-size", type=int, default=200)
    parser.add_argument(
        "--output", type=Path, required=True, help="metrics JSON (must not exist)"
    )
    parser.add_argument(
        "--save-arrays",
        action="store_true",
        help="also write <output stem>_counts.npy, _matrix.npy, _class_counts.npy and _masks.npy next to the JSON",
    )
    add_published_arguments(parser)
    add_slurm_arguments(parser)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    check_weights_source(parser, args.checkpoint, args.preset, args.weights_dir)
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.slurm:
        return submit(args, argv, main)
    published = None
    if args.preset:
        bundle, published = load_published(args.preset, args.which, args.weights_dir)
        model, generator, step = bundle.model, bundle.generator, bundle.step
    else:
        model, generator, step = load_run(args.checkpoint, args.which)
    metrics, counts, normalised, classes, class_counts = evaluate_model(
        model,
        generator,
        num_classes=args.num_classes,
        num_per_class=args.num_per_class,
        seed=args.seed,
        topk=args.topk,
        sim_batch_size=args.sim_batch_size,
        pred_batch_size=args.pred_batch_size,
    )
    record = {
        **metrics,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "published_weights": published,
        "which": args.which,
        "step": step,
        "params_used": "ema",
        "seed": args.seed,
        "num_per_class": args.num_per_class,
        "jax_version": jax.__version__,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    if args.save_arrays:
        stem = args.output.with_suffix("")
        np.save(f"{stem}_counts.npy", counts)
        np.save(f"{stem}_matrix.npy", normalised)
        np.save(f"{stem}_masks.npy", classes)
        np.save(f"{stem}_class_counts.npy", class_counts)
    print(json.dumps(record, indent=2))
    return record


if __name__ == "__main__":
    main()
