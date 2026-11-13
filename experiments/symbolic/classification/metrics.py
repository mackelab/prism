"""Confusion matrices, classification metrics and the similarity ordering."""

from __future__ import annotations

import numpy as np


def confusion_matrices(true_idx, pred_idx, num_classes):
    """Return integer counts and the row-normalised (recall) matrix."""
    counts = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(counts, (true_idx, pred_idx), 1)
    rows = counts.sum(axis=1, keepdims=True)
    normalised = np.divide(counts, rows, out=np.zeros(counts.shape), where=rows > 0)
    return counts, normalised


def _ratio(numerator, denominator):
    return np.divide(
        numerator,
        denominator,
        out=np.zeros(np.shape(numerator), dtype=float),
        where=np.asarray(denominator) > 0,
    )


def classification_metrics(counts, normalised, pred_topk, true_idx, topk=5):
    """Metrics of Table 1 from the confusion matrix and the top-k predictions.

    ``top1_micro`` is the overall accuracy and ``top1_macro`` equals the macro
    recall. ``topk_micro`` is the fraction of samples whose true class is among
    the k most probable classes; ``topk_macro`` is the fraction of classes whose
    true class is among the k most frequent predictions of that class's own
    samples (a row-wise statistic of the confusion matrix). As in the source,
    ties are broken by class index (``np.argsort``), so a class whose own samples
    are predicted into fewer than ``k`` distinct classes can count as a hit
    without ever having been predicted; this inflates ``topk_macro`` when
    predictions are concentrated.
    """
    total = int(counts.sum())
    classes = counts.shape[0]
    diagonal = np.diag(counts)
    support = counts.sum(axis=1)
    precision = _ratio(diagonal, counts.sum(axis=0))
    recall = _ratio(diagonal, support)
    f1 = _ratio(2 * precision * recall, precision + recall)
    macro_hits = sum(i in np.argsort(normalised[i])[-topk:] for i in range(classes))
    return {
        "num_samples": total,
        "num_classes": classes,
        "top1_micro": float(diagonal.sum() / total) if total else 0.0,
        "top1_macro": float(recall.mean()),
        f"top{topk}_micro": float(np.mean((pred_topk == true_idx[:, None]).any(axis=1)))
        if total
        else 0.0,
        f"top{topk}_macro": float(macro_hits / classes),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(np.average(f1, weights=support)) if total else 0.0,
        "support_min": int(support.min()),
        "support_max": int(support.max()),
        "support_mean": float(support.mean()),
    }


def similarity_order(masks, counts):
    """Greedy nearest-neighbour ordering of masks by Hamming distance.

    Starts at the first mask; ties go to the class with more samples.
    """
    bits = np.asarray(masks).astype(int)
    counts = np.asarray(counts)
    distance = (bits[:, None, :] != bits[None, :, :]).sum(axis=-1)
    order = [0]
    remaining = set(range(1, len(bits)))
    while remaining:
        candidates = np.array(sorted(remaining))
        best = candidates[
            np.lexsort((-counts[candidates], distance[order[-1], candidates]))
        ][0]
        order.append(int(best))
        remaining.remove(int(best))
    return np.array(order, dtype=int)
