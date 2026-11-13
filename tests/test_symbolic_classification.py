import json
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

import prism.config  # noqa: F401  (prism.simulators.config needs it imported first)
from experiments.symbolic.classification import evaluate, plot, table
from experiments.symbolic.classification.metrics import (
    classification_metrics,
    confusion_matrices,
    similarity_order,
)
from prism.simulators.config import resolve_simulator_model
from prism.train.build_model import build_model
from prism.train.build_simulator import build_simulator

FIXTURES = Path(__file__).parent / "fixtures"


def test_metrics_match_the_frozen_source_functions():
    arrays = np.load(FIXTURES / "symbolic_classification_source.npz")
    expected = json.loads(
        (FIXTURES / "symbolic_classification_source.json").read_text()
    )
    counts, normalised = confusion_matrices(arrays["true_idx"], arrays["pred_idx"], 12)
    np.testing.assert_array_equal(counts, arrays["counts"])
    np.testing.assert_array_equal(normalised, arrays["conf"])
    actual = classification_metrics(
        counts, normalised, arrays["pred_topk"], arrays["true_idx"], topk=5
    )
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        assert actual[key] == pytest.approx(value, rel=0, abs=1e-12), key


def test_perfect_and_empty_classes():
    true = np.array([0, 0, 1, 1])
    counts, normalised = confusion_matrices(true, true, 3)  # class 2 has no samples
    metrics = classification_metrics(
        counts, normalised, np.stack([true] * 2, 1), true, topk=2
    )
    assert metrics["top1_micro"] == 1.0
    assert metrics["macro_recall"] == pytest.approx(
        2 / 3
    )  # the empty class counts as zero
    assert metrics["support_min"] == 0


def test_similarity_order_walks_to_the_nearest_mask():
    masks = np.array(
        [[0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 1], [1, 1, 1, 0]], dtype=bool
    )
    order = similarity_order(masks, np.array([5, 5, 5, 5]))
    assert order.tolist() == [0, 2, 1, 3]
    # ties go to the more frequent class
    tied = similarity_order(
        np.array([[0, 0], [0, 1], [1, 0]], dtype=bool), np.array([1, 1, 9])
    )
    assert tied.tolist() == [0, 2, 1]


def _config():
    fixture = FIXTURES / "symbolic_sbmi_nn.json"
    cfg = OmegaConf.create(json.loads(fixture.read_text())["config"])
    cfg.simulator.model_class = "prism.symbolic.paper.model_specs.SymbolicModelV15"
    cfg.simulator.input_dim = 10
    cfg.simulator.observation_format = "signal_blocks"
    cfg.simulator.x_min, cfg.simulator.x_max, cfg.simulator.x_step = 0.0, 10.0, 0.02
    cfg.simulator.mask_prior = {"alpha": 1.0, "beta": 1.0, "min_active_models": 1}
    cfg.model.name = (
        "prism.symbolic.paper.model.SymbolicPaperModelConfigMaskPriorAmortized"
    )
    cfg.model.model_selection_net.name = "prism.symbolic.paper.autoregressive.SymbolicPaperModelSelectionAmortizedPriorConfig"
    cfg.model.model_selection_net.params.prior_params_embed_dim = 8
    cfg.model.model_selection_net.params.mask_prior_dim = 1
    cfg.model.model_selection_net.params.precondition_with_mask_prior = True
    return cfg


@pytest.fixture(scope="module")
def symbolic():
    cfg = _config()
    model = build_model(cfg, resolve_simulator_model(cfg))
    model.eval()
    return model, build_simulator(cfg)[1][0]


def test_dataset_has_one_block_of_samples_per_support_class(symbolic):
    _, generator = symbolic
    xs, masks, priors = evaluate.build_dataset(generator, 3, 4, sim_batch_size=4)
    assert len(xs) == len(masks) == len(priors) == 5 * 3 + 3 * 4
    classes, counts, labels = evaluate.support_classes(masks, 3)
    assert classes.shape == (3, masks.shape[1]) and (counts >= 4).all()
    assert (labels >= 0).sum() == counts.sum()


def test_evaluation_is_reproducible_and_consistent(symbolic):
    model, generator = symbolic
    kwargs = dict(
        num_classes=3, num_per_class=3, sim_batch_size=3, pred_batch_size=3, topk=2
    )
    first = evaluate.evaluate_model(model, generator, **kwargs)
    again = evaluate.evaluate_model(model, generator, **kwargs)
    metrics, counts, normalised, classes, class_counts = first
    np.testing.assert_array_equal(counts, again[1])
    assert metrics["num_samples"] == counts.sum() == class_counts.sum()
    assert metrics["top2_micro"] >= metrics["top1_micro"]
    np.testing.assert_allclose(normalised.sum(axis=1), 1.0)
    with pytest.raises(ValueError):
        evaluate.evaluate_model(model, generator, **{**kwargs, "topk": 9})


def test_table_and_figure(tmp_path, symbolic):
    stored = {
        "15": json.loads(
            '{"top1_micro": 0.797, "top5_micro": 0.981, "top5_macro": 1.0, "macro_precision": 0.813,'
            ' "macro_recall": 0.797, "macro_f1": 0.794}'
        )
    }
    text = table.markdown_table(stored)
    assert "| Accuracy (Top-1) | 0.797 |" in text and "| Macro F1 | 0.794 |" in text
    rng = np.random.default_rng(0)
    figure = plot.plot_confusion(
        rng.random((6, 6)), rng.random((6, 4)) > 0.5, np.arange(6) + 1
    )
    assert len(figure.axes) == 4


def test_main_refuses_to_overwrite(tmp_path):
    output = tmp_path / "metrics.json"
    output.write_text("{}")
    with pytest.raises(FileExistsError):
        evaluate.main([str(tmp_path), "--output", str(output)])
