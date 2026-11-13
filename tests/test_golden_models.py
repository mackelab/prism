"""Saved runs and presets rebuild the same models; see ``golden_models.py``."""

import json

import numpy as np
import pytest
from golden_models import (
    GOLDEN_PATH,
    build_saved_run,
    model_outputs,
    param_digest,
    preset_digest,
)

GOLDEN = json.loads(GOLDEN_PATH.read_text())
RTOL = 1e-4
ATOL = 1e-5


def assert_params_match(actual, expected):
    assert actual["layout_sha256"] == expected["layout_sha256"]
    assert actual["num_params"] == expected["num_params"]
    np.testing.assert_allclose(actual["leaf_sums"], expected["leaf_sums"], RTOL, ATOL)
    np.testing.assert_allclose(
        actual["leaf_sq_sums"], expected["leaf_sq_sums"], RTOL, ATOL
    )


@pytest.mark.parametrize("name", sorted(GOLDEN["saved_runs"]))
def test_saved_run_rebuilds_same_model(tmp_path, name):
    case = GOLDEN["saved_runs"][name]
    cfg, _, model = build_saved_run(tmp_path, case["files"])

    assert_params_match(param_digest(model), case["params"])
    outputs = model_outputs(cfg, model)
    for key, expected in case["outputs"].items():
        np.testing.assert_allclose(
            outputs[key], expected, RTOL, ATOL, err_msg=f"{name}: {key}"
        )


@pytest.mark.parametrize("name", sorted(GOLDEN["presets"]))
def test_training_preset_builds_same_model(name):
    expected = GOLDEN["presets"][name]
    actual = preset_digest(name)

    assert actual == expected
