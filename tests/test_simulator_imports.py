"""Canonical simulator modules, packaged data, and config targets resolve."""

import importlib
import pickle
from importlib.resources import files

import pytest
from omegaconf import OmegaConf


@pytest.mark.parametrize(
    ("module", "name"),
    [
        ("prism.dmri.simulators.models", "Ball3StickNoise"),
        ("prism.dmri.simulators.local_signal_models.ball", "Ball"),
        ("prism.dmri.simulators.spherical_distributions", "Uniform"),
        ("prism.symbolic.simulators", "SimplePolynomial"),
        ("prism.symbolic.simulators.base", "AdditiveModel"),
        ("prism.dmri.simulators.base", "SignalCompartment"),
        ("prism.dmri.simulators.base", "NoiseCompartment"),
        ("prism.dmri.simulators.base", "SharedParameterState"),
        ("prism.simulators.base", "Compartment"),
    ],
)
def test_pickle_roundtrip_records_the_canonical_module(module, name):
    cls = getattr(importlib.import_module(module), name)
    assert cls.__module__ == module
    assert pickle.loads(pickle.dumps(cls)) is cls


def test_frozen_dmri_model_and_acquisition_targets_resolve():
    from prism.dmri.simulators.acquisition_scheme import random_hcp_acquisition
    from prism.dmri.simulators.models import Ball3StickNoise
    from prism.dmri.simulators.training import resolve_acquisition_factories
    from prism.simulators.config import resolve_simulator_model, simulator_model_config

    cfg = OmegaConf.create({
        "simulator": {
            "model_class": "prism.dmri.simulators.models.Ball3StickNoise",
            "acquisitions": [
                {
                    # Resolved through the re-exports in ``prism.simulators``.
                    "_target_": "prism.dmri.simulators.random_hcp_acquisition",
                    "_partial_": True,
                    "num_acquisitions": 105,
                }
            ],
        }
    })
    assert resolve_simulator_model(cfg) is Ball3StickNoise
    assert simulator_model_config(cfg)["model_class"] == (
        "prism.dmri.simulators.models.Ball3StickNoise"
    )
    assert resolve_acquisition_factories(cfg)[0].func is random_hcp_acquisition


def test_symbolic_package_ships_its_binary_model_data():
    asset = files("prism.symbolic.paper").joinpath("data/binary_model_0.npy")
    assert asset.read_bytes()
