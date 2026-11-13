"""Class names written by older saved runs resolve to current code."""

import pytest
from hydra.utils import get_class
from omegaconf import OmegaConf

from prism.config.aliases import (
    CLASS_ALIASES,
    LEGACY_SIMULATOR_MODULE,
    SHORT_NAME_ALIASES,
    resolve_class_path,
)
from prism.dmri.simulators.acquisition_scheme import random_hcp_acquisition
from prism.dmri.simulators.training import resolve_acquisition_factories
from prism.simulators.config import resolve_simulator_model


@pytest.mark.parametrize(
    "target", sorted({*SHORT_NAME_ALIASES.values(), *CLASS_ALIASES.values()})
)
def test_every_alias_target_imports(target):
    assert isinstance(get_class(target), type)


@pytest.mark.parametrize(
    "old,new",
    [
        ("DMRIInferenceModelConfigMaskPriorAmortizedPP", "prism.dmri.nn.model"),
        ("prism.nn.dmri_reconstruction_model.DMRIInferenceModel", "prism.nn.model"),
        ("prism.nn.embedding_net.SBMIEmbeddingTransformer", "prism.nn.embedding"),
        ("prism.nn.symbolic.model.SBMIInferenceModel", "prism.symbolic.paper.model"),
        (
            "prism.simulators.dmri.models.Ball3StickSharedDiffusivity",
            "prism.dmri.simulators.models",
        ),
        (
            "prism.simulators.symbolic.model_specs.SymbolicModelV0",
            "prism.symbolic.paper.model_specs",
        ),
        ("prism.simulators.symbolic.SimplePolynomial", "prism.symbolic.simulators"),
        # Written by the published msb3s_2_4_6_128 artifact.yaml.
        (
            "dmri.simulators.models.MultiShellBall3StickSharedDiffusivity",
            "prism.dmri.simulators.models",
        ),
    ],
)
def test_old_paths_resolve_to_current_modules(old, new):
    resolved = resolve_class_path(old)
    assert resolved.rpartition(".")[0] == new
    assert isinstance(get_class(resolved), type)


def test_current_paths_are_unchanged():
    path = "prism.nn.model.PrismModel"
    assert resolve_class_path(path) == path
    assert resolve_class_path("prism.nn.symbolic_like.Thing") == (
        "prism.nn.symbolic_like.Thing"
    )
    assert resolve_class_path("dmri_like.Thing") == "dmri_like.Thing"


def test_frozen_simulator_names_resolve():
    cfg = OmegaConf.create({"simulator": {"sim_type": {"name": "Ball2Stick"}}})
    model_class = resolve_simulator_model(cfg)
    assert model_class.__module__ == LEGACY_SIMULATOR_MODULE


def test_acquisition_targets_from_older_configs_resolve():
    cfg = OmegaConf.create({
        "simulator": {
            "acquisitions": [
                {
                    "_target_": "prism.simulators.random_hcp_acquisition",
                    "_partial_": True,
                }
            ]
        }
    })
    (factory,) = resolve_acquisition_factories(cfg)
    assert factory.func is random_hcp_acquisition


def test_simulator_without_interface_is_rejected():
    cfg = OmegaConf.create({"simulator": {"model_class": "collections.OrderedDict"}})
    with pytest.raises(TypeError, match="simulator interface"):
        resolve_simulator_model(cfg)
