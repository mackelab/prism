import importlib
from functools import partial

from flax import nnx
from hydra.utils import get_class
from omegaconf import DictConfig

from prism.config.aliases import resolve_class_path
from prism.nn.model import PrismModel
from prism.simulators.config import mask_prior_overrides


def _resolve_class(name, module):
    """Resolve an import path, a legacy name, or a name in ``prism.nn.model``."""
    name = resolve_class_path(name)
    return get_class(name) if "." in name else getattr(module, name)


def build_model(cfg: DictConfig, sim_type):
    name_embed = cfg.model.embedding_net.name
    params_embed = cfg.model.embedding_net.params
    cls_embed_name = cfg.model.embedding_net.get("cls", None)
    name_model_selection_net = cfg.model.model_selection_net.name
    params_model_selection_net = cfg.model.model_selection_net.params
    name_inference_net = cfg.model.inference_net.name
    params_inference_net = cfg.model.inference_net.params
    module_embed = importlib.import_module("prism.nn.model")

    cfg_embed = _resolve_class(name_embed, module_embed)(**params_embed)
    embedding_cls = None
    if cls_embed_name is not None:
        try:
            embedding_cls = _resolve_class(cls_embed_name, module_embed)
        except (AttributeError, ImportError) as error:
            raise ValueError(
                f"Unknown embedding class '{cls_embed_name}' requested in config."
            ) from error
        cls_embed_params = dict(cfg.model.embedding_net.get("cls_params", {}) or {})
        if cfg.model.get("architecture") is not None:
            input_dim = cfg.simulator.get("input_dim")
            if input_dim is not None:
                cls_embed_params["input_dim"] = input_dim
        if cls_embed_params:
            embedding_cls = partial(embedding_cls, **cls_embed_params)
    cfg_model_selection_net = _resolve_class(name_model_selection_net, module_embed)(
        **params_model_selection_net
    )
    if getattr(cfg_model_selection_net, "precondition_with_mask_prior", False):
        cfg_model_selection_net.prior_mask = sim_type.create_mask_prior(
            **mask_prior_overrides(cfg.simulator)
        )
    cfg_inference_net = _resolve_class(name_inference_net, module_embed)(
        **params_inference_net
    )
    name_model = cfg.model.name
    cfg_class = _resolve_class(name_model, module_embed)
    precision_keys = (
        "dtype",
        "param_dtype",
        "precision",
        "preferred_element_type",
    )

    precision_kwargs = {
        key: cfg.model.get(key)
        for key in precision_keys
        if key in cfg.model and cfg.model.get(key) is not None
    }
    # Preserve legacy spelling if present.
    legacy_preferred = cfg.model.get("prefered_element_type", None)
    if (
        "preferred_element_type" not in precision_kwargs
        and legacy_preferred is not None
    ):
        precision_kwargs["preferred_element_type"] = legacy_preferred

    cfg_kwargs = dict(
        model_dim=cfg.model.model_dim,
        use_attention_mask=cfg.model.use_attention_mask,
        embedding_cfg=cfg_embed,
        model_selection_cfg=cfg_model_selection_net,
        theta_inference_cfg=cfg_inference_net,
        **precision_kwargs,
    )
    if embedding_cls is not None:
        cfg_kwargs["embedding_cls"] = embedding_cls
    if cfg.model.get("tokenizer_cls") is not None:
        cfg_kwargs["tokenizer_cls"] = _resolve_class(
            cfg.model.tokenizer_cls, module_embed
        )
    if cfg.model.get("inference_loss_type") is not None:
        cfg_kwargs["inference_loss_type"] = cfg.model.inference_loss_type
    cfg_m = cfg_class(
        sim_type,
        **cfg_kwargs,
    )
    architecture = cfg.model.get("architecture")
    model_class = (
        get_class(resolve_class_path(architecture))
        if architecture is not None
        else PrismModel
    )
    model = model_class(cfg_m, nnx.Rngs(cfg.model.init_seed))

    return model
