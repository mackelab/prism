from dataclasses import dataclass
from typing import Any

from flax import nnx
from huggingface_hub import HfApi

# Defaults point at the published diffusion MRI checkpoints.
DEFAULT_REPO_ID = "manugloeck/dmri-pretrained"
DEFAULT_MODEL_NAME = "msb3s_2_4_6_128"


@dataclass(frozen=True)
class PretrainedModel:
    model: Any
    simulator: Any
    config: Any
    step: int | None
    params_name: str = "params_ema"


def load_checkpoint(**kwargs):
    """Load the heavy checkpoint implementation only when a model is requested."""
    from prism.train.utils import load_checkpoint as load

    return load(**kwargs)


def load_pretrained(
    model_name=DEFAULT_MODEL_NAME,
    repo_id=DEFAULT_REPO_ID,
    which="best",
    *,
    revision=None,
    cache_dir=None,
    token=None,
    local_files_only=False,
    precision=None,
    params_name="params_ema",
) -> PretrainedModel:
    """Load a pretrained PRISM model from the Hugging Face Hub.

    Parameters
    ----------
    model_name:
        Checkpoint name inside the repository.
    repo_id:
        Hugging Face model repository identifier.
    which:
        Which checkpoint to load ("latest", "best", or a step number).
    revision:
        Branch, tag, or commit hash to pin.
    cache_dir:
        Local directory for downloaded artifacts.
    token:
        Hugging Face access token for private repos.
    local_files_only:
        Skip any remote download and use the cache exclusively.
    precision:
        Network-forward precision (e.g. "fp16", "bf16"); parameters stay
        float32. Half precision falls back to fp32 on CPU.
    params_name:
        Parameter set to use: "params_ema" (default) or "params" for raw
        weights. Raises ValueError if the selected weights are unavailable.

    Returns
    -------
    PretrainedModel
        The instantiated model, simulator, config, training step, and parameter set.
    """
    from prism.train.utils import select_checkpoint_params

    checkpoint, model, simulator = load_checkpoint(
        repo_id=repo_id,
        model_name=model_name,
        which=which,
        revision=revision,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
        precision=precision,
    )
    params = select_checkpoint_params(checkpoint, params_name)
    nnx.update(model, params)
    model.eval()

    return PretrainedModel(
        model=model,
        simulator=simulator,
        config=getattr(model, "cfg", None),
        step=checkpoint.get("step"),
        params_name=params_name,
    )


def list_pretrained_models(
    repo_id=DEFAULT_REPO_ID,
    *,
    revision=None,
    token=None,
) -> list[str]:
    """Return the names of pretrained models available in a repository.

    Parameters
    ----------
    repo_id:
        Hugging Face model repository identifier.
    revision:
        Branch, tag, or commit hash to inspect.
    token:
        Hugging Face access token for private repos.

    Returns
    -------
    list[str]
        Sorted model names found in the repository.
    """
    files = HfApi(token=token).list_repo_files(
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
    )
    return sorted(
        path.split("/", 1)[0]
        for path in files
        if path.count("/") == 1 and path.endswith("/config.yaml")
    )
