"""Import paths written by older saved runs and their current locations.

Saved ``config.yaml`` and ``artifact.yaml`` files name classes as strings.
:func:`resolve_class_path` maps those names to the current code so older
checkpoints, including the published pretrained models, keep loading. The
table holds strings only; nothing here imports an application package.
"""

_DMRI_NN = "prism.dmri.nn"
_PAPER = "prism.symbolic.paper"

# Short names were resolved against the former model module's namespace.
SHORT_NAME_ALIASES = {
    "DMRIInferenceModel": "prism.nn.model.PrismModel",
    "DMRIInferenceModelConfig": f"{_DMRI_NN}.model.DMRIModelConfig",
    "DMRIInferenceModelConfigMaskPriorAmortized": (
        f"{_DMRI_NN}.model.DMRIModelConfigMaskPriorAmortized"
    ),
    "DMRIInferenceModelConfigMaskPriorAmortizedPP": (
        f"{_DMRI_NN}.model.DMRIModelConfigMaskPriorAmortizedPP"
    ),
    "DMRIInferenceModelConfigMaskPriorAmortizedPPP": (
        f"{_DMRI_NN}.model.DMRIModelConfigMaskPriorAmortizedPPP"
    ),
    "SSFPInferenceModelConfig": f"{_DMRI_NN}.model.SSFPModelConfig",
    "DMRIModelSelectionConfig": "prism.nn.autoregressive.ModelSelectionConfig",
    "DMRIModelSelectionAmortizedPriorConfig": (
        "prism.nn.autoregressive.ModelSelectionAmortizedPriorConfig"
    ),
    "DMRIThetaInferenceConfig": "prism.nn.simformer.ThetaInferenceConfig",
    "DMRIEmbeddingConfig": f"{_DMRI_NN}.embedding.DMRIEmbeddingConfig",
    "BvalBvecSignalEmbeddingNet": f"{_DMRI_NN}.embedding.BvalBvecSignalEmbeddingNet",
    "SSFPEmbeddingNetConfig": f"{_DMRI_NN}.embedding.SSFPEmbeddingNetConfig",
    "SSFPEmbeddingNet": f"{_DMRI_NN}.embedding.SSFPEmbeddingNet",
    "SBMIEmbeddingTransformer": "prism.nn.embedding.TransformerEmbedding",
    "DMRITokenizer": f"{_DMRI_NN}.tokenizer.DMRITokenizer",
    "DMRITokenizerPP": f"{_DMRI_NN}.tokenizer.DMRITokenizerPP",
    "DMRITokenizerPPP": f"{_DMRI_NN}.tokenizer.DMRITokenizerPPP",
}

# Classes that moved module or were renamed.
CLASS_ALIASES = {
    **{
        f"prism.nn.dmri_reconstruction_model.{name}": path
        for name, path in SHORT_NAME_ALIASES.items()
    },
    "prism.nn.autoregressive.DMRIModelSelectionConfig": (
        "prism.nn.autoregressive.ModelSelectionConfig"
    ),
    "prism.nn.autoregressive.DMRIModelSelectionAmortizedPriorConfig": (
        "prism.nn.autoregressive.ModelSelectionAmortizedPriorConfig"
    ),
    "prism.nn.simformer.DMRIThetaInferenceConfig": (
        "prism.nn.simformer.ThetaInferenceConfig"
    ),
    "prism.nn.embedding_net.TransformerEmbeddingConfig": (
        "prism.nn.embedding.TransformerEmbeddingConfig"
    ),
    "prism.nn.embedding_net.SBMIEmbeddingTransformer": (
        "prism.nn.embedding.TransformerEmbedding"
    ),
    **{
        f"prism.nn.embedding_net.{name}": f"{_DMRI_NN}.embedding.{name}"
        for name in (
            "DMRIEmbeddingConfig",
            "BvalBvecSignalEmbeddingNet",
            "SSFPEmbeddingNetConfig",
            "SSFPEmbeddingNet",
        )
    },
    **{
        f"prism.nn.tokenizer.{name}": f"{_DMRI_NN}.tokenizer.{name}"
        for name in ("DMRITokenizer", "DMRITokenizerPP", "DMRITokenizerPPP")
    },
    "prism.nn.symbolic_tokenizer.SymbolicTokenizer": (
        "prism.nn.compartment_tokenizer.CompartmentTokenizer"
    ),
    "prism.nn.symbolic_tokenizer.SymbolicTokenizerBase": (
        "prism.nn.compartment_tokenizer.CompartmentTokenizerBase"
    ),
    "prism.nn.symbolic.model.SBMIInferenceModel": f"{_PAPER}.model.SymbolicPaperModel",
    "prism.nn.symbolic.model.SBMIInferenceModelConfig": (
        f"{_PAPER}.model.SymbolicPaperModelConfig"
    ),
    "prism.nn.symbolic.model.SBMIInferenceModelConfigMaskPriorAmortized": (
        f"{_PAPER}.model.SymbolicPaperModelConfigMaskPriorAmortized"
    ),
    "prism.nn.symbolic.autoregressive.SBMIModelSelectionConfig": (
        f"{_PAPER}.autoregressive.SymbolicPaperModelSelectionConfig"
    ),
    "prism.nn.symbolic.autoregressive.SBMIModelSelectionAmortizedPriorConfig": (
        f"{_PAPER}.autoregressive.SymbolicPaperModelSelectionAmortizedPriorConfig"
    ),
    "prism.nn.symbolic.simformer.SBMIThetaInferenceConfig": (
        f"{_PAPER}.simformer.SymbolicPaperThetaInferenceConfig"
    ),
    "prism.nn.symbolic.embedding.SBMIEmbeddingTransformer": (
        f"{_PAPER}.embedding.SymbolicPaperEmbeddingTransformer"
    ),
    "prism.nn.symbolic.embedding.TransformerEmbeddingConfig": (
        f"{_PAPER}.embedding.SymbolicPaperEmbeddingConfig"
    ),
    "prism.nn.symbolic.tokenizer.SymbolicTokenizer": (
        f"{_PAPER}.tokenizer.SymbolicPaperTokenizer"
    ),
    "prism.nn.symbolic.tokenizer.SymbolicTokenizerPP": (
        f"{_PAPER}.tokenizer.SymbolicPaperTokenizerPP"
    ),
}

# Modules that moved without renaming their classes; longest prefix wins.
MODULE_ALIASES = {
    # The package was once called ``dmri``; the published msb3s_2_4_6_128
    # artifact.yaml still names its simulator under that root.
    "dmri": "prism.dmri",
    "prism.simulators.dmri": "prism.dmri.simulators",
    "prism.simulators.symbolic.model_specs": f"{_PAPER}.model_specs",
    "prism.simulators.symbolic.mask_prior": f"{_PAPER}.mask_prior",
    "prism.simulators.symbolic": "prism.symbolic.simulators",
    "prism.nn.symbolic": _PAPER,
}

# ``prism.simulators`` used to re-export the dMRI acquisition factories.
_REEXPORTED_PREFIX = ("prism.simulators.random_", "prism.dmri.simulators.random_")

# Frozen dMRI configs name simulators by class name alone (``sim_type.name``).
LEGACY_SIMULATOR_MODULE = "prism.dmri.simulators.models"


def resolve_class_path(path: str) -> str:
    """Return the current import path for a class path from a saved config."""
    path = str(path)
    if path in SHORT_NAME_ALIASES:
        return SHORT_NAME_ALIASES[path]
    if path in CLASS_ALIASES:
        return CLASS_ALIASES[path]
    old, new = _REEXPORTED_PREFIX
    if path.startswith(old):
        return new + path.removeprefix(old)
    for prefix in sorted(MODULE_ALIASES, key=len, reverse=True):
        if path == prefix or path.startswith(prefix + "."):
            return MODULE_ALIASES[prefix] + path.removeprefix(prefix)
    return path
