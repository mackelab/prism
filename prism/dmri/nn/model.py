"""PRISM model configurations with diffusion MRI tokenizers and encoders.

These presets only change defaults of :class:`prism.nn.model.PrismModelConfig`;
the model class itself is :class:`prism.nn.model.PrismModel`.
"""

from dataclasses import dataclass, field
from typing import Any

from prism.nn.model import (
    PrismModelConfig,
    PrismModelConfigMaskPriorAmortized,
    TokenizerType,
)

from .embedding import (
    BvalBvecSignalEmbeddingNet,
    DMRIEmbeddingConfig,
    SSFPEmbeddingNet,
    SSFPEmbeddingNetConfig,
)
from .tokenizer import DMRITokenizer, DMRITokenizerPP, DMRITokenizerPPP


@dataclass
class DMRIModelConfig(PrismModelConfig):
    tokenizer_cls: TokenizerType = DMRITokenizer
    embedding_cls: type = BvalBvecSignalEmbeddingNet
    embedding_cfg: Any = field(default_factory=DMRIEmbeddingConfig)


@dataclass
class DMRIModelConfigMaskPriorAmortized(PrismModelConfigMaskPriorAmortized):
    tokenizer_cls: TokenizerType = DMRITokenizer
    embedding_cls: type = BvalBvecSignalEmbeddingNet
    embedding_cfg: Any = field(default_factory=DMRIEmbeddingConfig)


@dataclass
class DMRIModelConfigMaskPriorAmortizedPP(DMRIModelConfigMaskPriorAmortized):
    tokenizer_cls: TokenizerType = DMRITokenizerPP


@dataclass
class DMRIModelConfigMaskPriorAmortizedPPP(DMRIModelConfigMaskPriorAmortized):
    """Architecture used by early multishell pretrained checkpoints."""

    tokenizer_cls: TokenizerType = DMRITokenizerPPP


@dataclass
class SSFPModelConfig(DMRIModelConfig):
    tokenizer_cls: TokenizerType = DMRITokenizerPP
    embedding_cls: type = SSFPEmbeddingNet
    embedding_cfg: Any = field(default_factory=SSFPEmbeddingNetConfig)
