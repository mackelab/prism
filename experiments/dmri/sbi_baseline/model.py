"""Transformer encoder and neural spline flow over the dMRI parameters.

Requires ``torch`` and ``sbi``. The flow is ``sbi``'s neural spline flow
conditioned on an embedding of the measured signal and its acquisition.

A condition is one tensor per voxel: the signal, the b-values and the three
gradient components of every measurement, packed side by side. Its width is
``5 * number_of_measurements``, so one flow serves acquisitions of different
lengths.
"""

from __future__ import annotations

import math

import torch
from sbi.neural_nets.net_builders.flow import build_nsf
from torch import Tensor, nn

ROWS_PER_MEASUREMENT = 5  # signal, b-value, three gradient components


def pack_condition(signal: Tensor, bvals: Tensor, bvecs: Tensor) -> Tensor:
    """Pack ``[B, S]`` signals, ``[B, S]`` b-values and ``[B, S, 3]`` gradients."""
    return torch.cat([signal, bvals, bvecs.reshape(bvecs.shape[0], -1)], dim=1)


def unpack_condition(condition: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Inverse of :func:`pack_condition`."""
    batch, width = condition.shape
    if width % ROWS_PER_MEASUREMENT:
        raise ValueError(f"Condition width {width} is not a multiple of 5.")
    n = width // ROWS_PER_MEASUREMENT
    bvecs = condition[:, 2 * n :].reshape(batch, n, 3)
    return condition[:, :n], condition[:, n : 2 * n], bvecs


class GaussianFourierFeatures(nn.Module):
    """Sine and cosine features of a scalar at random Gaussian frequencies."""

    def __init__(self, out_dim: int, scale: float = 1.0) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.register_buffer("frequencies", torch.randn(math.ceil(out_dim / 2)) * scale)

    def forward(self, x: Tensor) -> Tensor:
        phases = 2.0 * math.pi * x * self.frequencies
        return torch.cat([torch.sin(phases), torch.cos(phases)], dim=-1)[
            ..., : self.out_dim
        ]


class DMRIEncoder(nn.Module):
    """Transformer over the measurements with learnable summary tokens.

    Every measurement becomes a token from its Fourier-embedded b-value, its
    squashed signal and its gradient direction. The outputs of the summary
    tokens are concatenated and projected to ``output_dim``.
    """

    def __init__(
        self,
        model_dim: int = 128,
        output_dim: int = 512,
        num_cls_tokens: int = 4,
        num_layers: int = 4,
        num_heads: int = 8,
        widening_factor: int = 8,
        feature_dim: int = 16,
        bvec_repeats: int = 4,
        max_bval: float = 4000.0,
        squash_temperature: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_cls_tokens = num_cls_tokens
        self.feature_dim = feature_dim
        self.bvec_repeats = bvec_repeats
        self.max_bval = max_bval
        self.squash_temperature = squash_temperature
        self.embed_bvals = GaussianFourierFeatures(feature_dim)
        token_dim = 2 * feature_dim + 3 * bvec_repeats
        self.input_projection = nn.Linear(token_dim, model_dim)
        self.cls_tokens = nn.Parameter(torch.randn(1, num_cls_tokens, model_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            model_dim,
            num_heads,
            dim_feedforward=model_dim * widening_factor,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(model_dim)
        self.output_projection = nn.Linear(model_dim * num_cls_tokens, output_dim)

    def tokens(self, signal: Tensor, bvals: Tensor, bvecs: Tensor) -> Tensor:
        bvals = bvals.clamp(0.0, self.max_bval) / self.max_bval * 2.0 - 1.0
        squashed = torch.sigmoid((signal - 0.5) / self.squash_temperature)
        features = torch.cat(
            [
                self.embed_bvals(bvals.unsqueeze(-1)),
                squashed.unsqueeze(-1).repeat_interleave(self.feature_dim, dim=-1),
                bvecs.repeat_interleave(self.bvec_repeats, dim=-1),
            ],
            dim=-1,
        )
        return self.input_projection(features)

    def forward(self, condition: Tensor) -> Tensor:
        tokens = self.tokens(*unpack_condition(condition))
        summary = self.cls_tokens.expand(tokens.shape[0], -1, -1)
        encoded = self.transformer(torch.cat([summary, tokens], dim=1))
        pooled = self.output_norm(encoded[:, : self.num_cls_tokens])
        return self.output_projection(pooled.flatten(1))


def build_flow(
    theta: Tensor,
    condition: Tensor,
    *,
    model_dim: int = 128,
    output_dim: int = 512,
    num_cls_tokens: int = 4,
    num_layers: int = 4,
    hidden_features: int = 256,
    num_transforms: int = 8,
    num_bins: int = 10,
    tail_bound: float = 3.0,
) -> nn.Module:
    """Build the conditional flow; ``theta`` and ``condition`` fix the shapes.

    The returned ``nflows`` flow provides ``log_prob(theta, context=condition)``
    and ``sample(n, context=condition)`` (shape ``[batch, n, parameters]``) for
    conditions of any width.
    """
    encoder = DMRIEncoder(model_dim, output_dim, num_cls_tokens, num_layers)
    estimator = build_nsf(
        theta,
        condition,
        z_score_x="independent",
        z_score_y="none",
        embedding_net=encoder,
        hidden_features=hidden_features,
        num_transforms=num_transforms,
        num_bins=num_bins,
        tail_bound=tail_bound,
    )
    return estimator.net.to(theta.device)


def negative_log_likelihood(
    flow: nn.Module, theta: Tensor, condition: Tensor
) -> Tensor:
    """Per-voxel negative log-density of ``theta`` under the flow."""
    return -flow.log_prob(theta, context=condition)
