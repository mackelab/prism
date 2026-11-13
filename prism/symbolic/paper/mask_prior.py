"""The empirical model-mask distribution used for Figure 4b.

Adapted from mackelab/sbmi_transformer_symbolic (MIT), c8e118b8. The order
and multiplicity of the stored rows are part of the sampling distribution.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from prism.simulators.mask_prior import MaskPrior


class SymbolicMaskPrior(MaskPrior):
    """Sample the paper's twenty-bit masks without a hyperparameter input."""

    mask_prior_dim = 0

    def __init__(self, num_model_components: int, num_noise_components: int):
        super().__init__(num_model_components, num_noise_components)
        if (num_model_components, num_noise_components) != (15, 5):
            raise ValueError(
                "SymbolicMaskPrior requires 15 model and 5 noise components"
            )
        directory = Path(__file__).parent / "data"
        masks = [
            np.load(directory / f"binary_model_{index}.npy", allow_pickle=False)
            for index in range(10)
        ]
        if any(mask.shape != (100_000, 20) or mask.dtype != np.bool_ for mask in masks):
            raise ValueError(
                "Invalid Figure 4b mask table: expected ten 100000x20 Boolean arrays"
            )
        self._mask_support = jnp.asarray(np.concatenate(masks, axis=0))
        self._num_masks = self._mask_support.shape[0]

    def sample_model_mask(self, rng, hyperparameters=None):
        return jax.random.choice(rng, self._mask_support, shape=(), axis=0)

    def sample_hyperparameters(self, rng):
        return None

    def log_prob_model_mask(self, model_mask, hyperparameters=None):
        """Preserve the frozen source's lookup score, not duplicate-adjusted mass.

        Figure 4b uses only sampling from this prior. This legacy score assigns
        1/N to a supported mask even when the table contains repeated rows.
        """
        mask = jnp.asarray(model_mask, dtype=jnp.bool_)
        matches = jnp.all(self._mask_support == mask[None, :], axis=-1)
        return jnp.where(jnp.any(matches), -jnp.log(self._num_masks), -jnp.inf)
