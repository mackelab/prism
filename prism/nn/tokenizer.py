"""Base class for tokenizers that embed model structure and parameters."""

from abc import abstractmethod
from typing import Any

from flax import nnx
from probjax.utils.typing import Array


class Tokenizer(nnx.Module):
    def __call__(self, *args: Any, **kwds: Any) -> Array:
        return self.encode(*args, **kwds)

    @abstractmethod
    def encode(self, *args: Any, **kwargs: Any) -> Array:
        pass

    @abstractmethod
    def decode(self, *args: Any, **kwargs: Any) -> Array:
        pass
