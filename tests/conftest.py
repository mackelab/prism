import jax
import pytest
from jax import config

config.update("jax_platform_name", "cpu")
# Importing the dMRI eval script points JAX at an on-disk compilation cache.
# Tests should not read or write cache state shared across tests and runs.
config.update("jax_enable_compilation_cache", False)


@pytest.fixture(autouse=True, scope="module")
def _release_compiled_executables():
    """Free compiled programs after each test module.

    Every compiled XLA CPU executable keeps memory mappings alive. Without
    clearing, the whole suite exceeds the kernel limit (``vm.max_map_count``,
    65530 by default) in one process and XLA segfaults in whatever it maps next.
    """
    yield
    jax.clear_caches()
