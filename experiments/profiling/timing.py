"""Time and memory of one jitted JAX function: compile, first call, steady state."""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from typing import Any

import jax
import numpy as np

COMPILER_FIELDS = (
    "generated_code_size_in_bytes",
    "argument_size_in_bytes",
    "output_size_in_bytes",
    "alias_size_in_bytes",
    "temp_size_in_bytes",
)


def tree_nbytes(tree: Any) -> int:
    return sum(
        int(leaf.nbytes)
        for leaf in jax.tree_util.tree_leaves(tree)
        if hasattr(leaf, "nbytes")
    )


def device_memory() -> dict[str, int]:
    """Summed integer allocator statistics over devices (empty on CPU)."""
    totals: dict[str, int] = {}
    for device in jax.devices():
        try:
            stats = device.memory_stats()
        except Exception:  # not every backend reports memory
            continue
        for key, value in (stats or {}).items():
            if isinstance(value, (int, np.integer)):
                totals[key] = totals.get(key, 0) + int(value)
    return totals


def profile_function(
    fn: Callable[..., Any],
    args: tuple[Any, ...] = (),
    *,
    name: str,
    warmup: int = 1,
    repeat: int = 10,
) -> dict[str, Any]:
    """Compile ``fn`` ahead of time, then time ``repeat`` blocked calls.

    ``compile_s`` is lowering plus XLA compilation. ``first_exec_s`` is the first
    call of the compiled function; ``steady_state`` summarises the later calls.
    ``peak_bytes_in_use`` is the device allocator's high-water mark after the
    steady-state runs (absent on CPU); ``compiler_memory`` is XLA's own estimate.
    """
    if warmup < 0 or repeat < 1:
        raise ValueError("Need warmup >= 0 and repeat >= 1")
    start = time.perf_counter()
    compiled = jax.jit(fn).lower(*args).compile()
    compile_s = time.perf_counter() - start
    analysis = compiled.memory_analysis()
    compiler_memory = (
        {
            field: int(getattr(analysis, field))
            for field in COMPILER_FIELDS
            if hasattr(analysis, field)
        }
        if analysis is not None
        else None
    )

    start = time.perf_counter()
    output = jax.block_until_ready(compiled(*args))
    first_exec_s = time.perf_counter() - start
    for _ in range(warmup):
        jax.block_until_ready(compiled(*args))
    timings = []
    for _ in range(repeat):
        start = time.perf_counter()
        jax.block_until_ready(compiled(*args))
        timings.append(time.perf_counter() - start)
    memory = device_memory()
    return {
        "name": name,
        "backend": jax.default_backend(),
        "compile_s": compile_s,
        "first_exec_s": first_exec_s,
        "steady_state": {
            "runs": repeat,
            "mean_s": statistics.mean(timings),
            "median_s": statistics.median(timings),
            "std_s": statistics.pstdev(timings),
            "min_s": min(timings),
            "max_s": max(timings),
        },
        "input_nbytes": tree_nbytes(args),
        "output_nbytes": tree_nbytes(output),
        "compiler_memory": compiler_memory,
        "peak_bytes_in_use": memory.get("peak_bytes_in_use"),
    }
