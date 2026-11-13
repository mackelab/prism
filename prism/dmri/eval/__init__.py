"""Evaluation utilities for prism."""

from importlib import import_module

from prism import console

__all__ = [
    "console",
    "MetricAggregation",
    "MetricContext",
    "MetricResult",
    "MetricSpec",
    "run_configured_metrics",
]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if name == "console":
        return console
    return getattr(import_module("prism.dmri.eval.export_metrics"), name)
