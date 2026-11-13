"""Submit an experiment script as a Slurm job using the Hydra partition presets.

Scripts add ``--slurm PARTITION`` with :func:`add_slurm_arguments` and call
:func:`submit` after parsing. The job re-runs the script's ``main`` with the same
arguments minus the Slurm flags. Resources come from
``conf/infrastructure/launcher/slurm.yaml`` merged with
``conf/infrastructure/partition/PARTITION.yaml``, as for ``prism train``.
"""

import argparse
import importlib
import os
import sys
from pathlib import Path

from omegaconf import OmegaConf

INFRASTRUCTURE = Path(__file__).resolve().parents[2] / "conf" / "infrastructure"
SLURM_FLAGS = ("--slurm", "--slurm-timeout-min")


def partitions():
    return sorted(
        path.stem
        for path in (INFRASTRUCTURE / "partition").glob("*.yaml")
        if path.stem != "none"
    )


def add_slurm_arguments(parser):
    parser.add_argument(
        "--slurm",
        metavar="PARTITION",
        choices=partitions(),
        help="submit this command as a Slurm job with the resources of "
        "conf/infrastructure/partition/PARTITION.yaml",
    )
    parser.add_argument(
        "--slurm-timeout-min",
        type=int,
        help="override the partition preset's scheduler time limit",
    )


def launcher_settings(partition, **overrides):
    """The merged ``hydra.launcher`` settings of the Slurm launcher and a preset."""
    base = OmegaConf.load(INFRASTRUCTURE / "launcher" / "slurm.yaml").hydra.launcher
    preset = OmegaConf.load(INFRASTRUCTURE / "partition" / f"{partition}.yaml")
    merged = OmegaConf.merge(base, preset.hydra.launcher)
    settings = OmegaConf.to_container(merged, resolve=False)
    settings.update({k: v for k, v in overrides.items() if v is not None})
    return settings


def submitit_parameters(settings):
    """Split launcher settings into submitit executor and update parameters.

    Mirrors ``hydra_submitit_launcher``: queue-independent keys pass through,
    everything else gets the ``slurm_`` prefix.
    """
    from hydra_plugins.hydra_submitit_launcher.config import BaseQueueConf

    shared = set(OmegaConf.structured(BaseQueueConf).keys())
    settings = {
        k: v for k, v in settings.items() if k not in ("_target_", "submitit_folder")
    }
    init = {}
    if "max_num_timeout" in settings:
        init["slurm_max_num_timeout"] = settings.pop("max_num_timeout")
    update = {k if k in shared else f"slurm_{k}": v for k, v in settings.items()}
    return init, update


def strip_slurm_arguments(argv):
    """Remove the Slurm flags (and their values) from an argument list."""
    stripped, skip = [], False
    for arg in argv:
        if skip:
            skip = False
        elif arg in SLURM_FLAGS:
            skip = True
        elif not arg.startswith(tuple(f"{flag}=" for flag in SLURM_FLAGS)):
            stripped.append(arg)
    return stripped


def _module_name(entry):
    module = entry.__module__
    if module == "__main__":
        spec = sys.modules["__main__"].__spec__
        if spec is None:
            raise SystemExit("--slurm requires running the script with `python -m`.")
        module = spec.name
    return module


def _executor(partition, name, timeout_min):
    import submitit

    init, update = submitit_parameters(
        launcher_settings(partition, name=name, timeout_min=timeout_min)
    )
    folder = Path("results") / "slurm" / "%j"
    executor = submitit.AutoExecutor(folder=folder, cluster="slurm", **init)
    executor.update_parameters(**update)
    return executor


def run_in_job(module, argv):
    """Job-side entry point: require a GPU backend, then run ``module.main``."""
    import jax

    if not os.environ.get("SLURM_JOB_ID") or jax.default_backend() != "gpu":
        raise RuntimeError("The Slurm job has no active JAX GPU backend.")
    return importlib.import_module(module).main(argv)


def submit(args, argv, entry):
    """Submit ``entry`` (a script's ``main``) with ``argv`` minus the Slurm flags."""
    argv = strip_slurm_arguments(sys.argv[1:] if argv is None else list(argv))
    module = _module_name(entry)
    executor = _executor(args.slurm, module, args.slurm_timeout_min)
    job = executor.submit(run_in_job, module, argv)
    print(f"Submitted Slurm job {job.job_id}; logs in {job.paths.folder}")
    return job


def report():
    import jax

    return {
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
    }


def check(partition, timeout_min):
    """Submit a tiny job and wait for its JAX backend report."""
    job = _executor(partition, "prism-slurm-check", timeout_min).submit(report)
    print(f"Submitted Slurm job {job.job_id}; waiting for it to finish")
    result = job.result()
    print(result)
    if result["backend"] != "gpu":
        raise SystemExit("The job ran without a JAX GPU backend.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check Slurm submission.")
    parser.add_argument("--check", metavar="PARTITION", choices=partitions())
    parser.add_argument("--timeout-min", type=int, default=10)
    cli = parser.parse_args()
    if cli.check is None:
        parser.error("--check PARTITION is required")
    # Call through the importable module so the pickled job function does not
    # refer to `__main__`, which the job process cannot import.
    from prism.utils import slurm

    slurm.check(cli.check, cli.timeout_min)
