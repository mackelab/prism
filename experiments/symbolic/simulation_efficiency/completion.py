"""Completion and diffusion checks retained from the larger Figure 4b runs."""

import hashlib
import json
import math
from pathlib import Path

from omegaconf import OmegaConf

from experiments.symbolic.simulation_efficiency.evaluate import (
    FIG4B as REFERENCES,
)
from experiments.symbolic.simulation_efficiency.evaluate import (
    LARGER_POINTS,
)


def read_completion(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    record = json.loads(raw)
    if not isinstance(record, dict):
        raise RuntimeError("Training completion must be a JSON object")
    # Reject non-finite metadata before performing any checkpoint inference.
    json.dumps(record, allow_nan=False)
    return record, {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "record": record,
    }


def validate_completion(record, cfg, point, completed_step, run_dir):
    """Require an explicit successful process stop and a matching saved step."""
    required = {
        "point",
        "run_dir",
        "process_exit_code",
        "stop_reason",
        "completed_training_step",
        "configured_max_train_steps",
        "configured_max_train_hours",
        "elapsed_training_seconds",
    }
    missing = required - record.keys()
    if missing:
        raise RuntimeError(f"Training completion is missing fields: {sorted(missing)}")
    if point not in LARGER_POINTS or record["point"] != point:
        raise RuntimeError("Training completion refers to a different Figure 4b point")
    if Path(record["run_dir"]).resolve() != Path(run_dir).resolve():
        raise RuntimeError("Training completion refers to a different run directory")
    if type(record["process_exit_code"]) is not int or record["process_exit_code"] != 0:
        raise RuntimeError("Training process did not finish successfully")
    if (
        type(record["completed_training_step"]) is not int
        or record["completed_training_step"] != completed_step
    ):
        raise RuntimeError("Training completion disagrees with the latest saved step")
    steps = OmegaConf.select(cfg, "training.max_train_steps")
    hours = OmegaConf.select(cfg, "training.max_train_hours")
    if (
        record["configured_max_train_steps"] != steps
        or record["configured_max_train_hours"] != hours
    ):
        raise RuntimeError("Training completion budget disagrees with the saved config")
    elapsed = record["elapsed_training_seconds"]
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(elapsed)
        or elapsed <= 0
    ):
        raise RuntimeError("Training completion has invalid elapsed time")
    expected_cap = REFERENCES[point][1]
    if expected_cap is not None:
        if (
            steps != expected_cap
            or completed_step != expected_cap
            or record["stop_reason"] != "step_limit"
        ):
            raise RuntimeError(
                f"The {point} run must finish exactly {expected_cap} optimizer steps"
            )
    else:
        if steps is not None or hours != 8 or record["stop_reason"] != "time_limit":
            raise RuntimeError(
                f"The {point} run must finish its configured eight-hour training budget"
            )
        if completed_step < 2000:
            raise RuntimeError(
                "Time-limited training did not reach its first validation/checkpoint interval"
            )
        if elapsed < hours * 3600:
            raise RuntimeError(
                "Training process elapsed time is shorter than its configured time limit"
            )


def validate_model_bounds(model):
    """Check the installed probjax defaults instead of assuming their values."""
    decoder = model.inference_decoder
    for item in (decoder.train_cfg, decoder.schedule, decoder.solver_cfg.schedule):
        if not math.isclose(
            float(item.t_min), 2e-4, rel_tol=0, abs_tol=1e-10
        ) or not math.isclose(float(item.t_max), 80.0):
            raise RuntimeError(
                "Restored decoder does not use the frozen diffusion bounds [2e-4, 80]"
            )
    if decoder.solver_cfg.num_steps != 64 or decoder.loss_type != "v":
        raise RuntimeError(
            "Restored decoder must use 64 diffusion steps and v-prediction"
        )
    if decoder.solver_cfg.ode_method != "heun" or decoder.solver_cfg.rho != 7.0:
        raise RuntimeError(
            "Restored decoder must use the frozen EDM Heun solver with rho=7"
        )
    if (
        decoder.train_cfg.lognoise_mean != -1.2
        or decoder.train_cfg.lognoise_scale != 1.2
    ):
        raise RuntimeError(
            "Restored decoder does not use the frozen lognormal training-noise distribution"
        )
