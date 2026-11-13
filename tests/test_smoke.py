import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.parametrize("relative_output", ["smoke", "symbolic/trial1"])
def test_smoke_keeps_outputs_in_requested_directory(tmp_path, relative_output):
    output_dir = tmp_path / relative_output
    repo = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    process = subprocess.run(
        [
            sys.executable,
            str(repo / "examples" / "smoke_polynomial.py"),
            "--output-dir",
            str(output_dir),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    (tmp_path / "smoke.log").write_text(process.stdout + process.stderr)
    assert process.returncode == 0, process.stdout + process.stderr

    result = json.loads((output_dir / "smoke.json").read_text())
    assert result["training_step"] == 20
    assert result["best_step"] == 20
    assert result["weights"] == "ema"
    assert (output_dir / "checkpoints" / "20").is_dir()
    assert (output_dir / "checkpoints" / "best" / "20").is_dir()
    assert (output_dir / ".hydra" / "config.yaml").is_file()
    assert (output_dir / "artifact.yaml").is_file()
    assert not (output_dir.parent / "checkpoints").exists()
    assert not (tmp_path / "results").exists()

    with np.load(output_dir / "samples.npz") as samples:
        for name in ("observation", "theta", "prediction"):
            assert np.isfinite(samples[name]).all()
