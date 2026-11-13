"""SBI baseline command line: works without torch and resumes the saved network."""

import subprocess
import sys

import pytest

from experiments.dmri.sbi_baseline import train


def test_scripts_import_and_print_help_without_torch():
    # --help used to fail with "No module named 'torch'".
    script = (
        "import sys\n"
        "import experiments.dmri.sbi_baseline.train, experiments.dmri.sbi_baseline.evaluate\n"
        "assert 'torch' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", script], check=True)


def test_resume_reuses_the_saved_architecture():
    # Resuming without repeating every flag used to build the default network
    # and fail on the checkpoint's shapes, after overwriting config.json.
    parser = train.build_parser()
    args = parser.parse_args(["--output", "run", "--resume", "--num-steps", "10"])
    saved = {
        "model_dim": 16,
        "num_layers": 1,
        "sim_type": "Ball3StickSharedDiffusivity",
    }
    train.apply_saved_architecture(args, saved, parser)
    assert (args.model_dim, args.num_layers, args.num_steps) == (16, 1, 10)


def test_resume_rejects_a_different_architecture():
    parser = train.build_parser()
    args = parser.parse_args(["--output", "run", "--resume", "--model-dim", "32"])
    with pytest.raises(ValueError, match="--model-dim 32 differs"):
        train.apply_saved_architecture(args, {"model_dim": 16}, parser)
