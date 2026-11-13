import os
import sys
from importlib import import_module

# XLA writes compiler diagnostics directly to the native stderr file descriptor,
# bypassing Python and Hydra logging. Keep the user-facing CLI readable unless a
# backend developer explicitly asks for the raw stream.
if os.environ.get("PRISM_SHOW_NATIVE_LOGS", "").lower() not in {"1", "true", "yes"}:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

USAGE = """usage: {prog} <command> [options]

Commands:
  train     Train a model using Hydra configuration
  simulate  Save training and validation datasets for file-based training
  predict   Run a pretrained dMRI model on a standard dMRI folder
  eval      Run the dMRI volumetric evaluation pipeline

Run `{prog} <command> --help` for command-specific help.
"""

# The CLI composes the method with its applications. Commands are imported by
# path so importing the core package never loads an application package.
COMMANDS = {
    "train": "prism.train.train_script",
    "simulate": "prism.train.simulate",
    "predict": "prism.dmri.predict",
    "eval": "prism.dmri.eval.eval_script",
}


def main(prog=None):
    """Dispatch the public command-line interface."""
    if prog is None:
        prog = "prism"
    usage = USAGE.format(prog=prog)
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(usage)
        return

    command = sys.argv.pop(1)
    if command not in COMMANDS:
        print(f"Unknown command: {command}\n", file=sys.stderr)
        print(usage, file=sys.stderr)
        raise SystemExit(2)

    import_module(COMMANDS[command]).main()
