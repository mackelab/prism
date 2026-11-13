"""Calibration entry points and compatibility for saved experiment targets."""

import sys

from experiments.dmri._shared import acquisition, model

from . import chunks

sys.modules[__name__ + ".acquisition"] = acquisition
sys.modules[__name__ + ".model"] = model
sys.modules[__name__ + ".storage"] = chunks
