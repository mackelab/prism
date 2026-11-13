from prism.dmri.simulators.convolved_models import ConvolvedSignalCompartment
from prism.dmri.simulators.local_signal_models.signal_response_kernels import (
    ZeppelinKernel,
)
from prism.dmri.simulators.spherical_distributions import Bingham, Watson


class WatsonZeppelin(ConvolvedSignalCompartment):
    """Watson distribution with zeppelin kernel."""

    fod_type = Watson
    signal_kernel_type = ZeppelinKernel


class BinghamZeppelin(ConvolvedSignalCompartment):
    """Bingham distribution with zeppelin kernel."""

    fod_type = Bingham
    signal_kernel_type = ZeppelinKernel
