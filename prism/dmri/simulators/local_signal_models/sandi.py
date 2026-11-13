from prism.dmri.simulators.convolved_models import ConvolvedSignalCompartment
from prism.dmri.simulators.local_signal_models.signal_response_kernels import (
    SimpleSANDIKernel,
)
from prism.dmri.simulators.spherical_distributions import Bingham, Watson


class SandiW(ConvolvedSignalCompartment):
    """Watson distribution with SANDI kernel."""

    fod_type = Watson
    signal_kernel_type = SimpleSANDIKernel


class SandiB(ConvolvedSignalCompartment):
    """Bingham distribution with SANDI kernel."""

    fod_type = Bingham
    signal_kernel_type = SimpleSANDIKernel
