"""bioelectrics: Differentiable Somatic Bioelectricity Engine."""

from . import constants
from .channels import (
    ChannelProfile,
    find_fixed_points,
    ghk_current,
    i_dep,
    i_leak,
    i_pol,
    ionic_current,
    membrane_rhs,
)
from .integrators import euler_step, integrate, rk4_step

__version__ = "0.1.0"
