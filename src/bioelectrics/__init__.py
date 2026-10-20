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
from .gap_junctions import GapJunctionProfile, gj_conductance, grid_conductances
from .integrators import euler_step, integrate, rk4_step
from .tissue import BioelectricState, Tissue, tumor_prepattern
from .control import OptimizationResult, PatternOptimizer

__version__ = "0.1.0"
