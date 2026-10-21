"""bioelectrics: a differentiable engine for somatic bioelectricity.

Subpackages and modules
-----------------------
* ``bioelectrics.ionic``: ion-level physics (Na+, K+, Cl-, Ca2+; GHK fluxes, pumps,
  Hodgkin-Huxley channels, electrodiffusive gap junctions), BETSE-compatible.
* ``bioelectrics.interventions``: channel misexpression, drugs, gap junction blockers,
  optogenetics, and their inverse design (``InterventionOptimizer``,
  ``ChronicInterventionOptimizer``).
* ``bioelectrics.attractors``: steady states, stability, basins, dose ramps, and
  implicit gradients of steady states.
* ``bioelectrics.pharmacology``: drugs with published potency values, each carrying its
  assay, species and citation.
* ``bioelectrics.zoo``: published models with reproduction tests.
* Voltage-level (CAM-type) models: ``ChannelProfile``, ``Tissue``, ``PatternOptimizer``.
"""

from . import attractors, constants, interventions, ionic, pharmacology, zoo
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
from .control import OptimizationResult, PatternOptimizer
from .gap_junctions import GapJunctionProfile, gj_conductance, grid_conductances
from .integrators import euler_step, integrate, integrate_adaptive, integrate_rosenbrock, rk4_step
from .tissue import BioelectricState, Tissue, tumor_prepattern

__version__ = "0.3.0"
