"""Multicellular tissue: the bioelectric "software" state and its vectorized dynamics.

``C_m dV_i/dt = -I_pol - I_dep - I_leak + sum_j G_ij(V_i - V_j) (V_j - V_i) + I_ext,i``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch

from .channels import ChannelProfile, ionic_current
from .constants import DTYPE, V_to_mV, mV_to_V
from .gap_junctions import GapJunctionProfile, graph_coupling_current, grid_coupling_current
from .integrators import integrate

Stimulus = Callable[[float], torch.Tensor] | torch.Tensor | float | None


@dataclass
class BioelectricState:
    """Dynamic ``V_mem`` state (the bioelectric "software") at time ``t``.

    ``v`` is in volts with shape ``(batch, H, W)`` for grids or ``(batch, N_cells)``
    for graphs.
    """

    v: torch.Tensor
    t: float = 0.0

    @classmethod
    def uniform_grid(cls, h: int, w: int, v_mV: float = -70.0, batch: int = 1) -> "BioelectricState":
        return cls(torch.full((batch, h, w), mV_to_V(v_mV), dtype=DTYPE))

    @classmethod
    def from_mV(cls, v_mV: torch.Tensor, t: float = 0.0) -> "BioelectricState":
        return cls(mV_to_V(torch.as_tensor(v_mV, dtype=DTYPE)), t)

    @property
    def v_mV(self) -> torch.Tensor:
        return V_to_mV(self.v)

    def clone(self) -> "BioelectricState":
        return BioelectricState(self.v.clone(), self.t)


def tumor_prepattern(
    h: int,
    w: int,
    patch: int,
    v_healthy_mV: float = -70.0,
    v_tumor_mV: float = -15.0,
    batch: int = 1,
) -> tuple[BioelectricState, torch.Tensor]:
    """Polarized tissue with a central depolarized ``patch x patch`` tumor domain.

    Returns the state and a boolean tumor mask of shape ``(h, w)``.
    """
    mask = torch.zeros(h, w, dtype=torch.bool)
    r0, c0 = (h - patch) // 2, (w - patch) // 2
    mask[r0 : r0 + patch, c0 : c0 + patch] = True
    v = torch.full((batch, h, w), mV_to_V(v_healthy_mV), dtype=DTYPE)
    v[:, mask] = mV_to_V(v_tumor_mV)
    return BioelectricState(v), mask


@dataclass
class Tissue:
    """A multicellular network: channel hardware + gap junction hardware + topology.

    ``edges=None`` means a 2D grid with 4-neighbor coupling and state shape
    ``(batch, H, W)``. Otherwise ``edges`` is a ``(2, E)`` tensor and the state
    shape is ``(batch, N_cells)``.
    """

    channels: ChannelProfile = field(default_factory=ChannelProfile)
    gap_junctions: GapJunctionProfile = field(default_factory=GapJunctionProfile)
    edges: torch.Tensor | None = None

    def coupling_current(self, v: torch.Tensor) -> torch.Tensor:
        if self.edges is None:
            return grid_coupling_current(v, self.gap_junctions)
        return graph_coupling_current(v, self.edges, self.gap_junctions)

    def rhs(self, v: torch.Tensor, i_ext=0.0) -> torch.Tensor:
        """``dV/dt`` [V/s] for all cells at once."""
        i_total = -ionic_current(v, self.channels) + self.coupling_current(v) + i_ext
        return i_total / self.channels.c_m

    def simulate(
        self,
        state: BioelectricState,
        duration: float,
        dt: float = 5e-4,
        stimulus: Stimulus = None,
        method: str = "rk4",
        record_every: int | None = None,
    ):
        """Integrate the tissue for ``duration`` seconds.

        ``stimulus`` is ``None``, a constant tensor/float ``I_ext`` [A], or a
        callable ``t -> I_ext``. Returns the final :class:`BioelectricState`, and
        also ``(times, trajectory)`` when ``record_every`` is set.
        """
        if stimulus is None:
            stim_fn = lambda t: 0.0
        elif callable(stimulus):
            stim_fn = stimulus
        else:
            stim_fn = lambda t: stimulus
        rhs = lambda t, v: self.rhs(v, stim_fn(t))
        v_end, times, traj = integrate(
            rhs, state.v, state.t + duration, dt, t0=state.t, method=method, record_every=record_every
        )
        out = BioelectricState(v_end, state.t + duration)
        if record_every:
            return out, times, traj
        return out
