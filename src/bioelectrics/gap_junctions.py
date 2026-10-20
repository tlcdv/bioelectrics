"""Voltage-gated gap junction coupling on 2D grids and on graphs.

Gap junctions merge cells into an isopotential collective. A high transjunctional
voltage ``|V_i - V_j|`` closes them (``G_ij -> G_min``). This sets the
computational boundary of the self: a depolarized domain becomes electrically
isolated from its polarized surround.

All coupling functions are vectorized. There are no Python loops over cells or edges.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch
import torch.nn.functional as F_nn

from .constants import DTYPE


@dataclass
class GapJunctionProfile:
    """Gap junction (connexin) hardware, SI units. Any field can require gradients."""

    g_max: torch.Tensor = 1.0e-9  # [S]
    g_min: torch.Tensor = 2.0e-11  # [S]
    v0: torch.Tensor = 0.020  # half-closure transjunctional voltage [V]
    v_width: torch.Tensor = 0.004  # gating width [V]

    def __post_init__(self) -> None:
        for f in fields(self):
            value = getattr(self, f.name)
            if not isinstance(value, torch.Tensor):
                setattr(self, f.name, torch.tensor(value, dtype=DTYPE))

    def requires_grad_(self, *names: str) -> "GapJunctionProfile":
        for name in names or ("g_max",):
            setattr(self, name, getattr(self, name).detach().clone().requires_grad_(True))
        return self


def gj_conductance(v_ij: torch.Tensor, gj: GapJunctionProfile) -> torch.Tensor:
    """Voltage-gated conductance ``G_ij(V_ij)`` [S].

    ``G_ij = G_min + (G_max - G_min) / (1 + exp((|V_ij| - V_0) / V_width))``
    """
    return gj.g_min + (gj.g_max - gj.g_min) * torch.sigmoid((gj.v0 - v_ij.abs()) / gj.v_width)


def grid_conductances(v: torch.Tensor, gj: GapJunctionProfile) -> tuple[torch.Tensor, torch.Tensor]:
    """Conductances of horizontal and vertical junctions on a ``(..., H, W)`` grid.

    Returns ``(g_x, g_y)`` with shapes ``(..., H, W-1)`` and ``(..., H-1, W)``.
    """
    dvx = v[..., :, 1:] - v[..., :, :-1]
    dvy = v[..., 1:, :] - v[..., :-1, :]
    return gj_conductance(dvx, gj), gj_conductance(dvy, gj)


def grid_coupling_current(v: torch.Tensor, gj: GapJunctionProfile) -> torch.Tensor:
    """Gap junction current into each cell, ``sum_j G_ij (V_j - V_i)`` [A].

    ``v`` has shape ``(..., H, W)``. Von Neumann (4-neighbor) coupling with
    zero-flux borders, computed with tensor shifts.
    """
    dvx = v[..., :, 1:] - v[..., :, :-1]  # V_right - V_left
    dvy = v[..., 1:, :] - v[..., :-1, :]  # V_below - V_above
    jx = gj_conductance(dvx, gj) * dvx  # current into the left cell
    jy = gj_conductance(dvy, gj) * dvy  # current into the upper cell
    ix = F_nn.pad(jx, (0, 1)) - F_nn.pad(jx, (1, 0))
    iy = F_nn.pad(jy, (0, 0, 0, 1)) - F_nn.pad(jy, (0, 0, 1, 0))
    return ix + iy


def graph_coupling_current(
    v: torch.Tensor, edges: torch.Tensor, gj: GapJunctionProfile
) -> torch.Tensor:
    """Gap junction current into each node of a graph [A].

    ``v`` has shape ``(..., N_cells)``. ``edges`` is a ``(2, E)`` long tensor of
    undirected edges ``(i, j)``, each listed once.
    """
    i, j = edges[0], edges[1]
    dv = v[..., j] - v[..., i]
    flux = gj_conductance(dv, gj) * dv  # current into node i
    out = torch.zeros_like(v)
    return out.index_add(-1, i, flux).index_add(-1, j, -flux)


def edges_from_networkx(graph) -> torch.Tensor:
    """Convert a ``networkx`` graph with nodes ``0..N-1`` to a ``(2, E)`` edge tensor."""
    return torch.tensor(list(graph.edges()), dtype=torch.long).reshape(-1, 2).T.contiguous()
