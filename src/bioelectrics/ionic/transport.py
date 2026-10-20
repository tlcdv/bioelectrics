"""Membrane and gap junction transport: GHK electrodiffusion and ATPase pumps.

Equations are ported from BETSE (``betse/science/sim_toolbox.py``:
``electroflux``, ``pumpNaKATP``, ``pumpCaATP``; ``channels/gap_junction.py``),
Copyright 2014-2025 Alexis Pietak & Cecil Curry, BSD-2-Clause. See
``THIRD_PARTY_NOTICES.md``. The code here is a vectorized, differentiable
PyTorch implementation.

Sign convention: a positive molar flux moves ions **into** the cell [mol/(m^2 s)].
"""

from __future__ import annotations

import torch

from ..constants import F, R

#: Free energy of ATP hydrolysis [J/mol] and metabolite levels [mol/m^3] (BETSE defaults).
DELTA_G_ATP = -37000.0
C_ATP, C_ADP, C_PI = 1.5, 0.1, 0.1
KM_NAK_NA, KM_NAK_K, KM_NAK_ATP = 12.0, 0.2, 0.5
KM_CA_CA, KM_CA_ATP = 1.0e-3, 0.5


def ghk_influx(
    c_out: torch.Tensor,
    c_in: torch.Tensor,
    permeability: torch.Tensor,
    z: torch.Tensor,
    v: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Goldman-Hodgkin-Katz molar flux into the cell [mol/(m^2 s)].

    ``J = P a (c_out e^{-a} - c_in) / (1 - e^{-a})`` with ``a = z F V / (R T)``.
    ``V`` is inside minus outside. The ``a -> 0`` limit ``P (c_out - c_in)`` is
    handled without division by zero, so gradients stay finite.
    """
    a = z * F * v / (R * temperature)
    small = a.abs() < 1e-6
    a_safe = torch.where(small, torch.ones_like(a), a)
    factor = torch.where(small, 1.0 + 0.5 * a, a_safe / (-torch.expm1(-a_safe)))
    return permeability * factor * (c_out * torch.exp(-a) - c_in)


def nak_atpase_influx(
    c_na_in: torch.Tensor,
    c_na_out: torch.Tensor,
    c_k_in: torch.Tensor,
    c_k_out: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Na/K-ATPase fluxes ``(J_Na, J_K)`` into the cell [mol/(m^2 s)] (3 Na out, 2 K in).

    Thermodynamically limited rate from BETSE ``pumpNaKATP``. ``alpha`` is the
    maximum rate constant [m/(mol s)]; a block factor multiplies it.
    """
    q_num = (C_ADP * 1e-3) * (C_PI * 1e-3) * (c_na_out * 1e-3) ** 3 * (c_k_in * 1e-3) ** 2
    q_den = (C_ATP * 1e-3) * (c_na_in * 1e-3) ** 3 * (c_k_out * 1e-3) ** 2
    q = q_num / q_den.clamp_min(1e-30)
    k_eq = torch.exp(-(DELTA_G_ATP / (R * temperature) - F * v / (R * temperature)))
    fwd = (
        (c_na_in / KM_NAK_NA) ** 3
        * (c_k_out / KM_NAK_K) ** 2
        * (C_ATP / KM_NAK_ATP)
        / ((1 + (c_na_in / KM_NAK_NA) ** 3) * (1 + (c_k_out / KM_NAK_K) ** 2) * (1 + C_ATP / KM_NAK_ATP))
    )
    j_na = -3.0 * alpha * fwd * (1.0 - q / k_eq)
    return j_na, -(2.0 / 3.0) * j_na


def ca_atpase_influx(
    c_ca_in: torch.Tensor, c_ca_out: torch.Tensor, v: torch.Tensor, alpha: torch.Tensor, temperature: float
) -> torch.Tensor:
    """Plasma membrane Ca-ATPase flux into the cell [mol/(m^2 s)] (BETSE ``pumpCaATP``)."""
    q = C_ADP * C_PI * c_ca_out / (C_ATP * c_ca_in).clamp_min(1e-16)
    k_eq = torch.exp(-(DELTA_G_ATP / (R * temperature) - 2.0 * F * v / (R * temperature)))
    fwd = (c_ca_in / KM_CA_CA) * (C_ATP / KM_CA_ATP) / ((1 + c_ca_in / KM_CA_CA) * (1 + C_ATP / KM_CA_ATP))
    return -alpha * fwd * (1.0 - q / k_eq)


def betse_gj_open_fraction(
    v_ij: torch.Tensor, v_thresh: float = 0.015, v_window: float = 0.015, g_min: float = 0.10
) -> torch.Tensor:
    """BETSE voltage-sensitive gap junction open fraction (double Boltzmann in V_ij).

    ``g_min + 1/(1+e^{(V-Vth)/w}) - 1/(1+e^{(V+Vth)/w})`` with ``V`` in mV,
    from BETSE ``channels/gap_junction.py``. Defaults are BETSE's
    (``gj voltage threshold`` 15 mV, ``gj voltage window`` 15 mV, ``gj minimum`` 0.1).
    """
    vm = v_ij * 1e3
    vt, w = v_thresh * 1e3, v_window * 1e3
    return g_min + torch.sigmoid(-(vm - vt) / w) - torch.sigmoid(-(vm + vt) / w)
