"""Ion channel "hardware" and membrane currents of the Cervera-Alcaraz-Mafe model.

The genome specifies the hardware (channel expression levels and gating). This
module holds that hardware in :class:`ChannelProfile` and computes the membrane
currents it produces for a given ``V_mem`` (the software state).

Sign convention: currents are outward positive. ``C_m dV/dt = -I_ion + I_ext``.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
import torch
from scipy.optimize import brentq

from .constants import DTYPE, F, R, T, V_T


def _as_tensor(value, dtype=DTYPE, device=None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    return torch.tensor(value, dtype=dtype, device=device)


@dataclass
class ChannelProfile:
    """Fixed channel "hardware" of a cell type (SI units).

    Every field is a tensor. A field can be a scalar or broadcastable to the
    ``V_mem`` tensor, so different cells can express different hardware. Any
    field can have ``requires_grad=True``.

    Defaults are calibrated so that a single cell is bistable with stable fixed
    points at -70 mV (polarized) and -20 mV (depolarized) and a saddle at -45 mV.
    See ``MATH_REFERENCE.md``.
    """

    g_pol0: torch.Tensor = 1.0e-9  # [S]
    g_dep0: torch.Tensor = 1.76452e-9  # [S]
    z: torch.Tensor = 2.11341  # gating charge [-]
    v_th_pol: torch.Tensor = -0.02096  # [V]
    v_th_dep: torch.Tensor = -0.03900  # [V]
    e_pol: torch.Tensor = -0.080  # [V]
    e_dep: torch.Tensor = 0.0  # [V]
    g_leak: torch.Tensor = 0.0  # [S]
    e_leak: torch.Tensor = 0.0  # [V]
    c_m: torch.Tensor = 1.0e-11  # [F]
    v_t: torch.Tensor = V_T  # [V]

    def __post_init__(self) -> None:
        for f in fields(self):
            setattr(self, f.name, _as_tensor(getattr(self, f.name)))

    def to(self, dtype=None, device=None) -> "ChannelProfile":
        """Return a copy with all tensors moved to ``dtype`` / ``device``."""
        return ChannelProfile(
            **{f.name: getattr(self, f.name).to(dtype=dtype, device=device) for f in fields(self)}
        )

    def requires_grad_(self, *names: str) -> "ChannelProfile":
        """Enable gradients on the named fields (default: the conductances)."""
        for name in names or ("g_pol0", "g_dep0", "g_leak"):
            setattr(self, name, getattr(self, name).detach().clone().requires_grad_(True))
        return self


def g_pol(v: torch.Tensor, p: ChannelProfile) -> torch.Tensor:
    """Conductance of the hyperpolarizing channel [S]. Closes on depolarization."""
    return p.g_pol0 * torch.sigmoid(-p.z * (v - p.v_th_pol) / p.v_t)


def g_dep(v: torch.Tensor, p: ChannelProfile) -> torch.Tensor:
    """Conductance of the depolarizing channel [S]. Opens on depolarization."""
    return p.g_dep0 * torch.sigmoid(p.z * (v - p.v_th_dep) / p.v_t)


def i_pol(v: torch.Tensor, p: ChannelProfile) -> torch.Tensor:
    """Outward hyperpolarizing (potassium-like) current [A].

    ``I_pol = G_pol0 (V - E_pol) / (1 + exp(z (V - V_th,pol) / V_T))``
    """
    return g_pol(v, p) * (v - p.e_pol)


def i_dep(v: torch.Tensor, p: ChannelProfile) -> torch.Tensor:
    """Inward depolarizing (sodium/calcium-like) current [A].

    ``I_dep = G_dep0 (V - E_dep) / (1 + exp(-z (V - V_th,dep) / V_T))``
    """
    return g_dep(v, p) * (v - p.e_dep)


def i_leak(v: torch.Tensor, p: ChannelProfile) -> torch.Tensor:
    """Ohmic leak current [A]: ``G_leak (V - E_leak)``."""
    return p.g_leak * (v - p.e_leak)


def ionic_current(v: torch.Tensor, p: ChannelProfile) -> torch.Tensor:
    """Total outward ionic current ``I_pol + I_dep + I_leak`` [A]."""
    return i_pol(v, p) + i_dep(v, p) + i_leak(v, p)


def membrane_rhs(v: torch.Tensor, p: ChannelProfile, i_ext=0.0) -> torch.Tensor:
    """Single-cell ``dV/dt = (-I_ion(V) + I_ext) / C_m`` [V/s]."""
    return (i_ext - ionic_current(v, p)) / p.c_m


def ghk_current(
    v: torch.Tensor,
    c_in: torch.Tensor,
    c_out: torch.Tensor,
    permeability: torch.Tensor,
    valence: float,
    temperature: float = T,
) -> torch.Tensor:
    """Goldman-Hodgkin-Katz current density for one ion species [A/m^2], outward positive.

    ``I = P z^2 F^2 V / (R T) * (c_in - c_out exp(-a)) / (1 - exp(-a))`` with
    ``a = z F V / (R T)``. Concentrations in mol/m^3, permeability in m/s.
    Same form as ``electroflux`` in BETSE ``sim_toolbox.py``. The limit ``V -> 0``
    is handled without division by zero.
    """
    a = valence * F * v / (R * temperature)
    small = a.abs() < 1e-6
    a_safe = torch.where(small, torch.ones_like(a), a)
    # a / (1 - exp(-a)) -> 1 + a/2 as a -> 0
    factor = torch.where(small, 1.0 + a / 2.0, a_safe / (-torch.expm1(-a_safe)))
    return permeability * valence * F * factor * (c_in - c_out * torch.exp(-a))


def find_fixed_points(
    p: ChannelProfile, v_min: float = -0.1, v_max: float = 0.02, n: int = 4001
) -> list[tuple[float, bool]]:
    """Return the single-cell fixed points ``[(V*, is_stable), ...]`` in ascending V.

    A fixed point is a root of ``I_ion(V)``. It is stable when ``dI_ion/dV > 0``.
    Uses a grid scan and Brent refinement. Works on scalar profiles.
    """
    p0 = p.to(dtype=DTYPE)

    def f(x: float) -> float:
        with torch.no_grad():
            return float(ionic_current(torch.tensor(x, dtype=DTYPE), p0))

    grid = np.linspace(v_min, v_max, n)
    with torch.no_grad():
        vals = ionic_current(torch.tensor(grid, dtype=DTYPE), p0).numpy()
    out = []
    for k in np.nonzero(np.sign(vals[:-1]) != np.sign(vals[1:]))[0]:
        root = brentq(f, grid[k], grid[k + 1], xtol=1e-12)
        out.append((root, bool(vals[k + 1] > vals[k])))
    return out
