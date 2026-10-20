"""Cervera, Alcaraz & Mafe (2016): bistable cells coupled by gap junctions.

Reference
---------
J. Cervera, A. Alcaraz, S. Mafe. *Bioelectrical signals and ion channels in the
modeling of multicellular patterns and cancer biophysics.* Sci. Rep. 6, 20403 (2016).
doi:10.1038/srep20403 (open access, CC BY 4.0).

Model (paper Eqs. 1-5, capacitive coupling omitted as in the paper)::

    C0 dV_i/dt = -I_in(V_i) - I_out(V_i) + sum_j G (V_j - V_i)
    I_in  = G_in  (V - E_in)  / (1 + exp( z (V - V_th,in)  / V_T))
    I_out = G_out (V - E_out) / (1 + exp(-z (V - V_th,out) / V_T))

The paper's "inward-rectifying" channel (E_in = -60 mV) is the polarizing channel
(``i_pol`` in this library). The "outward-rectifying" channel (E_out = 0 mV) is the
depolarizing channel (``i_dep``).

Parameters from the paper text and Fig. 2: G_in = G_out = 1 nS, V_th = -25 mV,
z = 3, V_T = 27 mV, C0 = 100 pF (tau = C0 / G_in = 0.1 s). Gap junction conductance
G is constant (not voltage-gated) and given as G / G_in.

Fixed points with these parameters (this library): -58.6 mV (stable), -15.5 mV
(unstable), -7.5 mV (stable). They agree with the colour bar of the paper's Fig. 4.

Reproduced results (see ``tests/zoo/test_cam2016.py``):

* Fig. 4: a depolarized central region normalizes faster for stronger coupling
  (G/G_in = 0.05, 0.15, 0.25). Qualitative: the exact region shape is not published.
* Fig. 5: with G/G_in = 0.25, over-expression of the depolarizing channel
  (G_out/G_in = 2.5) prevents normalization.
"""

from __future__ import annotations

import torch

from ..channels import ChannelProfile
from ..constants import DTYPE
from ..gap_junctions import GapJunctionProfile

G_IN = 1.0e-9  # [S]
C0 = 100e-12  # [F]
TAU = C0 / G_IN  # characteristic time [s]
REFERENCE = "Cervera, Alcaraz & Mafe, Sci. Rep. 6, 20403 (2016), doi:10.1038/srep20403"


def channel_profile(g_out_ratio: float = 1.0) -> ChannelProfile:
    """Single-cell hardware of the paper. ``g_out_ratio`` is G_out / G_in."""
    return ChannelProfile(
        g_pol0=G_IN,
        g_dep0=g_out_ratio * G_IN,
        z=3.0,
        v_th_pol=-0.025,
        v_th_dep=-0.025,
        e_pol=-0.060,
        e_dep=0.0,
        v_t=0.027,
        c_m=C0,
    )


def gap_junctions(g_ratio: float) -> GapJunctionProfile:
    """Constant (not voltage-gated) gap junction conductance ``G = g_ratio * G_in``."""
    g = g_ratio * G_IN
    return GapJunctionProfile(g_max=g, g_min=g)


def cross_region(n: int = 50, arm: float = 7.5, width: float = 2.5, core: float = 4.5) -> torch.Tensor:
    """Boolean ``(n, n)`` mask of a cross-shaped central region, similar to paper Fig. 4.

    The paper does not publish the exact shape. The default mask has 172 cells.
    """
    y, x = torch.meshgrid(torch.arange(n, dtype=DTYPE), torch.arange(n, dtype=DTYPE), indexing="ij")
    c = (n - 1) / 2.0
    dx, dy = (x - c).abs(), (y - c).abs()
    return ((dx <= arm) & (dy <= width)) | ((dy <= arm) & (dx <= width)) | ((dx <= core) & (dy <= core))
