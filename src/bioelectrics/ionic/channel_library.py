"""Hodgkin-Huxley gated ion channels (ported from BETSE).

Each channel has gates ``m`` and ``h`` with ``dm/dt = (m_inf(V) - m) / tau_m(V)``
(same for ``h``) and open probability ``P = m^p h^q``. The channel adds
``P * rel_perm[ion] * max_D`` to the membrane diffusion constant of each ion it
conducts. Gating functions take ``V`` in mV and return ``tau`` in ms, exactly as
in BETSE (``betse/science/channels/vg_*.py``, BSD-2-Clause, see
``THIRD_PARTY_NOTICES.md``). The original experimental reference is given for
each channel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch

Gate = Callable[[torch.Tensor], torch.Tensor]


def _one(v: torch.Tensor) -> torch.Tensor:
    return torch.ones_like(v)


def _boltz(v: torch.Tensor, v_half: float, k: float) -> torch.Tensor:
    """``1 / (1 + exp((V - v_half) / k))``."""
    return torch.sigmoid(-(v - v_half) / k)


@dataclass(frozen=True)
class GatedChannel:
    """A voltage-gated channel type (hardware component)."""

    name: str
    ions: tuple[str, ...]
    rel_perm: tuple[float, ...]
    m_power: int
    h_power: int
    m_inf: Gate = field(repr=False)
    m_tau: Gate = field(repr=False)
    h_inf: Gate = field(default=_one, repr=False)
    h_tau: Gate = field(default=_one, repr=False)
    v_shift_mV: float = 0.0
    reference: str = ""

    def _v(self, v: torch.Tensor) -> torch.Tensor:
        return v * 1e3 + self.v_shift_mV

    def steady_state(self, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        vm = self._v(v)
        return self.m_inf(vm), self.h_inf(vm)

    def gate_rates(self, v: torch.Tensor, m: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(dm/dt, dh/dt)`` in 1/s. Time constants are clamped to >= 0.05 ms."""
        vm = self._v(v)
        tau_m = self.m_tau(vm).clamp_min(0.05) * 1e-3
        tau_h = self.h_tau(vm).clamp_min(0.05) * 1e-3
        return (self.m_inf(vm) - m) / tau_m, (self.h_inf(vm) - h) / tau_h

    def open_probability(self, m: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return m**self.m_power * h**self.h_power


KV1P5 = GatedChannel(
    name="Kv1.5",
    ions=("K",),
    rel_perm=(1.0,),
    m_power=1,
    h_power=1,
    m_inf=lambda v: _boltz(v, -6.0, -6.4),
    m_tau=lambda v: -0.1163 * v + 8.33,
    h_inf=lambda v: _boltz(v, -25.3, 3.5),
    h_tau=lambda v: -15.5 * v + 1620.0,
    reference="Philipson et al., PNAS 88, 53-57 (1991); BETSE vg_k.Kv1p5",
)

KIR2P1 = GatedChannel(
    name="Kir2.1",
    ions=("K",),
    rel_perm=(1.0,),
    m_power=1,
    h_power=2,
    m_inf=lambda v: _boltz(v, -96.48, 23.26),
    m_tau=lambda v: 3.7 - 3.37 * torch.sigmoid(-(v + 32.9) / 27.93),
    h_inf=lambda v: _boltz(v, -168.28, -44.13),
    h_tau=lambda v: 0.85 + 306.3 * torch.sigmoid((v + 118.29) / 27.23),
    reference="Makary et al., J. Physiol. 568, 749-766 (2005); BETSE vg_k.Kir2p1",
)

HCN2 = GatedChannel(
    name="HCN2",
    ions=("Na", "K", "Ca"),
    rel_perm=(0.2, 1.0, 0.05),
    m_power=1,
    h_power=0,
    m_inf=lambda v: _boltz(v, -99.0, 6.2),
    m_tau=lambda v: torch.full_like(v, 184.0),
    v_shift_mV=-10.0,
    reference="Moosmang et al., Eur. J. Biochem. 268, 1646-1652 (2001); BETSE vg_funny.HCN2",
)

CAV1P2 = GatedChannel(
    name="Cav1.2",
    ions=("Ca",),
    rel_perm=(1.0,),
    m_power=2,
    h_power=1,
    m_inf=lambda v: _boltz(v, -30.0, -6.0),
    m_tau=lambda v: 5.0 + 20.0 * torch.sigmoid(-(v + 25.0) / 5.0),
    h_inf=lambda v: _boltz(v, -80.0, 6.4),
    h_tau=lambda v: 20.0 + 50.0 * torch.sigmoid(-(v + 40.0) / 7.0),
    reference="BETSE vg_ca.Cav1p2",
)


def cam_type(
    name: str, ions: tuple[str, ...], rel_perm: tuple[float, ...], v_half_mV: float = -25.0,
    z: float = 3.0, opens_on_depolarization: bool = False, tau_ms: float = 1.0,
) -> GatedChannel:
    """Phenomenological channel with the steep Boltzmann gating of Cervera et al. (2016).

    Open probability ``1 / (1 + exp(+-z (V - V_half) / V_T))``, with a fast gate
    (``tau_ms``). It is not a model of one specific protein.
    """
    k = 26.727 / z  # V_T [mV] / z at 310.15 K
    sign = -1.0 if opens_on_depolarization else 1.0
    return GatedChannel(
        name=name, ions=ions, rel_perm=rel_perm, m_power=1, h_power=0,
        m_inf=lambda v: torch.sigmoid(-sign * (v - v_half_mV) / k),
        m_tau=lambda v: torch.full_like(v, tau_ms),
        reference="CAM-type gating, Cervera, Alcaraz & Mafe, Sci. Rep. 6, 20403 (2016)",
    )


#: Inward-rectifier K+ channel with CAM gating (open below -25 mV): polarizing.
K_IR_CAM = cam_type("K_ir(CAM)", ("K",), (1.0,))
#: Non-selective cation channel with CAM gating (open above -25 mV): depolarizing.
CAT_DEP_CAM = cam_type("Cat_dep(CAM)", ("Na", "K"), (1.0, 1.0), opens_on_depolarization=True)


def leak(ion: str) -> GatedChannel:
    """A constitutively open (ungated) channel for one ion."""
    return GatedChannel(
        name=f"{ion}Leak", ions=(ion,), rel_perm=(1.0,), m_power=0, h_power=0,
        m_inf=_one, m_tau=_one, reference="constitutive leak",
    )


LIBRARY: dict[str, GatedChannel] = {
    c.name: c for c in (KV1P5, KIR2P1, HCN2, CAV1P2, K_IR_CAM, CAT_DEP_CAM, leak("K"), leak("Na"), leak("Cl"))
}
