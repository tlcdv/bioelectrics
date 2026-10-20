"""Ion-level, differentiable multicellular tissue model (BETSE-compatible physics).

State per cell: intracellular concentrations of every ion and the ``m``/``h`` gates
of every expressed channel. The membrane potential follows from net intracellular
charge, as in BETSE (``sim.py``, ``update_V``)::

    V_mem = F * vol / (C_m * area) * sum_k z_k c_k

Fluxes into each cell [mol/(m^2 s)]:

* GHK electrodiffusion through the membrane, with permeability
  ``P_k = (D_m,k + sum_channels P_open * rel_perm_k * max_D) / d_mem``;
* Na/K-ATPase (3 Na out, 2 K in) and Ca-ATPase;
* GHK electrodiffusion through gap junctions to each neighbour, with
  ``P_gj,k = D_free,k * f_gj * open(V_ij) / L_gj`` over a contact area.

The extracellular space is a fixed bath (BETSE without ``is_ecm``). All operations
are vectorized over the batch, the ions and the cells (grid ``(B, H, W)`` or graph
``(B, N)``), and are differentiable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn.functional as F_nn

from ..constants import DTYPE, F, T
from ..integrators import integrate, integrate_adaptive
from .channel_library import GatedChannel
from .profiles import IonProfile
from .transport import betse_gj_open_fraction, ca_atpase_influx, ghk_influx, nak_atpase_influx


@dataclass
class CellGeometry:
    """Membrane area [m^2] and cytoplasm volume [m^3] of one cell."""

    area: float
    volume: float

    @classmethod
    def sphere(cls, radius: float = 5.0e-6) -> "CellGeometry":
        return cls(4.0 * math.pi * radius**2, 4.0 / 3.0 * math.pi * radius**3)

    @classmethod
    def cylinder(cls, radius: float = 5.0e-6, height: float = 10.0e-6) -> "CellGeometry":
        return cls(2.0 * math.pi * radius * (radius + height), math.pi * radius**2 * height)


@dataclass
class ChannelExpression:
    """Expression of one channel type: ``max_d`` [m^2/s], scalar or per cell (hardware)."""

    channel: GatedChannel
    max_d: torch.Tensor | float

    def __post_init__(self) -> None:
        if not isinstance(self.max_d, torch.Tensor):
            self.max_d = torch.tensor(self.max_d, dtype=DTYPE)


@dataclass
class Modifiers:
    """Time-dependent changes to the hardware, produced by interventions.

    Every field broadcasts against cell tensors of shape ``(B, *cells)``.

    * ``channel_scale[name]`` multiplies the channel's ``max_d`` (block < 1).
    * ``channel_add[name]`` adds to ``max_d`` [m^2/s] (misexpression; also for
      channels with zero baseline expression).
    * ``channel_gate[name]`` replaces the open probability (light-gated channels).
    * ``nak_scale``, ``ca_pump_scale``, ``gj_scale`` multiply pump rates and
      gap junction permeability.
    """

    channel_scale: dict[str, torch.Tensor] = field(default_factory=dict)
    channel_add: dict[str, torch.Tensor] = field(default_factory=dict)
    channel_gate: dict[str, torch.Tensor] = field(default_factory=dict)
    nak_scale: torch.Tensor | float = 1.0
    ca_pump_scale: torch.Tensor | float = 1.0
    gj_scale: torch.Tensor | float = 1.0


NO_MODIFIERS = Modifiers()


@dataclass
class IonicState:
    """Packed state tensor ``x`` of shape ``(B, n_ions + 2 * n_channels, *cells)`` at time ``t``."""

    x: torch.Tensor
    t: float = 0.0


class IonicTissue:
    """Multicellular tissue with ion-level physics.

    Parameters (defaults follow BETSE ``sim_config.yaml`` unless stated)
    ----------
    profile: ion concentrations (default :meth:`IonProfile.mammal`).
    geometry: cell area and volume (default sphere, r = 5 um).
    channels: expressed gated channels (hardware).
    membrane_d: baseline membrane diffusion constants per ion [m^2/s].
    alpha_nak: Na/K-ATPase rate [m/(mol s)] (1e-7); ``alpha_ca``: Ca-ATPase (5e-8).
    c_m: membrane capacitance [F/m^2]. Default 0.01 F/m^2 (1 uF/cm^2, the measured
        value); BETSE's default is 0.05.
    membrane_thickness: [m] (7.5 nm).
    gj_fraction: gap junction area fraction (BETSE ``gap junction surface area``, 5e-8).
    gj_length: junction pore length [m] (BETSE ``cell spacing``, 26 nm).
    gj_contact_area: membrane area shared with one neighbour [m^2] (default area / 6).
    gj_gating: ``"betse"`` (voltage-gated, BETSE defaults) or ``None`` (always open).
    edges: ``None`` for a 2D grid, or a ``(2, E)`` tensor of undirected edges.
    """

    def __init__(
        self,
        profile: IonProfile | None = None,
        geometry: CellGeometry | None = None,
        channels: list[ChannelExpression] | None = None,
        membrane_d: torch.Tensor | None = None,
        alpha_nak: float | torch.Tensor = 1.0e-7,
        alpha_ca: float | torch.Tensor = 5.0e-8,
        c_m: float = 0.01,
        temperature: float = T,
        membrane_thickness: float = 7.5e-9,
        gj_fraction: float | torch.Tensor = 5.0e-8,
        gj_length: float = 26.0e-9,
        gj_contact_area: float | None = None,
        gj_gating: str | None = "betse",
        edges: torch.Tensor | None = None,
    ) -> None:
        self.profile = profile or IonProfile.mammal()
        self.geometry = geometry or CellGeometry.sphere()
        self.channels = channels or []
        self.membrane_d = self.profile.default_membrane_d() if membrane_d is None else membrane_d
        self.alpha_nak = torch.as_tensor(alpha_nak, dtype=DTYPE)
        self.alpha_ca = torch.as_tensor(alpha_ca, dtype=DTYPE)
        self.c_m = c_m
        self.temperature = temperature
        self.d_mem = membrane_thickness
        self.gj_fraction = torch.as_tensor(gj_fraction, dtype=DTYPE)
        self.gj_length = gj_length
        self.gj_contact_area = gj_contact_area or self.geometry.area / 6.0
        self.gj_gating = gj_gating
        self.edges = edges
        self.n_ions = len(self.profile.ions)
        self._z = self.profile.z
        self._ion = {name: k for k, name in enumerate(self.profile.ions)}

    # ----------------------------------------------------------------- helpers
    def _bcast(self, vec: torch.Tensor, ndim_cells: int) -> torch.Tensor:
        """Reshape an ``(n_ions,)`` vector to broadcast over ``(B, n_ions, *cells)``."""
        return vec.reshape(1, -1, *([1] * ndim_cells))

    def vmem(self, c_in: torch.Tensor) -> torch.Tensor:
        """Membrane potential [V] from intracellular concentrations ``(B, n_ions, *cells)``."""
        ndim = c_in.dim() - 2
        rho = (c_in * self._bcast(self._z, ndim)).sum(dim=1)
        return F * self.geometry.volume / (self.c_m * self.geometry.area) * rho

    def split(self, x: torch.Tensor) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        c = x[:, : self.n_ions]
        gates = [(x[:, self.n_ions + 2 * k], x[:, self.n_ions + 2 * k + 1]) for k in range(len(self.channels))]
        return c, gates

    def initial_state(self, cells: tuple[int, ...], batch: int = 1) -> IonicState:
        """Uniform tissue at the profile concentrations (V = 0) with steady-state gates."""
        c = self.profile.c_in.reshape(1, -1, *([1] * len(cells))).expand(batch, -1, *cells).clone()
        v = self.vmem(c)
        parts = [c]
        for expr in self.channels:
            m, h = expr.channel.steady_state(v)
            parts += [m.unsqueeze(1), h.unsqueeze(1)]
        return IonicState(torch.cat(parts, dim=1))

    # -------------------------------------------------------------------- physics
    def _gap_junction_influx(self, c: torch.Tensor, v: torch.Tensor, mods: Modifiers) -> torch.Tensor:
        """Moles per second into each cell through gap junctions, ``(B, n_ions, *cells)``."""
        ndim = c.dim() - 2
        z = self._bcast(self._z, ndim)
        p0 = self._bcast(self.profile.d_free, ndim) * self.gj_fraction / self.gj_length
        a = self.gj_contact_area

        s = mods.gj_scale
        s = s if isinstance(s, torch.Tensor) and s.dim() > 0 else torch.as_tensor(s, dtype=c.dtype).expand_as(v)
        s = s.expand_as(v)

        def edge_flux(c_i, c_j, v_i, v_j, s_i, s_j):
            v_ij = v_i - v_j
            open_frac = betse_gj_open_fraction(v_ij) if self.gj_gating == "betse" else torch.ones_like(v_ij)
            perm = p0 * (open_frac * 0.5 * (s_i + s_j)).unsqueeze(1)
            return a * ghk_influx(c_j, c_i, perm, z, v_ij.unsqueeze(1), self.temperature)  # into cell i

        if self.edges is None:
            fx = edge_flux(c[..., :, :-1], c[..., :, 1:], v[..., :, :-1], v[..., :, 1:], s[..., :, :-1], s[..., :, 1:])
            fy = edge_flux(c[..., :-1, :], c[..., 1:, :], v[..., :-1, :], v[..., 1:, :], s[..., :-1, :], s[..., 1:, :])
            return (F_nn.pad(fx, (0, 1)) - F_nn.pad(fx, (1, 0))
                    + F_nn.pad(fy, (0, 0, 0, 1)) - F_nn.pad(fy, (0, 0, 1, 0)))
        i, j = self.edges[0], self.edges[1]
        f = edge_flux(c[..., i], c[..., j], v[..., i], v[..., j], s[..., i], s[..., j])
        out = torch.zeros_like(c)
        return out.index_add(-1, i, f).index_add(-1, j, -f)

    def rhs(self, x: torch.Tensor, mods: Modifiers = NO_MODIFIERS) -> torch.Tensor:
        """Time derivative of the packed state tensor."""
        c, gates = self.split(x)
        ndim = c.dim() - 2
        v = self.vmem(c)
        c_out = self._bcast(self.profile.c_out, ndim)
        z = self._bcast(self._z, ndim)

        # Membrane diffusion constants per ion and cell.
        extra = [torch.zeros_like(v) for _ in range(self.n_ions)]
        dgates = []
        for expr, (m, h) in zip(self.channels, gates):
            ch = expr.channel
            dm, dh = ch.gate_rates(v, m, h)
            dgates += [dm.unsqueeze(1), dh.unsqueeze(1)]
            p_open = mods.channel_gate.get(ch.name)
            if p_open is None:
                p_open = ch.open_probability(m, h)
            max_d = expr.max_d * mods.channel_scale.get(ch.name, 1.0) + mods.channel_add.get(ch.name, 0.0)
            level = p_open * max_d  # (B, *cells)
            for ion, rel in zip(ch.ions, ch.rel_perm):
                if ion in self._ion:
                    k = self._ion[ion]
                    extra[k] = extra[k] + rel * level
        perm = (self._bcast(self.membrane_d, ndim) + torch.stack(extra, dim=1)) / self.d_mem
        j = ghk_influx(c_out, c, perm, z, v.unsqueeze(1), self.temperature)

        # Pumps.
        pump = [torch.zeros_like(v) for _ in range(self.n_ions)]
        if "Na" in self._ion and "K" in self._ion:
            ina, ik = self._ion["Na"], self._ion["K"]
            pump[ina], pump[ik] = nak_atpase_influx(
                c[:, ina], c_out[:, ina], c[:, ik], c_out[:, ik], v, self.alpha_nak * mods.nak_scale, self.temperature
            )
        if "Ca" in self._ion:
            ica = self._ion["Ca"]
            pump[ica] = ca_atpase_influx(c[:, ica], c_out[:, ica], v, self.alpha_ca * mods.ca_pump_scale, self.temperature)
        pump = torch.stack(pump, dim=1)

        dc = ((j + pump) * self.geometry.area + self._gap_junction_influx(c, v, mods)) / self.geometry.volume
        return torch.cat([dc] + dgates, dim=1) if dgates else dc

    # ------------------------------------------------------------------ simulation
    def tolerances(self, x: torch.Tensor, conc_atol: float = 1e-9, gate_atol: float = 1e-6):
        """Per-component ``(rtol, atol)`` tensors for the adaptive solver.

        V depends on a small difference of large concentrations
        (dV/dc = F vol / (C_m area), about 16 V per mol/m^3), so concentrations get a
        pure absolute tolerance: ``conc_atol = 1e-9`` mol/m^3 bounds the voltage
        error per step to about 20 nV.
        """
        shape = [1, x.shape[1]] + [1] * (x.dim() - 2)
        atol = torch.full(shape, gate_atol, dtype=x.dtype)
        rtol = torch.full(shape, 1e-4, dtype=x.dtype)
        atol[:, : self.n_ions] = conc_atol
        rtol[:, : self.n_ions] = 0.0
        return rtol, atol

    def simulate(
        self,
        state: IonicState,
        duration: float,
        dt: float = 1e-3,
        modifiers: Callable[[float], Modifiers] | Modifiers | None = None,
        method: str = "dopri5",
        record_every: int | None = None,
        t_eval: list[float] | None = None,
        dt_max: float = 0.05,
    ):
        """Integrate for ``duration`` seconds. ``modifiers`` may depend on time.

        ``method="dopri5"`` (default) is adaptive and safe for stiff channel
        expression; ``dt`` is then the first trial step and ``t_eval`` selects
        recorded times. ``method="rk4"``/``"euler"`` use the fixed step ``dt`` and
        ``record_every``. Time-dependent modifiers should change only at times in
        ``t_eval`` or be smooth, so the adaptive steps resolve them.
        """
        if modifiers is None:
            mod_fn = lambda t: NO_MODIFIERS
        elif callable(modifiers):
            mod_fn = modifiers
        else:
            mod_fn = lambda t: modifiers
        rhs = lambda t, x: self.rhs(x, mod_fn(t))
        t_end = state.t + duration
        if method == "dopri5":
            rtol, atol = self.tolerances(state.x)
            x_end, times, traj, self.last_stats = integrate_adaptive(
                rhs, state.x, t_end, t0=state.t, rtol=rtol, atol=atol, dt0=dt, dt_max=dt_max, t_eval=t_eval
            )
            out = IonicState(x_end, t_end)
            return (out, times, traj) if t_eval else out
        x_end, times, traj = integrate(
            rhs, state.x, t_end, dt, t0=state.t, method=method, record_every=record_every
        )
        out = IonicState(x_end, t_end)
        return (out, times, traj) if record_every else out

    def vmem_of(self, state: IonicState) -> torch.Tensor:
        return self.vmem(state.x[:, : self.n_ions])
