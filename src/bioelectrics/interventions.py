"""Experimentally realizable bioelectric interventions and their inverse design.

An intervention changes the tissue *hardware* for a limited time: it adds channel
expression, blocks channels, pumps or gap junctions, or opens light-gated channels.
:class:`InterventionOptimizer` finds the intervention parameters (dose, region,
light pattern) that move the tissue into a Target Morphology, by backpropagation
through the ion-level simulation. The result is a :class:`Protocol` stated in lab
terms.

Every rollout has two phases (see ``DOMAIN_CONCEPTS.md``):

* ``[0, T_stim]``: all interventions active (constant in time).
* ``[T_stim, T_relax]``: no intervention. The pattern error is measured at
  ``T_relax``, so success means a permanent attractor switch (a trigger).

:class:`ChronicInterventionOptimizer` handles interventions that stay on (a
constant drug concentration, a stable misexpression). Its readout is the steady
state that the tissue reaches under the intervention, and its gradients come from
the implicit function theorem (:mod:`bioelectrics.attractors`), not from BPTT.

Each intervention parameter is bounded: ``value = lo + (hi - lo) * sigmoid(u)``,
where ``u`` is the unconstrained optimization variable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .constants import DTYPE
from .ionic.model import IonicState, IonicTissue, Modifiers


class Intervention:
    """Base class. Subclasses set ``kind``, ``target`` and implement :meth:`apply`."""

    kind: str = "intervention"
    unit: str = ""

    def __init__(
        self,
        target: str,
        region: torch.Tensor | None,
        lo: float,
        hi: float,
        init: float,
        spatial: bool = False,
        cost: float = 0.0,
    ) -> None:
        self.target = target
        self.region = None if region is None else region.to(DTYPE)
        self.lo, self.hi = lo, hi
        self.spatial = spatial
        self.cost = cost
        frac = (init - lo) / (hi - lo)
        frac = min(max(frac, 1e-4), 1 - 1e-4)
        u0 = torch.logit(torch.tensor(frac, dtype=DTYPE))
        self._init_u = u0
        self.u: torch.Tensor | None = None

    def build(self, cell_shape: tuple[int, ...]) -> torch.Tensor:
        """Create the optimization variable for a tissue with cells of ``cell_shape``."""
        shape = cell_shape if self.spatial else ()
        self.u = torch.full(shape, float(self._init_u), dtype=DTYPE, requires_grad=True)
        return self.u

    def value(self) -> torch.Tensor:
        """Physical value (scalar or per-cell map) within ``[lo, hi]``."""
        return self.lo + (self.hi - self.lo) * torch.sigmoid(self.u)

    def field(self) -> torch.Tensor:
        """Per-cell value: ``value * region`` (region ``None`` = whole tissue)."""
        v = self.value()
        return v if self.region is None else v * self.region

    def normalized_magnitude(self) -> torch.Tensor:
        """Mean fraction of the allowed range used, for the cost term."""
        return ((self.field() - self.lo) / (self.hi - self.lo)).abs().mean()

    def apply(self, mods: Modifiers) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def describe(self) -> dict:
        """Summary for a protocol table.

        For a spatial intervention, a cell counts as targeted when its value is at
        least 10 % of the maximum; ``value`` is the mean over targeted cells.
        """
        f = self.field().detach()
        if f.dim() and self.spatial:
            active = f >= 0.1 * f.max()
        else:
            active = f.abs() > 1e-3 * max(abs(self.hi), abs(self.lo), 1e-30)
        return {
            "intervention": self.kind,
            "target": self.target,
            "value": float(f[active].mean()) if active.any() else 0.0,
            "max": float(f.max()),
            "unit": self.unit,
            "cells": int(active.sum()) if f.dim() else None,
        }


class ChannelMisexpression(Intervention):
    """Add expression of a channel in a region (e.g. mRNA injection of Kir2.1).

    The channel must be part of ``tissue.channels`` (baseline ``max_d`` may be 0).
    Value: added ``max_d`` [m^2/s].
    """

    kind, unit = "channel misexpression", "m^2/s"

    def __init__(self, channel: str, region=None, max_d: float = 1e-15, init: float | None = None,
                 spatial: bool = False, cost: float = 0.0):
        super().__init__(channel, region, 0.0, max_d, init if init is not None else 0.05 * max_d, spatial, cost)

    def apply(self, mods: Modifiers) -> None:
        mods.channel_add[self.target] = mods.channel_add.get(self.target, 0.0) + self.field()


class _Blocker(Intervention):
    """Base for the three blockers. The optimized value is a fraction or a dose.

    Without ``ic50`` the value is the blocked fraction directly, in
    ``[0, max_block]``. With ``ic50`` the value is a concentration in the same
    unit as ``ic50``, and the blocked fraction follows the Hill equation
    ``1 / (1 + (ic50 / dose)^hill)``. Subclasses implement :meth:`apply`.
    """

    def __init__(self, target: str, region, max_block: float, ic50: float | None, hill: float,
                 max_dose: float | None, init: float | None, spatial: bool, cost: float):
        self.ic50, self.hill = ic50, hill
        if ic50 is None:
            self.unit = "fraction blocked"
            super().__init__(target, region, 0.0, max_block, init if init is not None else 0.1, spatial, cost)
        else:
            self.unit = "dose"
            hi = max_dose if max_dose is not None else 100.0 * ic50
            super().__init__(target, region, 0.0, hi, init if init is not None else 0.1 * ic50, spatial, cost)

    def blocked_fraction(self) -> torch.Tensor:
        """The fraction of the target blocked, in ``[0, 1]``."""
        f = self.field()
        if self.ic50 is None:
            return f
        return 1.0 / (1.0 + (self.ic50 / f.clamp_min(1e-30)) ** self.hill)


class ChannelBlocker(_Blocker):
    """Block a channel type (drug). Value: blocked fraction, or a dose with ``ic50``."""

    kind = "channel blocker"

    def __init__(self, channel: str, region=None, max_block: float = 1.0, ic50: float | None = None,
                 hill: float = 1.0, max_dose: float | None = None, init: float | None = None,
                 spatial: bool = False, cost: float = 0.0):
        super().__init__(channel, region, max_block, ic50, hill, max_dose, init, spatial, cost)

    def apply(self, mods: Modifiers) -> None:
        s = mods.channel_scale.get(self.target, 1.0)
        mods.channel_scale[self.target] = s * (1.0 - self.blocked_fraction())


class PumpBlocker(_Blocker):
    """Block the Na/K-ATPase (ouabain-like). Value: blocked fraction, or a dose with ``ic50``."""

    kind = "Na/K-ATPase blocker"

    def __init__(self, region=None, max_block: float = 1.0, ic50: float | None = None,
                 hill: float = 1.0, max_dose: float | None = None, init: float | None = None,
                 spatial: bool = False, cost: float = 0.0):
        super().__init__("Na/K-ATPase", region, max_block, ic50, hill, max_dose, init, spatial, cost)

    def apply(self, mods: Modifiers) -> None:
        mods.nak_scale = mods.nak_scale * (1.0 - self.blocked_fraction())


class GapJunctionBlocker(_Blocker):
    """Block gap junctions (carbenoxolone-like). Value: blocked fraction, or a dose with ``ic50``."""

    kind = "gap junction blocker"

    def __init__(self, region=None, max_block: float = 1.0, ic50: float | None = None,
                 hill: float = 1.0, max_dose: float | None = None, init: float | None = None,
                 spatial: bool = False, cost: float = 0.0):
        super().__init__("gap junctions", region, max_block, ic50, hill, max_dose, init, spatial, cost)

    def apply(self, mods: Modifiers) -> None:
        mods.gj_scale = mods.gj_scale * (1.0 - self.blocked_fraction())


class Optogenetic(Intervention):
    """Light pattern for an expressed light-gated channel. Value: open probability in ``[0, 1]``.

    The opsin is a channel in ``tissue.channels`` (for example ``leak("Cl")`` for an
    anion channelrhodopsin, or a Na/K channel for a cation channelrhodopsin) whose
    ``max_d`` sets its expression. Light sets its open probability.
    """

    kind, unit = "optogenetic light", "open probability"

    def __init__(self, opsin: str, region=None, init: float = 0.05, spatial: bool = True, cost: float = 0.0):
        super().__init__(opsin, region, 0.0, 1.0, init, spatial, cost)

    def apply(self, mods: Modifiers) -> None:
        mods.channel_gate[self.target] = self.field()


@dataclass
class Protocol:
    """An optimized intervention protocol in lab terms.

    ``chronic=True``: the interventions stay on and the readout is the steady state.
    """

    steps: list[dict]
    t_stim: float
    t_relax: float
    baseline_error_mV: float
    final_error_mV: float
    chronic: bool = False

    @property
    def error_reduction(self) -> float:
        return 1.0 - self.final_error_mV / self.baseline_error_mV

    def to_markdown(self) -> str:
        head = (
            "Protocol: apply continuously; read out at the steady state."
            if self.chronic
            else f"Protocol: apply for {self.t_stim:g} s, then remove; read out at {self.t_relax:g} s."
        )
        lines = [
            head,
            "",
            "| # | Intervention | Target | Mean value | Max | Unit | Cells |",
            "|---|---|---|---|---|---|---|",
        ]
        for k, s in enumerate(self.steps, 1):
            cells = "all" if s["cells"] is None else s["cells"]
            lines.append(
                f"| {k} | {s['intervention']} | {s['target']} | {s['value']:.3g} | {s['max']:.3g} | {s['unit']} | {cells} |"
            )
        lines += [
            "",
            f"RMS pattern error: {self.baseline_error_mV:.2f} mV without intervention, "
            f"{self.final_error_mV:.2f} mV with it ({100 * self.error_reduction:.1f} % reduction).",
        ]
        return "\n".join(lines)


@dataclass
class InterventionResult:
    protocol: Protocol
    loss_history: list[float] = field(default_factory=list)
    error_history_mV: list[float] = field(default_factory=list)
    v_stim_end: torch.Tensor | None = None
    v_final: torch.Tensor | None = None
    v_baseline: torch.Tensor | None = None


class InterventionOptimizer:
    """Find intervention parameters that write a Target Morphology into an ion-level tissue.

    Loss = mean((V(T_relax) - V_target)^2) / v_scale^2 + sum_i cost_i * magnitude_i
    (plus a shaping term at T_stim, see :meth:`loss`). ``background`` holds the
    modifiers that apply when no intervention is active (for example a dark opsin:
    ``Modifiers(channel_gate={"ClLeak": 0})``); interventions are applied on top of
    it during ``[0, T_stim]``. The gradient is normalized by its largest entry before each Adam step (BPTT
    through a contracting attractor gives very small gradients).
    """

    def __init__(
        self,
        tissue: IonicTissue,
        initial: IonicState,
        v_target: torch.Tensor,
        interventions: list[Intervention],
        t_stim: float,
        t_relax: float,
        lr: float = 0.3,
        v_scale: float = 1e-3,
        weight: torch.Tensor | None = None,
        stim_end_weight: float = 0.1,
        background: Modifiers | None = None,
    ) -> None:
        if not 0.0 < t_stim < t_relax:
            raise ValueError("need 0 < t_stim < t_relax")
        self.tissue, self.initial = tissue, initial
        self.v_target = v_target.detach()
        self.interventions = interventions
        self.t_stim, self.t_relax = t_stim, t_relax
        self.v_scale = v_scale
        self.weight = weight
        self.stim_end_weight = stim_end_weight
        self.background = background or Modifiers()
        cell_shape = tuple(initial.x.shape[2:])
        self.params = [iv.build(cell_shape) for iv in interventions]
        self.optimizer = torch.optim.Adam(self.params, lr=lr)

    def modifiers(self) -> Modifiers:
        """Background modifiers with all interventions applied on top."""
        b = self.background
        mods = Modifiers(dict(b.channel_scale), dict(b.channel_add), dict(b.channel_gate),
                         b.nak_scale, b.ca_pump_scale, b.gj_scale)
        for iv in self.interventions:
            iv.apply(mods)
        return mods

    def rollout(self, active: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        """``(V(T_stim), V(T_relax))`` with (or without) the interventions."""
        mods = self.modifiers() if active else self.background
        mid = self.tissue.simulate(self.initial, self.t_stim, modifiers=mods)
        end = self.tissue.simulate(mid, self.t_relax - self.t_stim, modifiers=self.background)
        return self.tissue.vmem_of(mid), self.tissue.vmem_of(end)

    def pattern_error(self, v: torch.Tensor) -> torch.Tensor:
        d2 = (v - self.v_target) ** 2
        if self.weight is not None:
            return torch.sqrt((d2 * self.weight).sum() / self.weight.expand_as(d2).sum())
        return torch.sqrt(d2.mean())

    def loss(self, v_stim_end: torch.Tensor, v_final: torch.Tensor) -> torch.Tensor:
        """Normalized squared pattern error at T_relax, a shaping term at T_stim, and costs.

        The shaping term (weight ``stim_end_weight``) gives an informative gradient
        while the stimulus is too weak to switch the attractor. Success is always
        judged by the error at ``T_relax``.
        """
        err = (self.pattern_error(v_final) / self.v_scale) ** 2
        if self.stim_end_weight:
            err = err + self.stim_end_weight * (self.pattern_error(v_stim_end) / self.v_scale) ** 2
        return err + sum(iv.cost * iv.normalized_magnitude() for iv in self.interventions)

    def optimize(self, n_iters: int = 30, tol_mV: float | None = None, verbose: bool = False) -> InterventionResult:
        with torch.no_grad():
            _, v_base = self.rollout(active=False)
        base_err = float(self.pattern_error(v_base)) * 1e3
        result = InterventionResult(protocol=None, v_baseline=v_base)
        for it in range(n_iters):
            self.optimizer.zero_grad()
            v_mid, v_final = self.rollout()
            loss = self.loss(v_mid, v_final)
            loss.backward()
            scale = max(float(p.grad.abs().max()) for p in self.params if p.grad is not None)
            if scale > 0:
                for p in self.params:
                    if p.grad is not None:
                        p.grad.div_(scale)
            self.optimizer.step()
            err = float(self.pattern_error(v_final.detach())) * 1e3
            result.loss_history.append(loss.item())
            result.error_history_mV.append(err)
            if verbose:
                print(f"iter {it:3d}  loss {loss.item():.4e}  rms error {err:.3f} mV")
            if tol_mV is not None and err < tol_mV:
                break
        with torch.no_grad():
            v_mid, v_final = self.rollout()
        result.v_stim_end, result.v_final = v_mid, v_final
        result.protocol = Protocol(
            steps=[iv.describe() for iv in self.interventions],
            t_stim=self.t_stim,
            t_relax=self.t_relax,
            baseline_error_mV=base_err,
            final_error_mV=float(self.pattern_error(v_final)) * 1e3,
        )
        return result


class ChronicInterventionOptimizer:
    """Find constant interventions whose steady state is a Target Morphology.

    The interventions stay on. For each candidate, the tissue relaxes from
    ``initial`` to its steady state (:func:`bioelectrics.attractors.steady_state_implicit`),
    so the readout is the attractor that the start state reaches, at infinite time.
    Loss = pattern error^2 / v_scale^2 + sum_i cost_i * magnitude_i. Gradients use one
    sparse linear solve per iteration; the memory cost does not grow with time.

    When an intervention parameter moves the start state into another basin, the
    steady state jumps; the gradient describes the attractor on the current side of
    the jump.
    """

    def __init__(
        self,
        tissue: IonicTissue,
        initial: IonicState,
        v_target: torch.Tensor,
        interventions: list[Intervention],
        lr: float = 0.3,
        v_scale: float = 1e-3,
        weight: torch.Tensor | None = None,
        background: Modifiers | None = None,
        solver_options: dict | None = None,
    ) -> None:
        if initial.x.shape[0] != 1:
            raise ValueError("ChronicInterventionOptimizer needs batch size 1")
        self.tissue, self.initial = tissue, initial
        self.v_target = v_target.detach()
        self.interventions = interventions
        self.v_scale, self.weight = v_scale, weight
        self.background = background or Modifiers()
        self.solver_options = solver_options or {}
        cell_shape = tuple(initial.x.shape[2:])
        self.params = [iv.build(cell_shape) for iv in interventions]
        self.optimizer = torch.optim.Adam(self.params, lr=lr)

    modifiers = InterventionOptimizer.modifiers
    pattern_error = InterventionOptimizer.pattern_error

    def steady_vmem(self, active: bool = True) -> torch.Tensor:
        """V_mem of the steady state reached from ``initial`` (differentiable when ``active``)."""
        from .attractors import find_steady_state, steady_state_implicit

        if not active:
            ss = find_steady_state(self.tissue, self.initial, self.background, stability=False,
                                   **self.solver_options)
            return ss.vmem
        x = steady_state_implicit(self.tissue, self.initial, self.modifiers, self.params, **self.solver_options)
        return self.tissue.vmem_of(x)

    def loss(self, v: torch.Tensor) -> torch.Tensor:
        err = (self.pattern_error(v) / self.v_scale) ** 2
        return err + sum(iv.cost * iv.normalized_magnitude() for iv in self.interventions)

    def optimize(self, n_iters: int = 30, tol_mV: float | None = None, verbose: bool = False) -> InterventionResult:
        v_base = self.steady_vmem(active=False)
        base_err = float(self.pattern_error(v_base)) * 1e3
        result = InterventionResult(protocol=None, v_baseline=v_base)
        for it in range(n_iters):
            self.optimizer.zero_grad()
            v = self.steady_vmem()
            loss = self.loss(v)
            loss.backward()
            scale = max(float(p.grad.abs().max()) for p in self.params if p.grad is not None)
            if scale > 0:
                for p in self.params:
                    if p.grad is not None:
                        p.grad.div_(scale)
            self.optimizer.step()
            err = float(self.pattern_error(v.detach())) * 1e3
            result.loss_history.append(loss.item())
            result.error_history_mV.append(err)
            if verbose:
                print(f"iter {it:3d}  loss {loss.item():.4e}  rms error {err:.3f} mV")
            if tol_mV is not None and err < tol_mV:
                break
        with torch.no_grad():
            v = self.steady_vmem().detach()
        result.v_final = v
        result.protocol = Protocol(
            steps=[iv.describe() for iv in self.interventions],
            t_stim=float("inf"),
            t_relax=float("inf"),
            baseline_error_mV=base_err,
            final_error_mV=float(self.pattern_error(v)) * 1e3,
            chronic=True,
        )
        return result
