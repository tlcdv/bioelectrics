"""Inverse bioelectric pattern control by backpropagation through time.

:class:`PatternOptimizer` is an error-minimizing homeostatic controller. It finds a
brief spatiotemporal stimulus ``I_ext(t, x, y)`` that moves a tissue from a
corrupted basin of attraction into the target morphogenetic attractor.

Every rollout has two phases:

* ``[0, T_stim]``: active intervention with the optimized stimulus.
* ``[T_stim, T_relax]``: zero external stimulus.

The pattern error is measured at ``T_relax``, after the stimulus has ended. A low
error therefore proves a permanent attractor switch (a "subroutine call"), not a
forced clamp of the cells.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .constants import DTYPE
from .integrators import integrate
from .tissue import BioelectricState, Tissue


@dataclass
class OptimizationResult:
    """Outcome of :meth:`PatternOptimizer.optimize`. Errors are RMS values in volts."""

    baseline_error: float
    final_error: float
    loss_history: list[float] = field(default_factory=list)
    error_history: list[float] = field(default_factory=list)
    stimulus: torch.Tensor | None = None  # (n_time_bins, *state_shape) [A]
    v_stim_end: torch.Tensor | None = None  # V(T_stim) [V]
    v_final: torch.Tensor | None = None  # V(T_relax) [V]

    @property
    def error_reduction(self) -> float:
        """Fraction of the baseline RMS pattern error that the stimulus removes."""
        return 1.0 - self.final_error / self.baseline_error


class PatternOptimizer:
    """Optimize a stimulus ``I_ext(t, x, y)`` that writes a Target Morphology into a tissue.

    Parameters
    ----------
    tissue:
        The tissue hardware.
    v_initial:
        Initial ``V_mem`` [V], shape ``(batch, H, W)`` or ``(batch, N_cells)``.
    v_target:
        Target morphogenetic ``V_mem`` pattern [V], broadcastable to ``v_initial``.
    t_stim, t_relax:
        End of the stimulus phase and end of the relaxation phase [s].
    n_time_bins:
        The stimulus is piecewise constant in time over this many bins in ``[0, t_stim]``.
    i_scale:
        Current scale [A] of the optimized parameter: ``I_ext = u * i_scale``.
    stim_penalty:
        Weight of the L2 stimulus energy term (in units of ``i_scale``).
    stim_end_weight:
        Weight of an extra error term at ``T_stim``. It gives a strong gradient during
        the stimulus. The success criterion is always the error at ``T_relax``.
    """

    def __init__(
        self,
        tissue: Tissue,
        v_initial: torch.Tensor,
        v_target: torch.Tensor,
        t_stim: float = 0.1,
        t_relax: float = 0.5,
        dt: float = 5e-4,
        n_time_bins: int = 4,
        lr: float = 2.0,
        i_scale: float = 1e-12,
        stim_penalty: float = 1e-4,
        stim_end_weight: float = 0.0,
        v_scale: float = 1e-3,
    ) -> None:
        if not 0.0 < t_stim < t_relax:
            raise ValueError("need 0 < t_stim < t_relax")
        self.tissue = tissue
        self.v_initial = v_initial.detach()
        self.v_target = torch.as_tensor(v_target, dtype=v_initial.dtype).detach()
        self.t_stim, self.t_relax, self.dt = t_stim, t_relax, dt
        self.n_time_bins = n_time_bins
        self.i_scale = i_scale
        self.stim_penalty = stim_penalty
        self.stim_end_weight = stim_end_weight
        self.v_scale = v_scale
        self.u = torch.zeros((n_time_bins, *v_initial.shape), dtype=DTYPE, requires_grad=True)
        # eps is far below the default because BPTT through a contracting attractor
        # gives very small (but correctly signed) gradients.
        self.optimizer = torch.optim.Adam([self.u], lr=lr, eps=1e-30)

    # ------------------------------------------------------------------ rollout
    @property
    def stimulus(self) -> torch.Tensor:
        """Stimulus tensor ``(n_time_bins, *state_shape)`` [A]."""
        return self.u * self.i_scale

    def _stimulus_fn(self, stim: torch.Tensor):
        bin_len = self.t_stim / self.n_time_bins

        def fn(t: float) -> torch.Tensor:
            k = min(int(t / bin_len), self.n_time_bins - 1)
            return stim[k]

        return fn

    def rollout(self, stim: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(V(T_stim), V(T_relax))`` for stimulus ``stim`` (default: current)."""
        stim = self.stimulus if stim is None else stim
        stim_fn = self._stimulus_fn(stim)
        rhs_on = lambda t, v: self.tissue.rhs(v, stim_fn(t))
        v_stim_end, _, _ = integrate(rhs_on, self.v_initial, self.t_stim, self.dt)
        rhs_off = lambda t, v: self.tissue.rhs(v)  # zero external stimulus
        v_final, _, _ = integrate(rhs_off, v_stim_end, self.t_relax, self.dt, t0=self.t_stim)
        return v_stim_end, v_final

    def pattern_error(self, v: torch.Tensor) -> torch.Tensor:
        """RMS difference between ``v`` and the Target Morphology [V]."""
        return torch.sqrt(torch.mean((v - self.v_target) ** 2))

    def loss(self, v_stim_end: torch.Tensor, v_final: torch.Tensor) -> torch.Tensor:
        """Dimensionless loss: normalized squared pattern error plus stimulus energy."""
        err = torch.mean(((v_final - self.v_target) / self.v_scale) ** 2)
        if self.stim_end_weight:
            err = err + self.stim_end_weight * torch.mean(
                ((v_stim_end - self.v_target) / self.v_scale) ** 2
            )
        return err + self.stim_penalty * torch.mean(self.u**2)

    # ------------------------------------------------------------- optimization
    def optimize(self, n_iters: int = 50, tol: float | None = None, verbose: bool = False) -> OptimizationResult:
        """Run Adam for ``n_iters`` iterations. Stop early if RMS error < ``tol`` [V]."""
        with torch.no_grad():
            _, v_base = self.rollout(torch.zeros_like(self.stimulus))
            baseline = self.pattern_error(v_base).item()
        result = OptimizationResult(baseline_error=baseline, final_error=baseline)
        for it in range(n_iters):
            self.optimizer.zero_grad()
            v_stim_end, v_final = self.rollout()
            loss = self.loss(v_stim_end, v_final)
            loss.backward()
            self.optimizer.step()
            err = self.pattern_error(v_final).item()
            result.loss_history.append(loss.item())
            result.error_history.append(err)
            if verbose:
                print(f"iter {it:3d}  loss {loss.item():.4e}  rms error {err * 1e3:.3f} mV")
            if tol is not None and err < tol:
                break
        with torch.no_grad():
            v_stim_end, v_final = self.rollout()
        result.final_error = self.pattern_error(v_final).item()
        result.stimulus = self.stimulus.detach().clone()
        result.v_stim_end = v_stim_end
        result.v_final = v_final
        return result

    def final_state(self) -> BioelectricState:
        """``BioelectricState`` at ``T_relax`` under the current stimulus."""
        with torch.no_grad():
            _, v_final = self.rollout()
        return BioelectricState(v_final, self.t_relax)
