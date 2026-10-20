"""Differentiable ODE solvers in PyTorch: fixed-step Euler/RK4 and adaptive Dormand-Prince.

The Python loop runs over time steps only. Each step is a vectorized tensor
operation over all cells and batch elements, so gradients flow through the full
trajectory (backpropagation through time). In the adaptive solver the step-size
decisions are not differentiated (they are detached), which is the standard
"discretize-then-optimize" treatment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

RHS = Callable[[float, torch.Tensor], torch.Tensor]


def euler_step(rhs: RHS, t: float, v: torch.Tensor, dt: float) -> torch.Tensor:
    """One explicit Euler step."""
    return v + dt * rhs(t, v)


def rk4_step(rhs: RHS, t: float, v: torch.Tensor, dt: float) -> torch.Tensor:
    """One classical 4th-order Runge-Kutta step."""
    k1 = rhs(t, v)
    k2 = rhs(t + 0.5 * dt, v + 0.5 * dt * k1)
    k3 = rhs(t + 0.5 * dt, v + 0.5 * dt * k2)
    k4 = rhs(t + dt, v + dt * k3)
    return v + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


_STEPPERS = {"euler": euler_step, "rk4": rk4_step}


def integrate(
    rhs: RHS,
    v0: torch.Tensor,
    t_end: float,
    dt: float,
    t0: float = 0.0,
    method: str = "rk4",
    record_every: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Integrate ``dV/dt = rhs(t, V)`` from ``t0`` to ``t_end`` with step ``dt`` [s].

    Returns ``(v_final, times, trajectory)``. ``times`` and ``trajectory`` are
    ``None`` unless ``record_every`` is set; then every ``record_every``-th state
    (and the initial state) is stacked along a new leading dimension.

    Raises ``FloatingPointError`` if the state becomes non-finite (step too large
    for a stiff system); use :func:`integrate_adaptive` in that case.
    """
    step = _STEPPERS[method]
    n_steps = int(round((t_end - t0) / dt))
    v = v0
    times, traj = ([t0], [v0]) if record_every else (None, None)
    for k in range(n_steps):
        t = t0 + k * dt
        v = step(rhs, t, v, dt)
        if (k + 1) % 50 == 0 or k + 1 == n_steps:
            if not torch.isfinite(v).all():
                raise FloatingPointError(
                    f"non-finite state at t = {t + dt:.4g} s with dt = {dt:.3g} s; "
                    "reduce dt or use method='dopri5'"
                )
        if record_every and (k + 1) % record_every == 0:
            times.append(t + dt)
            traj.append(v)
    if record_every:
        return v, torch.tensor(times, dtype=v0.dtype), torch.stack(traj)
    return v, None, None


# Dormand-Prince 5(4) Butcher tableau.
_C = (0.0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0)
_A = (
    (),
    (1 / 5,),
    (3 / 40, 9 / 40),
    (44 / 45, -56 / 15, 32 / 9),
    (19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729),
    (9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656),
    (35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84),
)
_B5 = (35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0.0)
_B4 = (5179 / 57600, 0.0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40)
_E = tuple(b5 - b4 for b5, b4 in zip(_B5, _B4))


#: ROS2 parameter gamma = 1 + 1/sqrt(2) (Verwer, Spee, Blom & Hundsdorfer, SIAM J. Sci. Comput. 20, 1999).
_ROS2_GAMMA = 1.0 + 2.0**-0.5


def integrate_rosenbrock(
    rhs: RHS,
    v0: torch.Tensor,
    t_end: float,
    make_solver: Callable[[float, torch.Tensor, float], Callable[[torch.Tensor], torch.Tensor]],
    t0: float = 0.0,
    rtol: float | torch.Tensor = 1e-6,
    atol: float | torch.Tensor = 1e-9,
    dt0: float = 1e-4,
    dt_max: float = float("inf"),
    t_eval: list[float] | None = None,
    max_steps: int = 10_000_000,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, "SolverStats"]:
    """Adaptive linearly implicit ROS2 integration for stiff systems.

    ``make_solver(t, v, gamma_h)`` returns a function ``b -> (I - gamma_h J)^-1 b``
    for an (approximate) Jacobian ``J`` of ``rhs`` at ``(t, v)``. ROS2 is a
    W-method: it keeps order 2 with an approximate Jacobian, so ``J`` may omit
    weak couplings and should be detached from the autograd graph. Gradients flow
    through ``rhs`` and the linear solves. The error estimate compares ROS2 with
    linearly implicit Euler.

    Returns ``(v_final, times, trajectory, stats)``.
    """
    g = _ROS2_GAMMA
    t, v, h = t0, v0, min(dt0, t_end - t0)
    stats = SolverStats()
    outputs = sorted(x for x in (t_eval or []) if t0 <= x <= t_end)
    out_t, out_v = [], []
    while outputs and outputs[0] <= t0:
        out_t.append(outputs.pop(0))
        out_v.append(v0)
    f0 = rhs(t, v)
    stats.n_rhs += 1
    while t < t_end - 1e-15 * max(1.0, abs(t_end)):
        if stats.n_accepted + stats.n_rejected >= max_steps:
            raise RuntimeError("integrate_rosenbrock: max_steps reached")
        target = min(t_end, outputs[0]) if outputs else t_end
        h_free = min(h, dt_max)
        h = min(h_free, target - t)
        clipped = h < h_free
        solve = make_solver(t, v.detach(), g * h)
        k1 = solve(f0)
        f1 = rhs(t + h, v + h * k1)
        k2 = solve(f1 - 2.0 * k1)
        stats.n_rhs += 1
        v_new = v + h * (1.5 * k1 + 0.5 * k2)
        err = 0.5 * h * (k1 + k2)
        with torch.no_grad():
            scale = atol + rtol * torch.maximum(v.abs(), v_new.abs())
            ratio = (err / scale).abs()
            ok = bool(torch.isfinite(ratio).all()) and bool(torch.isfinite(v_new).all())
            err_norm = float(torch.sqrt(torch.mean(ratio**2))) if ok else float("inf")
        if err_norm <= 1.0:
            t = t + h
            v = v_new
            f0 = rhs(t, v)
            stats.n_rhs += 1
            stats.n_accepted += 1
            stats.dt_min, stats.dt_max = min(stats.dt_min, h), max(stats.dt_max, h)
            if outputs and abs(t - outputs[0]) <= 1e-12 * max(1.0, abs(t)):
                out_t.append(outputs.pop(0))
                out_v.append(v)
            factor = 5.0 if err_norm == 0.0 else min(5.0, 0.9 * err_norm ** (-0.5))
        else:
            stats.n_rejected += 1
            factor = 0.2 if err_norm == float("inf") else max(0.2, 0.9 * err_norm ** (-0.5))
        h = max(h_free, h * factor) if (clipped and err_norm <= 1.0) else h * factor
        if h < 1e-14:
            raise FloatingPointError(f"integrate_rosenbrock: step size underflow at t = {t:.4g} s")
    if t_eval:
        return v, torch.tensor(out_t, dtype=v0.dtype), torch.stack(out_v), stats
    return v, None, None, stats


@dataclass
class SolverStats:
    n_accepted: int = 0
    n_rejected: int = 0
    n_rhs: int = 0
    dt_min: float = float("inf")
    dt_max: float = 0.0


def integrate_adaptive(
    rhs: RHS,
    v0: torch.Tensor,
    t_end: float,
    t0: float = 0.0,
    rtol: float | torch.Tensor = 1e-6,
    atol: float | torch.Tensor = 1e-9,
    dt0: float = 1e-4,
    dt_max: float = float("inf"),
    t_eval: list[float] | None = None,
    max_steps: int = 10_000_000,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, SolverStats]:
    """Adaptive Dormand-Prince 5(4) integration with error control (FSAL).

    ``rtol`` and ``atol`` may be tensors that broadcast against the state, so
    each state component can have its own tolerance. Steps land exactly on every
    time in ``t_eval``; the states at those times are returned as a trajectory.

    Returns ``(v_final, times, trajectory, stats)``.
    """
    t, v, h = t0, v0, min(dt0, t_end - t0)
    stats = SolverStats()
    outputs = sorted(x for x in (t_eval or []) if t0 <= x <= t_end)
    out_t, out_v = [], []
    while outputs and outputs[0] <= t0:
        out_t.append(outputs.pop(0))
        out_v.append(v0)
    k1 = rhs(t, v)
    stats.n_rhs += 1
    while t < t_end - 1e-15 * max(1.0, abs(t_end)):
        if stats.n_accepted + stats.n_rejected >= max_steps:
            raise RuntimeError("integrate_adaptive: max_steps reached")
        target = min(t_end, outputs[0]) if outputs else t_end
        h_free = min(h, dt_max)
        h = min(h_free, target - t)
        clipped = h < h_free
        ks = [k1]
        for i in range(1, 7):
            vi = v + h * sum(a * k for a, k in zip(_A[i], ks))
            ks.append(rhs(t + _C[i] * h, vi))
        stats.n_rhs += 6
        v_new = v + h * sum(b * k for b, k in zip(_B5, ks) if b != 0.0)
        err = h * sum(e * k for e, k in zip(_E, ks) if e != 0.0)
        with torch.no_grad():
            scale = atol + rtol * torch.maximum(v.abs(), v_new.abs())
            ratio = (err / scale).abs()
            err_norm = float(torch.sqrt(torch.mean(ratio**2))) if torch.isfinite(ratio).all() else float("inf")
        if err_norm <= 1.0:
            t = t + h
            v = v_new
            k1 = ks[6]  # FSAL
            stats.n_accepted += 1
            stats.dt_min, stats.dt_max = min(stats.dt_min, h), max(stats.dt_max, h)
            if outputs and abs(t - outputs[0]) <= 1e-12 * max(1.0, abs(t)):
                out_t.append(outputs.pop(0))
                out_v.append(v)
            factor = 5.0 if err_norm == 0.0 else min(5.0, 0.9 * err_norm ** (-0.2))
        else:
            stats.n_rejected += 1
            factor = 0.2 if err_norm == float("inf") else max(0.2, 0.9 * err_norm ** (-0.2))
        h = max(h_free, h * factor) if (clipped and err_norm <= 1.0) else h * factor
        if h < 1e-14:
            raise FloatingPointError(f"integrate_adaptive: step size underflow at t = {t:.4g} s")
    if t_eval:
        return v, torch.tensor(out_t, dtype=v0.dtype), torch.stack(out_v), stats
    return v, None, None, stats
