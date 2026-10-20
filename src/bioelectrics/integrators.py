"""Differentiable fixed-step ODE solvers in PyTorch.

The Python loop runs over time steps only. Each step is a vectorized tensor
operation over all cells and batch elements, so gradients flow through the full
trajectory (backpropagation through time).
"""

from __future__ import annotations

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
    """
    step = _STEPPERS[method]
    n_steps = int(round((t_end - t0) / dt))
    v = v0
    times, traj = ([t0], [v0]) if record_every else (None, None)
    for k in range(n_steps):
        t = t0 + k * dt
        v = step(rhs, t, v, dt)
        if record_every and (k + 1) % record_every == 0:
            times.append(t + dt)
            traj.append(v)
    if record_every:
        return v, torch.tensor(times, dtype=v0.dtype), torch.stack(traj)
    return v, None, None
