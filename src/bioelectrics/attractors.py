"""Attractors of ion-level tissues: steady states, stability, basins and implicit gradients.

A bioelectric pattern memory is a stable steady state of the tissue dynamics
``dx/dt = f(x, theta)``. This module finds such states directly, without long
simulations, and differentiates them with the implicit function theorem.

* :func:`find_steady_state` uses pseudo-transient continuation: implicit Euler
  steps ``(I/h - J) dx = f`` with a step ``h`` that grows to infinity (Newton's
  method in the limit). For small ``h`` the iteration follows the dynamics, so it
  converges to the attractor of the basin that contains the start state, not to
  a nearby saddle. ``J`` is the exact sparse Jacobian from forward-mode autograd;
  a distance-2 coloring of the cell graph needs only ``5 n`` Jacobian-vector
  products on a grid (``n`` = state components per cell), for any tissue size.
* :func:`steady_state_implicit` returns the steady state with gradients for any
  parameter ``theta``: ``dL/dtheta = -lambda^T df/dtheta`` with
  ``J^T lambda = dL/dx*``. One sparse linear solve replaces backpropagation through
  the whole relaxation.
* :func:`enumerate_attractors`, :func:`dose_response` and :func:`basin_map` map the
  attractor landscape.

State components whose time derivative is identically zero (for example
impermeant proteins) are held fixed and removed from the linear systems.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
from torch.func import jvp

from .constants import DTYPE
from .ionic.model import NO_MODIFIERS, IonicState, IonicTissue, Modifiers

__all__ = [
    "SteadyState",
    "Attractor",
    "sparse_jacobian",
    "find_steady_state",
    "steady_state_implicit",
    "enumerate_attractors",
    "random_vmem_states",
    "dose_response",
    "basin_map",
]


# ------------------------------------------------------------------ Jacobian
def _cell_edges(tissue: IonicTissue, cells: tuple[int, ...]) -> np.ndarray:
    """Undirected cell-cell edges ``(2, E)`` of the coupling graph (flat cell indices)."""
    if tissue.edges is not None:
        return tissue.edges.detach().cpu().numpy().astype(np.int64).reshape(2, -1)
    if len(cells) != 2:
        raise ValueError("a tissue without explicit edges needs a 2D grid (H, W)")
    idx = np.arange(cells[0] * cells[1]).reshape(cells)
    horiz = np.stack([idx[:, :-1].ravel(), idx[:, 1:].ravel()])
    vert = np.stack([idx[:-1, :].ravel(), idx[1:, :].ravel()])
    return np.concatenate([horiz, vert], axis=1)


def _coloring(tissue: IonicTissue, cells: tuple[int, ...], edges: np.ndarray) -> np.ndarray:
    """Distance-2 coloring: cells of one color have disjoint closed neighbourhoods."""
    n_cells = int(np.prod(cells))
    if tissue.edges is None:
        r, c = np.meshgrid(np.arange(cells[0]), np.arange(cells[1]), indexing="ij")
        return ((r + 2 * c) % 5).ravel()  # 4-neighbour grid: 5 colors suffice
    nbrs: list[set[int]] = [set() for _ in range(n_cells)]
    for a, b in edges.T:  # setup only, not ODE evaluation
        nbrs[a].add(int(b))
        nbrs[b].add(int(a))
    color = -np.ones(n_cells, dtype=np.int64)
    for p in range(n_cells):
        used = {color[q] for q in nbrs[p]}
        for q in nbrs[p]:
            used |= {color[s] for s in nbrs[q]}
        k = 0
        while k in used:
            k += 1
        color[p] = k
    return color


def _source_table(color: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """``src[k, p]`` = the cell of color ``k`` in the closed neighbourhood of ``p`` (or -1)."""
    n_colors, n_cells = int(color.max()) + 1, color.size
    src = -np.ones((n_colors, n_cells), dtype=np.int64)
    src[color, np.arange(n_cells)] = np.arange(n_cells)
    a, b = edges
    src[color[b], a] = b
    src[color[a], b] = a
    return src


def sparse_jacobian(
    tissue: IonicTissue, x: torch.Tensor, modifiers: Modifiers = NO_MODIFIERS
) -> sp.csr_matrix:
    """Exact Jacobian ``df/dx`` of ``tissue.rhs`` at ``x`` (batch size 1) as a sparse matrix.

    Rows and columns use the flat index ``component * n_cells + cell``.
    """
    if x.shape[0] != 1:
        raise ValueError("sparse_jacobian needs batch size 1")
    n_comp, cells = x.shape[1], tuple(x.shape[2:])
    n_cells = int(np.prod(cells))
    edges = _cell_edges(tissue, cells)
    color = _coloring(tissue, cells, edges)
    src = _source_table(color, edges)
    n_colors = src.shape[0]
    color_t = torch.as_tensor(color).reshape(cells)
    x = x.detach()
    # All seeds at once: the batch dimension of rhs carries one seed per (color, component).
    seeds = torch.zeros((n_colors, n_comp, n_comp, *cells), dtype=x.dtype)
    masks = (color_t.unsqueeze(0) == torch.arange(n_colors).reshape(-1, *([1] * len(cells)))).to(x.dtype)
    idx = torch.arange(n_comp)
    seeds[:, idx, idx] = masks.unsqueeze(1)
    seeds = seeds.reshape(n_colors * n_comp, n_comp, *cells)
    xb = x.expand(seeds.shape[0], *x.shape[1:]).contiguous()
    with torch.no_grad():
        _, df = jvp(lambda z: tissue.rhs(z, modifiers), (xb,), (seeds,))
    df = df.reshape(n_colors, n_comp, n_comp, n_cells).cpu().numpy()  # (color, j, i, cell)
    rows, cols, vals = [], [], []
    comp = np.arange(n_comp)[:, None]
    for k in range(n_colors):  # colors, not cells
        have = src[k] >= 0
        p = np.nonzero(have)[0]
        q = src[k][have]
        for j in range(n_comp):
            d = df[k, j][:, p]  # (n_comp, len(p))
            rows.append((comp * n_cells + p[None, :]).ravel())
            cols.append(np.broadcast_to(j * n_cells + q[None, :], d.shape).ravel())
            vals.append(d.ravel())
    n = n_comp * n_cells
    jac = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    jac = jac.tocsr()
    jac.eliminate_zeros()
    return jac


def _active_components(jac: sp.csr_matrix, n_comp: int, n_cells: int) -> np.ndarray:
    """Flat indices of components that change in time (nonzero Jacobian rows)."""
    row_nnz = np.diff(jac.indptr).reshape(n_comp, n_cells).sum(axis=1)
    comps = np.nonzero(row_nnz > 0)[0]
    return (comps[:, None] * n_cells + np.arange(n_cells)[None, :]).ravel()


# --------------------------------------------------------------- steady states
@dataclass
class SteadyState:
    """Result of :func:`find_steady_state` (one entry per batch element)."""

    state: IonicState
    vmem: torch.Tensor
    stable: list[bool | None]
    max_real_eig: list[float | None]
    slowest_time_s: list[float | None]
    n_iter: list[int]
    residual: list[float]


def _solve_one(
    tissue: IonicTissue,
    x0: torch.Tensor,
    mods: Modifiers,
    h0: float,
    growth: float,
    dv_max: float,
    max_iter: int,
    conc_tol: float,
    gate_tol: float,
    dc_rel_max: float = 0.25,
):
    """Pseudo-transient continuation for one batch element. Returns ``(x, jac, active, iters)``."""
    n_comp, cells = x0.shape[1], tuple(x0.shape[2:])
    n_cells = int(np.prod(cells))
    tol = np.full(n_comp, gate_tol)
    tol[: tissue.n_ions] = conc_tol
    tol = np.repeat(tol, n_cells)
    x = x0.detach().clone()
    h = h0
    v_old = tissue.vmem(x[:, : tissue.n_ions])
    for it in range(1, max_iter + 1):
        with torch.no_grad():
            f = tissue.rhs(x, mods).reshape(-1).cpu().numpy()
        jac = sparse_jacobian(tissue, x, mods)
        act = _active_components(jac, n_comp, n_cells)
        j_aa = jac[act][:, act]
        m = sp.identity(act.size, format="csc") / h - j_aa.tocsc()
        try:
            dx_a = spla.splu(m).solve(f[act])
        except RuntimeError:  # singular for this h
            h /= 4.0
            continue
        dx = np.zeros(x.numel())
        dx[act] = dx_a
        x_new = x + torch.as_tensor(dx, dtype=x.dtype).reshape(x.shape)
        v_new = tissue.vmem(x_new[:, : tissue.n_ions])
        c_old, c_new = x[:, : tissue.n_ions], x_new[:, : tissue.n_ions]
        gates = x_new[:, tissue.n_ions :]
        ok = (
            bool(torch.isfinite(x_new).all())
            and float((v_new - v_old).abs().max()) <= dv_max
            and bool((c_new >= 0).all())  # concentrations stay physical
            and bool(((c_new - c_old).abs() <= dc_rel_max * c_old + 1e-9).all())
            and bool(((gates >= -1e-3) & (gates <= 1 + 1e-3)).all())
        )
        if not ok:
            h /= 4.0
            if h < 1e-12:
                raise FloatingPointError("steady-state search failed: step size underflow")
            continue
        x, v_old = x_new, v_new
        if h >= 1e8 and np.all(np.abs(dx) <= tol):
            return x, sparse_jacobian(tissue, x, mods), act, it
        h *= growth
    raise RuntimeError(f"steady-state search did not converge in {max_iter} iterations")


def _stability(jac: sp.csr_matrix, act: np.ndarray, dense_limit: int):
    """``(max Re(eig), slowest time constant [s])`` of the active Jacobian, or ``(None, None)``."""
    if act.size > dense_limit:
        return None, None
    ev = np.linalg.eigvals(jac[act][:, act].toarray()).real
    neg = ev[ev < 0]
    slow = float(1.0 / np.abs(neg).min()) if neg.size else None
    return float(ev.max()), slow


def find_steady_state(
    tissue: IonicTissue,
    state: IonicState,
    modifiers: Modifiers | None = None,
    *,
    h0: float = 1e-3,
    growth: float = 3.0,
    dv_max: float = 0.01,
    max_iter: int = 300,
    conc_tol: float = 1e-10,
    gate_tol: float = 1e-8,
    stability: bool = True,
    dense_limit: int = 3000,
) -> SteadyState:
    """Steady state reached from ``state`` under constant ``modifiers``.

    Parameters
    ----------
    h0, growth: first pseudo-time step [s] and its growth factor per accepted step.
    dv_max: largest V_mem change [V] in any cell allowed in one step. Small values
        keep the iteration in the basin of the start state. A step is also rejected
        when a concentration becomes negative or changes by more than 25 %, or a gate
        leaves [0, 1].
    conc_tol, gate_tol: convergence limits for the last Newton step
        (``1e-10`` mol/m^3 is about 2 uV of V_mem).
    stability: compute the eigenvalues of ``J`` (dense, up to ``dense_limit`` unknowns).

    Returns
    -------
    :class:`SteadyState`. ``stable`` is ``True`` when all eigenvalues of ``J`` have
    negative real parts. ``slowest_time_s`` is ``1 / min |Re(eig)|``: the time scale of
    the slowest approach (often the slow exchange of ions with the bath).
    """
    mods = modifiers or NO_MODIFIERS
    xs, stable, mre, slow, iters, res = [], [], [], [], [], []
    for b in range(state.x.shape[0]):  # batch elements, not cells
        x, jac, act, it = _solve_one(tissue, state.x[b : b + 1], mods, h0, growth, dv_max, max_iter,
                                     conc_tol, gate_tol)
        xs.append(x)
        iters.append(it)
        with torch.no_grad():
            res.append(float(tissue.rhs(x, mods).abs().max()))
        if stability:
            e_max, t_slow = _stability(jac, act, dense_limit)
            stable.append(None if e_max is None else e_max < 0.0)
            mre.append(e_max)
            slow.append(t_slow)
        else:
            stable.append(None)
            mre.append(None)
            slow.append(None)
    out = IonicState(torch.cat(xs), float("inf"))
    return SteadyState(out, tissue.vmem_of(out), stable, mre, slow, iters, res)


# ----------------------------------------------------------- implicit gradient
class _ImplicitSteadyState(torch.autograd.Function):
    @staticmethod
    def forward(ctx, closure, *params):
        ctx.closure = closure
        return closure["x"].clone()

    @staticmethod
    def backward(ctx, grad_x):
        c = ctx.closure
        g = grad_x.detach().reshape(-1).cpu().numpy()
        act = c["active"]
        lam = np.zeros_like(g)
        lam[act] = c["lu"].solve(g[act], trans="T")
        lam_t = torch.as_tensor(lam, dtype=grad_x.dtype).reshape(grad_x.shape)
        with torch.enable_grad():
            fx = c["tissue"].rhs(c["x"], c["modifiers_fn"]())
            grads = torch.autograd.grad(fx, c["params"], grad_outputs=-lam_t, allow_unused=True)
        return (None, *grads)


def steady_state_implicit(
    tissue: IonicTissue,
    state: IonicState,
    modifiers_fn: Callable[[], Modifiers] | None,
    params: Sequence[torch.Tensor],
    **kwargs,
) -> IonicState:
    """Steady state from ``state`` that is differentiable with respect to ``params``.

    ``modifiers_fn()`` must build the modifiers from ``params`` (it is called again in
    the backward pass, so the autograd graph to ``params`` is rebuilt there). Tissue
    parameters that ``tissue.rhs`` reads directly (for example a ``max_d`` tensor
    with ``requires_grad=True``) can also be listed in ``params``. Batch size 1.

    The gradient follows the implicit function theorem at ``f(x*, theta) = 0``; it is
    the derivative of the attractor that the start state relaxes to. It does not
    include the jump when a parameter change moves the start state into another basin.
    """
    if state.x.shape[0] != 1:
        raise ValueError("steady_state_implicit needs batch size 1")
    modifiers_fn = modifiers_fn or (lambda: NO_MODIFIERS)
    with torch.no_grad():
        mods = modifiers_fn()
    x, jac, act, _ = _solve_one(
        tissue, state.x, _detach(mods), kwargs.get("h0", 1e-3), kwargs.get("growth", 3.0),
        kwargs.get("dv_max", 0.01), kwargs.get("max_iter", 300), kwargs.get("conc_tol", 1e-10),
        kwargs.get("gate_tol", 1e-8),
    )
    lu = spla.splu(jac[act][:, act].tocsc())
    closure = {"x": x.detach(), "active": act, "lu": lu, "tissue": tissue,
               "modifiers_fn": modifiers_fn, "params": list(params)}
    x_star = _ImplicitSteadyState.apply(closure, *params)
    return IonicState(x_star, float("inf"))


def _detach(m: Modifiers) -> Modifiers:
    d = lambda v: v.detach() if isinstance(v, torch.Tensor) else v
    return Modifiers({k: d(v) for k, v in m.channel_scale.items()}, {k: d(v) for k, v in m.channel_add.items()},
                     {k: d(v) for k, v in m.channel_gate.items()}, d(m.nak_scale), d(m.ca_pump_scale),
                     d(m.gj_scale))


# ------------------------------------------------------------------ landscape
@dataclass
class Attractor:
    """One distinct stable pattern found by :func:`enumerate_attractors`."""

    vmem: torch.Tensor  # (*cells) [V]
    state: IonicState
    count: int  # number of start states that reached it
    stable: bool | None
    slowest_time_s: float | None


def random_vmem_states(
    tissue: IonicTissue,
    cells: tuple[int, ...],
    n: int,
    v_range: tuple[float, float] = (-0.09, 0.0),
    seed: int = 0,
) -> IonicState:
    """``n`` start states with independent uniform random V_mem per cell [V]."""
    g = torch.Generator().manual_seed(seed)
    v = v_range[0] + (v_range[1] - v_range[0]) * torch.rand((n, *cells), generator=g, dtype=DTYPE)
    return tissue.with_vmem(tissue.initial_state(cells, batch=n), v)


def enumerate_attractors(
    tissue: IonicTissue,
    starts: IonicState,
    modifiers: Modifiers | None = None,
    v_tol: float = 1e-3,
    **kwargs,
) -> list[Attractor]:
    """Relax every start state and group the steady states whose V_mem differ by < ``v_tol``.

    Returns the attractors sorted by ``count`` (largest basin sample first).
    """
    ss = find_steady_state(tissue, starts, modifiers, **kwargs)
    found: list[Attractor] = []
    for b in range(starts.x.shape[0]):
        v = ss.vmem[b]
        for a in found:
            if float((a.vmem - v).abs().max()) < v_tol:
                a.count += 1
                break
        else:
            found.append(Attractor(v, IonicState(ss.state.x[b : b + 1], float("inf")), 1, ss.stable[b],
                                   ss.slowest_time_s[b]))
    return sorted(found, key=lambda a: -a.count)


def dose_response(
    tissue: IonicTissue,
    state: IonicState,
    modifiers_at: Callable[[float], Modifiers],
    values: Sequence[float],
    **kwargs,
) -> tuple[torch.Tensor, list[bool | None]]:
    """Follow the attractor of ``state`` while a parameter takes ``values`` in order.

    Natural continuation: each solve starts from the previous steady state, as in a
    slow dose ramp. Run it with increasing and with decreasing ``values`` to see
    hysteresis (a pattern memory). Returns ``(V* of shape (len(values), *cells), stable)``.
    """
    vs, stable = [], []
    cur = state
    for val in values:  # parameter values, not cells
        ss = find_steady_state(tissue, cur, modifiers_at(float(val)), **kwargs)
        cur = ss.state
        vs.append(ss.vmem[0])
        stable.append(ss.stable[0])
    return torch.stack(vs), stable


def basin_map(
    tissue: IonicTissue,
    cells: tuple[int, ...],
    region_a: torch.Tensor,
    region_b: torch.Tensor,
    v_values: Sequence[float],
    modifiers: Modifiers | None = None,
    v_tol: float = 1e-3,
    **kwargs,
) -> tuple[torch.Tensor, list[Attractor]]:
    """Attractor label for each pair of start voltages of two regions.

    Start state ``(i, j)``: cells in ``region_a`` at ``v_values[i]``, cells in
    ``region_b`` at ``v_values[j]`` (boolean masks of shape ``cells``; other cells
    at the mean of the two). Returns ``(labels (len, len), attractors)``.
    """
    vv = torch.as_tensor(list(v_values), dtype=DTYPE)
    n = vv.numel()
    va = vv.reshape(n, 1).expand(n, n).reshape(-1)
    vb = vv.reshape(1, n).expand(n, n).reshape(-1)
    shape = (n * n,) + (1,) * len(cells)
    v0 = 0.5 * (va + vb).reshape(shape).expand(n * n, *cells).clone()
    a, b = region_a.bool(), region_b.bool()
    v0 = torch.where(a, va.reshape(shape), v0)
    v0 = torch.where(b, vb.reshape(shape), v0)
    starts = tissue.with_vmem(tissue.initial_state(cells, batch=n * n), v0)
    ss = find_steady_state(tissue, starts, modifiers, **kwargs)
    found: list[Attractor] = []
    labels = torch.empty(n * n, dtype=torch.long)
    for k in range(n * n):
        v = ss.vmem[k]
        for idx, att in enumerate(found):
            if float((att.vmem - v).abs().max()) < v_tol:
                att.count += 1
                labels[k] = idx
                break
        else:
            labels[k] = len(found)
            found.append(Attractor(v, IonicState(ss.state.x[k : k + 1], float("inf")), 1, ss.stable[k],
                                   ss.slowest_time_s[k]))
    return labels.reshape(n, n), found
