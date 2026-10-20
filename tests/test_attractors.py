"""Steady states, stability, implicit gradients and the attractor landscape."""

import torch
from torch.func import jacfwd

from bioelectrics.attractors import (
    basin_map,
    dose_response,
    enumerate_attractors,
    find_steady_state,
    random_vmem_states,
    sparse_jacobian,
    steady_state_implicit,
)
from bioelectrics.constants import DTYPE
from bioelectrics.interventions import ChannelBlocker, ChronicInterventionOptimizer
from bioelectrics.ionic import CAT_DEP_CAM, K_IR_CAM, ChannelExpression, IonicTissue, Modifiers

CHANNELS = [ChannelExpression(K_IR_CAM, 1e-16), ChannelExpression(CAT_DEP_CAM, 1e-16)]


def single_cell():
    return IonicTissue(edges=torch.zeros(2, 0, dtype=torch.long), channels=CHANNELS)


def tumor_tissue(n=6, patch=slice(2, 4)):
    tissue = IonicTissue(channels=CHANNELS)
    v0 = torch.full((1, n, n), -0.078, dtype=DTYPE)
    v0[:, patch, patch] = -0.003
    return tissue, tissue.with_vmem(tissue.initial_state((n, n)), v0)


def test_sparse_jacobian_equals_dense_autograd_on_grid_and_graph():
    tissue, state = tumor_tissue(n=3, patch=slice(1, 2))
    x = state.x
    dense = jacfwd(lambda z: tissue.rhs(z.reshape(x.shape)).reshape(-1))(x.reshape(-1))
    sparse = torch.as_tensor(sparse_jacobian(tissue, x).toarray())
    assert torch.allclose(sparse, dense, rtol=1e-12, atol=1e-12 * float(dense.abs().max()))

    edges = torch.tensor([[0, 1, 2, 0], [1, 2, 3, 3]])  # a ring of 4 cells (graph coloring path)
    ring = IonicTissue(channels=CHANNELS, edges=edges)
    v = torch.tensor([[-0.078, -0.003, -0.05, -0.02]], dtype=DTYPE)
    xr = ring.with_vmem(ring.initial_state((4,)), v).x
    dense = jacfwd(lambda z: ring.rhs(z.reshape(xr.shape)).reshape(-1))(xr.reshape(-1))
    sparse = torch.as_tensor(sparse_jacobian(ring, xr).toarray())
    assert torch.allclose(sparse, dense, rtol=1e-12, atol=1e-12 * float(dense.abs().max()))


def test_single_cell_has_two_attractors_and_a_saddle():
    cell = single_cell()
    starts = cell.with_vmem(cell.initial_state((1,), batch=3), torch.tensor([[-0.078], [-0.003], [-0.037]], dtype=DTYPE))
    ss = find_steady_state(cell, starts)
    v = ss.vmem[:, 0] * 1e3
    assert ss.stable == [True, True, False]
    assert v[0] < -70.0 and v[1] > -20.0 and v[0] < v[2] < v[1]
    assert ss.max_real_eig[2] > 1.0  # saddle: one fast unstable direction
    assert max(ss.residual) < 1e-8  # |dx/dt| in mol/(m^3 s) and 1/s


def test_steady_states_are_fixed_points_of_the_simulation():
    """A stable state stays put under simulation; a perturbed saddle leaves it."""
    cell = single_cell()
    starts = cell.with_vmem(cell.initial_state((1,), batch=3), torch.tensor([[-0.078], [-0.003], [-0.037]], dtype=DTYPE))
    ss = find_steady_state(cell, starts)
    after = cell.simulate(ss.state.__class__(ss.state.x.clone(), 0.0), 1.0)
    dv = (cell.vmem_of(after) - ss.vmem).abs()[:, 0]
    assert float(dv[0]) < 1e-6 and float(dv[1]) < 1e-6

    saddle = cell.with_vmem(ss.state.__class__(ss.state.x[2:3].clone(), 0.0), ss.vmem[2:3] + 2e-3)
    away = cell.simulate(saddle, 1.0)
    assert float((cell.vmem_of(away) - ss.vmem[2:3]).abs()) > 10e-3


def test_tumor_domain_is_a_stable_attractor_of_the_tissue():
    tissue, state = tumor_tissue()
    ss = find_steady_state(tissue, state)
    v = ss.vmem[0] * 1e3
    assert ss.stable == [True]
    assert float(v[2:4, 2:4].max()) > -20.0 and float(v[0, 0]) < -70.0


def test_implicit_gradient_matches_finite_difference():
    tissue, state = tumor_tissue(n=4, patch=slice(1, 3))
    ramp = torch.linspace(0, 1, 16, dtype=DTYPE).reshape(4, 4)
    block = torch.tensor(0.3, dtype=DTYPE, requires_grad=True)
    mods = lambda b=block: Modifiers(channel_scale={"Cat_dep(CAM)": 1 - b * ramp})
    x = steady_state_implicit(tissue, state, mods, [block])
    (tissue.vmem_of(x) ** 2).sum().backward()

    def loss(b: float) -> float:
        ss = find_steady_state(tissue, state, mods(torch.tensor(b, dtype=DTYPE)), stability=False)
        return float((ss.vmem ** 2).sum())

    eps = 1e-4
    fd = (loss(0.3 + eps) - loss(0.3 - eps)) / (2 * eps)
    assert abs(float(block.grad) - fd) <= 1e-4 * abs(fd)


def test_implicit_gradient_reaches_tissue_parameters():
    """Gradients flow to a channel expression level read directly by the rhs."""
    d = torch.tensor(1e-16, dtype=DTYPE, requires_grad=True)
    cell = IonicTissue(edges=torch.zeros(2, 0, dtype=torch.long),
                       channels=[ChannelExpression(K_IR_CAM, d), ChannelExpression(CAT_DEP_CAM, 1e-16)])
    start = cell.with_vmem(cell.initial_state((1,)), torch.full((1, 1), -0.078, dtype=DTYPE))
    v = cell.vmem_of(steady_state_implicit(cell, start, None, [d]))
    v.sum().backward()
    assert float(d.grad) < 0  # more K+ inward rectifier hyperpolarizes the healthy state


def test_dose_ramp_shows_hysteresis():
    """Ramp a blocker of the depolarizing channel up from the tumor state, then down."""
    cell = single_cell()
    tumor = cell.with_vmem(cell.initial_state((1,)), torch.full((1, 1), -0.003, dtype=DTYPE))
    doses = [0.0, 0.2, 0.4, 0.6, 0.8]
    at = lambda b: Modifiers(channel_scale={"Cat_dep(CAM)": torch.tensor(1 - b, dtype=DTYPE)})
    up, _ = dose_response(cell, tumor, at, doses)
    healthy = find_steady_state(cell, cell.with_vmem(cell.initial_state((1,)), up[-1].reshape(1, 1)), at(0.8)).state
    down, _ = dose_response(cell, healthy, at, doses[::-1])
    down = down.flip(0)
    gap = (up - down).abs().max()
    assert float(up[0]) > -0.02 and float(up[-1]) < -0.07  # the ramp switches the cell
    assert float(down[0]) < -0.07  # removing the drug does not restore the tumor state
    assert float(gap) > 0.03


def test_two_coupled_cells_have_mixed_attractors():
    edges = torch.tensor([[0], [1]])
    pair = IonicTissue(channels=CHANNELS, edges=edges)
    found = enumerate_attractors(pair, random_vmem_states(pair, (2,), 24, seed=1))
    patterns = {tuple((a.vmem * 1e3).round().int().tolist()) for a in found}
    assert len(found) >= 3  # both healthy, both depolarized, and at least one mixed pattern
    assert all(a.stable for a in found)
    assert any(abs(p[0] - p[1]) > 30 for p in patterns)

    labels, atts = basin_map(pair, (2,), torch.tensor([True, False]), torch.tensor([False, True]),
                             [-0.08, -0.04, 0.0])
    assert labels.shape == (3, 3) and len(atts) >= 2


def test_chronic_drug_dose_normalizes_tumor_steady_state():
    tissue, state = tumor_tissue()
    target = torch.full((1, 6, 6), -0.0773, dtype=DTYPE)
    opt = ChronicInterventionOptimizer(tissue, state, target, [ChannelBlocker("Cat_dep(CAM)", cost=1.0)])
    res = opt.optimize(n_iters=40, tol_mV=1.0)
    assert res.protocol.chronic
    assert res.protocol.final_error_mV < 1.0
    assert res.protocol.error_reduction > 0.9


def test_fine_dose_ramp_crosses_the_fold_without_unphysical_steps():
    """Regression: near the fold, an accepted step once made [Na+]_in negative."""
    cell = single_cell()
    tumor = cell.with_vmem(cell.initial_state((1,)), torch.full((1, 1), -0.003, dtype=DTYPE))
    at = lambda b: Modifiers(channel_scale={"Cat_dep(CAM)": torch.tensor(1 - b, dtype=DTYPE)})
    doses = [0.0] + torch.linspace(0.70, 0.85, 7).tolist()  # start at 0: the ramp carries the history
    v, stable = dose_response(cell, tumor, at, doses)
    assert float(v[1]) > -0.03 and float(v[-1]) < -0.07
    assert all(stable)
