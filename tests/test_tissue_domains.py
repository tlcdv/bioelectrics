"""Phase 2: 2D multicellular tissue with voltage-gated gap junctions."""

import networkx as nx
import pytest
import torch

from bioelectrics.channels import ChannelProfile
from bioelectrics.constants import DTYPE, mV_to_V
from bioelectrics.gap_junctions import (
    GapJunctionProfile,
    edges_from_networkx,
    gj_conductance,
    graph_coupling_current,
    grid_conductances,
    grid_coupling_current,
)
from bioelectrics.tissue import BioelectricState, Tissue, tumor_prepattern

PASSIVE = ChannelProfile(g_pol0=0.0, g_dep0=0.0, g_leak=0.0)


def test_gj_conductance_is_bell_shaped():
    gj = GapJunctionProfile()
    v = torch.tensor([-0.06, -0.02, 0.0, 0.02, 0.06], dtype=DTYPE)
    g = gj_conductance(v, gj)
    assert torch.allclose(g, g.flip(0))  # symmetric in V_ij
    assert g[2].item() == pytest.approx(gj.g_max.item(), rel=0.01)
    assert g[3].item() == pytest.approx(0.5 * (gj.g_max + gj.g_min).item(), rel=1e-6)
    assert g[4].item() == pytest.approx(gj.g_min.item(), rel=0.05)


def test_grid_coupling_matches_graph_coupling():
    h, w = 5, 7
    v = mV_to_V(-70.0 + 40.0 * torch.rand(3, h, w, dtype=DTYPE))
    gj = GapJunctionProfile()
    edges = edges_from_networkx(nx.convert_node_labels_to_integers(nx.grid_2d_graph(h, w)))
    i_grid = grid_coupling_current(v, gj)
    i_graph = graph_coupling_current(v.reshape(3, -1), edges, gj).reshape(3, h, w)
    assert torch.allclose(i_grid, i_graph, atol=1e-24)
    # Gap junction currents conserve charge.
    assert i_grid.sum(dim=(-2, -1)).abs().max().item() < 1e-20


def test_passive_equalization_of_small_perturbations():
    torch.manual_seed(0)
    v0 = mV_to_V(-50.0 + 2.0 * torch.randn(2, 16, 16, dtype=DTYPE))
    tissue = Tissue(channels=PASSIVE)
    # Slowest diffusion mode on 16x16: rate (G_max/C_m)(2 - 2cos(pi/16)) ~ 3.8 /s.
    out = tissue.simulate(BioelectricState(v0), 3.0)
    # Small |V_ij| keeps junctions open: the tissue becomes isopotential.
    assert out.v.std(dim=(-2, -1)).max().item() < 1e-3 * v0.std(dim=(-2, -1)).min().item()
    assert torch.allclose(out.v.mean(dim=(-2, -1)), v0.mean(dim=(-2, -1)), atol=1e-9)


def test_domain_wall_isolation_32x32():
    state, mask = tumor_prepattern(32, 32, patch=10, v_healthy_mV=-70.0, v_tumor_mV=-15.0)
    tissue = Tissue()
    out = tissue.simulate(state, 2.0)
    v = out.v_mV[0]
    # Both domains coexist stably.
    assert v[mask].min().item() > -25.0 and v[mask].max().item() < -15.0
    assert v[~mask].max().item() < -65.0
    # Boundary junctions are closed, interior junctions are open.
    gx, gy = grid_conductances(out.v, tissue.gap_junctions)
    gx, gy = gx[0], gy[0]
    cross_x = mask[:, 1:] != mask[:, :-1]
    cross_y = mask[1:, :] != mask[:-1, :]
    g_min, g_max = tissue.gap_junctions.g_min.item(), tissue.gap_junctions.g_max.item()
    assert gx[cross_x].max().item() < 1.1 * g_min
    assert gy[cross_y].max().item() < 1.1 * g_min
    assert gx[~cross_x].min().item() > 0.9 * g_max
    assert gy[~cross_y].min().item() > 0.9 * g_max
    # The state is stationary: one more second does not change it.
    later = tissue.simulate(out, 1.0)
    assert (later.v - out.v).abs().max().item() < mV_to_V(0.1)


def test_ungated_junctions_cannot_hold_the_domain():
    """Control: without voltage gating the polarized surround erases the tumor."""
    state, mask = tumor_prepattern(32, 32, patch=10)
    tissue = Tissue(gap_junctions=GapJunctionProfile(g_min=1e-9, g_max=1e-9))
    v = tissue.simulate(state, 2.0).v_mV[0]
    assert v[mask].max().item() < -65.0


def test_batched_tissue_and_gradients():
    state, _ = tumor_prepattern(8, 8, patch=2, batch=3)
    tissue = Tissue(
        channels=ChannelProfile().requires_grad_("g_pol0", "g_dep0"),
        gap_junctions=GapJunctionProfile().requires_grad_("g_max"),
    )
    i_ext = torch.zeros(3, 8, 8, dtype=DTYPE, requires_grad=True)
    out = tissue.simulate(state, 0.02, stimulus=i_ext)
    assert out.v.shape == (3, 8, 8)
    out.v.mean().backward()
    for g in (tissue.channels.g_pol0.grad, tissue.channels.g_dep0.grad,
              tissue.gap_junctions.g_max.grad, i_ext.grad):
        assert g is not None and torch.isfinite(g).all() and g.abs().sum().item() > 0
