"""Ion-level physics: validation against analytic results and BETSE formulas."""

import math

import networkx as nx
import pytest
import torch

from bioelectrics.constants import DTYPE, F, R, T, V_T
from bioelectrics.gap_junctions import edges_from_networkx
from bioelectrics.ionic import (
    HCN2,
    KIR2P1,
    ChannelExpression,
    IonicTissue,
    IonProfile,
    Modifiers,
    betse_gj_open_fraction,
    ghk_influx,
    nak_atpase_influx,
)

NO_EDGES = torch.zeros(2, 0, dtype=torch.long)


def single_cell(**kw) -> IonicTissue:
    return IonicTissue(edges=NO_EDGES, **kw)


def test_profiles_are_charge_neutral_and_start_at_zero_voltage():
    for profile in (IonProfile.mammal(), IonProfile.amphibian()):
        assert (profile.z * profile.c_in).sum().abs().item() < 1e-12
        assert (profile.z * profile.c_out).sum().abs().item() < 1e-12
        tissue = single_cell(profile=profile)
        assert abs(tissue.vmem_of(tissue.initial_state((3,))).max().item()) < 1e-12


def test_ghk_flux_limits():
    c_out, c_in = torch.tensor(145.0, dtype=DTYPE), torch.tensor(12.0, dtype=DTYPE)
    p, z = torch.tensor(1e-9, dtype=DTYPE), torch.tensor(1.0, dtype=DTYPE)
    # V -> 0: Fick diffusion.
    j0 = ghk_influx(c_out, c_in, p, z, torch.tensor(0.0, dtype=DTYPE), T)
    assert j0.item() == pytest.approx(p.item() * (145.0 - 12.0), rel=1e-12)
    # Zero flux at the Nernst potential.
    e_na = V_T * math.log(145.0 / 12.0)
    assert abs(ghk_influx(c_out, c_in, p, z, torch.tensor(e_na, dtype=DTYPE), T).item()) < 1e-18
    # Finite gradient at V = 0.
    v = torch.tensor(0.0, dtype=DTYPE, requires_grad=True)
    ghk_influx(c_out, c_in, p, z, v, T).backward()
    assert torch.isfinite(v.grad)


def test_nernst_potential_with_only_potassium():
    d = torch.zeros(6, dtype=DTYPE)
    tissue = single_cell(membrane_d=d.index_fill(0, torch.tensor([1]), 1e-18), alpha_nak=0.0, alpha_ca=0.0)
    # GHK slope conductance near E_K is small (c_out = 5 mM): tau ~ 1.1 s.
    out = tissue.simulate(tissue.initial_state((1,)), 16.0, dt=5e-3)
    c = out.x[0, :, 0]
    e_k = (R * T / F) * math.log(tissue.profile.c_out[1].item() / c[1].item())
    assert tissue.vmem_of(out).item() == pytest.approx(e_k, abs=2e-4)


def test_ghk_voltage_without_pumps():
    """Monovalent ions only: V settles at the GHK voltage equation value."""
    d = torch.tensor([2e-18, 1e-18, 1e-18, 0.0, 1e-18, 0.0], dtype=DTYPE)  # Na K Cl Ca M P
    tissue = single_cell(membrane_d=d, alpha_nak=0.0, alpha_ca=0.0)
    out = tissue.simulate(tissue.initial_state((1,)), 16.0, dt=5e-3)
    ci, co = out.x[0, :, 0], tissue.profile.c_out
    num = d[0] * co[0] + d[1] * co[1] + d[2] * ci[2] + d[4] * ci[4]
    den = d[0] * ci[0] + d[1] * ci[1] + d[2] * co[2] + d[4] * co[4]
    v_ghk = (R * T / F) * math.log((num / den).item())
    assert tissue.vmem_of(out).item() == pytest.approx(v_ghk, abs=2e-4)


def test_nak_pump_direction_and_stoichiometry():
    c = lambda x: torch.tensor(x, dtype=DTYPE)
    j_na, j_k = nak_atpase_influx(c(12.0), c(145.0), c(139.0), c(5.0), c(-0.05), c(1e-7), T)
    assert j_na.item() < 0 < j_k.item()
    assert j_k.item() == pytest.approx(-2.0 / 3.0 * j_na.item())


def test_pump_hyperpolarizes_the_cell():
    with_pump = single_cell()
    no_pump = single_cell(alpha_nak=0.0)
    v1 = with_pump.vmem_of(with_pump.simulate(with_pump.initial_state((1,)), 1.0, dt=2e-3)).item()
    v0 = no_pump.vmem_of(no_pump.simulate(no_pump.initial_state((1,)), 1.0, dt=2e-3)).item()
    assert v1 < v0 - 0.01


def test_betse_gap_junction_gating_formula():
    v = torch.linspace(-0.08, 0.08, 17, dtype=DTYPE)
    vm = v * 1e3
    expected = 0.10 + (-1 / (1 + torch.exp((vm + 15) / 15)) + 1 / (1 + torch.exp((vm - 15) / 15)))
    assert torch.allclose(betse_gj_open_fraction(v), expected, atol=1e-14)


def test_gap_junctions_conserve_mass_and_equalize_voltage():
    zero_d = torch.zeros(6, dtype=DTYPE)
    tissue = IonicTissue(membrane_d=zero_d, alpha_nak=0.0, alpha_ca=0.0, gj_gating=None)
    state = tissue.initial_state((4, 4))
    x = state.x.clone()
    x[0, 1, 1, 1] += 1e-3  # 1 uM extra K+ in one cell: about +16 mV
    state.x = x
    v0 = tissue.vmem_of(state)
    assert v0[0, 1, 1].item() > 0.01
    total0 = x[0, : tissue.n_ions].sum(dim=(-2, -1))
    # Electrical coupling is fast (~32 /s per junction); salt diffusion is slow (tau ~ 45 min).
    out = tissue.simulate(state, 2.0)
    c = out.x[0, : tissue.n_ions]
    assert torch.allclose(c.sum(dim=(-2, -1)), total0, rtol=1e-12)
    assert tissue.vmem_of(out).std().item() < 1e-3 * v0.std().item()


def test_grid_and_graph_topologies_agree():
    h, w = 3, 4
    grid = IonicTissue()
    graph = IonicTissue(edges=edges_from_networkx(nx.convert_node_labels_to_integers(nx.grid_2d_graph(h, w))))
    sg = grid.initial_state((h, w))
    torch.manual_seed(0)
    sg.x[:, 0] += torch.rand(1, h, w, dtype=DTYPE)  # perturb Na+
    sg.x[:, 4] += sg.x[:, 0] - grid.profile.c_in[0]  # keep neutrality via M-
    sn = type(sg)(sg.x.reshape(1, -1, h * w).clone())
    xg = grid.rhs(sg.x)
    xn = graph.rhs(sn.x).reshape(1, -1, h, w)
    assert torch.allclose(xg, xn, rtol=1e-10, atol=1e-18)


def test_channel_expression_signs():
    base = single_cell()
    kir = single_cell(channels=[ChannelExpression(KIR2P1, 3e-16)])
    v_base = base.vmem_of(base.simulate(base.initial_state((1,)), 3.0)).item()
    v_kir = kir.vmem_of(kir.simulate(kir.initial_state((1,)), 3.0)).item()
    # Kir2.1 misexpression hyperpolarizes towards E_K (about -89 mV).
    assert v_kir < v_base - 0.04
    assert v_kir > -0.090
    # HCN2 opens when hyperpolarized and conducts Na+: it depolarizes a Kir2.1 cell.
    both = single_cell(channels=[ChannelExpression(KIR2P1, 3e-16), ChannelExpression(HCN2, 2e-16)])
    v_both = both.vmem_of(both.simulate(both.initial_state((1,)), 3.0)).item()
    assert v_both > v_kir + 0.002


def test_gradients_reach_ionic_hardware():
    max_d = torch.tensor(2e-17, dtype=DTYPE, requires_grad=True)
    alpha = torch.tensor(1e-7, dtype=DTYPE, requires_grad=True)
    gj = torch.tensor(5e-8, dtype=DTYPE, requires_grad=True)
    tissue = IonicTissue(channels=[ChannelExpression(KIR2P1, max_d)], alpha_nak=alpha, gj_fraction=gj)
    state = tissue.initial_state((2, 2))
    state.x[0, 0, 0, 0] += 1.0
    state.x[0, 4, 0, 0] += 1.0
    out = tissue.simulate(state, 0.05, dt=1e-3)
    v = tissue.vmem_of(out)
    (v[0, 0, 0] + v.mean()).backward()
    for g in (max_d.grad, alpha.grad, gj.grad):
        assert g is not None and torch.isfinite(g) and g.abs().item() > 0


def test_modifiers_block_channels_and_pumps():
    tissue = single_cell(channels=[ChannelExpression(KIR2P1, 3e-16)])
    s0 = tissue.initial_state((1,))
    v_ctrl = tissue.vmem_of(tissue.simulate(s0, 1.0, dt=1e-3)).item()
    v_block = tissue.vmem_of(
        tissue.simulate(s0, 1.0, dt=1e-3, modifiers=Modifiers(channel_scale={"Kir2.1": 0.0}))
    ).item()
    v_ouabain = tissue.vmem_of(tissue.simulate(s0, 1.0, dt=1e-3, modifiers=Modifiers(nak_scale=0.0))).item()
    assert v_block > v_ctrl + 0.005
    assert v_ouabain > v_ctrl
