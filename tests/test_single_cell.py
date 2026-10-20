"""Phase 1: differentiable single-cell bistability."""

import math

import pytest
import torch

from bioelectrics import ChannelProfile, find_fixed_points, ghk_current, integrate, ionic_current
from bioelectrics.channels import membrane_rhs
from bioelectrics.constants import DTYPE, F, R, T, V_T, V_to_mV, mV_to_V

DT = 5e-4  # [s]


def test_constants_si():
    assert F == pytest.approx(96485.332)
    assert R == pytest.approx(8.31446)
    assert T == pytest.approx(310.15)
    assert V_T == pytest.approx(0.0267, abs=1e-4)
    assert mV_to_V(-70.0) == pytest.approx(-0.070)
    assert V_to_mV(0.02) == pytest.approx(20.0)


def test_bistable_fixed_points():
    fps = find_fixed_points(ChannelProfile())
    assert len(fps) == 3
    (v_pol, s_pol), (v_sad, s_sad), (v_dep, s_dep) = fps
    assert s_pol and not s_sad and s_dep
    assert V_to_mV(v_pol) == pytest.approx(-70.0, abs=3.0)
    assert V_to_mV(v_dep) == pytest.approx(-20.0, abs=3.0)
    assert v_pol < v_sad < v_dep


def test_stability_from_autograd_slope():
    """dI_ion/dV > 0 at stable points and < 0 at the saddle (autograd check)."""
    p = ChannelProfile()
    for v_star, stable in find_fixed_points(p):
        v = torch.tensor(v_star, dtype=DTYPE, requires_grad=True)
        (slope,) = torch.autograd.grad(ionic_current(v, p), v)
        assert (slope.item() > 0) == stable
        assert abs(ionic_current(v, p).item()) < 1e-15


def _run_pulse(amplitude_a: float, t_pulse: float = 0.05, t_end: float = 0.6):
    p = ChannelProfile()
    v0 = torch.full((1,), mV_to_V(-70.0), dtype=DTYPE)

    def rhs(t, v):
        i_ext = amplitude_a if t < t_pulse else 0.0
        return membrane_rhs(v, p, i_ext)

    v_end, _, traj = integrate(rhs, v0, t_end, DT, record_every=10)
    return v_end, traj


def test_pulse_switches_cell_permanently():
    v_end, traj = _run_pulse(30e-12)
    v_dep = find_fixed_points(ChannelProfile())[2][0]
    assert V_to_mV(v_end.item()) == pytest.approx(V_to_mV(v_dep), abs=0.5)
    # After the pulse (t > 50 ms) the stimulus is zero; the last 300 ms stay depolarized.
    late = V_to_mV(traj[-60:])
    assert torch.all(late > -25.0)


def test_subthreshold_pulse_returns_to_polarized_state():
    v_end, _ = _run_pulse(2e-12)
    assert V_to_mV(v_end.item()) == pytest.approx(-70.0, abs=1.0)


def test_rk4_matches_euler_with_small_step():
    p = ChannelProfile()
    v0 = torch.tensor([mV_to_V(-60.0)], dtype=DTYPE)
    rhs = lambda t, v: membrane_rhs(v, p, 5e-12)
    v_rk4, _, _ = integrate(rhs, v0, 0.05, DT, method="rk4")
    v_eul, _, _ = integrate(rhs, v0, 0.05, DT / 20, method="euler")
    assert v_rk4.item() == pytest.approx(v_eul.item(), abs=1e-5)


def test_gradient_flow_to_hardware_and_stimulus():
    p = ChannelProfile().requires_grad_("g_pol0", "g_dep0")
    i_ext = torch.tensor(10e-12, dtype=DTYPE, requires_grad=True)
    v0 = torch.full((4,), mV_to_V(-60.0), dtype=DTYPE)
    rhs = lambda t, v: membrane_rhs(v, p, i_ext)
    v_end, _, _ = integrate(rhs, v0, 0.03, DT)
    v_end.sum().backward()
    for g in (p.g_pol0.grad, p.g_dep0.grad, i_ext.grad):
        assert g is not None
        assert torch.isfinite(g).all()
        assert g.abs().item() > 0.0
    # Physics signs: more pol conductance -> more negative V; more stimulus -> higher V.
    assert p.g_pol0.grad.item() < 0
    assert p.g_dep0.grad.item() > 0
    assert i_ext.grad.item() > 0


def test_ghk_current_reversal_and_zero_voltage_limit():
    c_in = torch.tensor(140.0, dtype=DTYPE)
    c_out = torch.tensor(5.0, dtype=DTYPE)
    perm = torch.tensor(1e-8, dtype=DTYPE)
    e_k = V_T * math.log(5.0 / 140.0)
    i_rev = ghk_current(torch.tensor(e_k, dtype=DTYPE), c_in, c_out, perm, 1)
    assert abs(i_rev.item()) < 1e-12
    i0 = ghk_current(torch.tensor(0.0, dtype=DTYPE), c_in, c_out, perm, 1)
    i_eps = ghk_current(torch.tensor(1e-9, dtype=DTYPE), c_in, c_out, perm, 1)
    assert i0.item() == pytest.approx(perm.item() * F * (140.0 - 5.0), rel=1e-6)
    assert i0.item() == pytest.approx(i_eps.item(), rel=1e-4)
