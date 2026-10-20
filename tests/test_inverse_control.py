"""Phase 3: inverse bioelectric pattern control (differentiable optimizer)."""

import pytest
import torch

from bioelectrics import ChannelProfile, Tissue, find_fixed_points, tumor_prepattern
from bioelectrics.constants import V_to_mV
from bioelectrics.control import PatternOptimizer
from bioelectrics.gap_junctions import grid_conductances

V_SADDLE = find_fixed_points(ChannelProfile())[1][0]
V_HEALTHY = find_fixed_points(ChannelProfile())[0][0]


@pytest.fixture(scope="module")
def reprogramming():
    state, mask = tumor_prepattern(20, 20, patch=6, v_healthy_mV=-70.0, v_tumor_mV=-15.0)
    tissue = Tissue()
    opt = PatternOptimizer(
        tissue,
        state.v,
        torch.full_like(state.v, V_HEALTHY),
        t_stim=0.1,
        t_relax=0.5,
        dt=5e-4,
        n_time_bins=4,
    )
    result = opt.optimize(n_iters=30, tol=1e-5)
    return tissue, opt, result, mask


def test_baseline_tumor_persists_without_stimulus(reprogramming):
    _, _, result, _ = reprogramming
    # Without intervention the isolated tumor stays depolarized: large error.
    assert V_to_mV(result.baseline_error) > 10.0


def test_pattern_error_reduced_by_85_percent(reprogramming):
    _, _, result, _ = reprogramming
    assert result.error_reduction >= 0.85
    assert len(result.loss_history) >= 1


def test_tumor_erased_and_locked_in_healthy_attractor(reprogramming):
    tissue, opt, result, mask = reprogramming
    v_final = result.v_final[0]
    # At T_relax (stimulus off since T_stim) every tumor cell is below the saddle.
    assert v_final[mask].max().item() < V_SADDLE
    assert V_to_mV(v_final).max().item() < -65.0
    # Zero-stimulus continuation: the tissue holds the healthy attractor (no clamp).
    later = tissue.simulate(opt.final_state(), duration=1.0)
    assert V_to_mV(later.v).max().item() < -69.0
    assert (later.v - result.v_final).abs().max().item() < 1e-3
    # Re-polarization re-opens the gap junctions: the self boundary is dissolved.
    gx, gy = grid_conductances(later.v, tissue.gap_junctions)
    g_max = tissue.gap_junctions.g_max.item()
    assert gx.min().item() > 0.95 * g_max and gy.min().item() > 0.95 * g_max


def test_stimulus_is_finite_and_brief(reprogramming):
    _, opt, result, _ = reprogramming
    assert result.stimulus.shape == (4, 1, 20, 20)
    assert torch.isfinite(result.stimulus).all()
    assert result.stimulus.abs().max().item() > 0.0
    # The stimulus only acts in [0, T_stim]; the relaxation phase has no stimulus term.
    assert opt.t_stim < opt.t_relax


def test_gradient_reaches_stimulus_parameters():
    state, _ = tumor_prepattern(6, 6, patch=2)
    opt = PatternOptimizer(Tissue(), state.v, torch.full_like(state.v, V_HEALTHY),
                           t_stim=0.02, t_relax=0.05)
    v_stim_end, v_final = opt.rollout()
    opt.loss(v_stim_end, v_final).backward()
    assert opt.u.grad is not None
    assert torch.isfinite(opt.u.grad).all()
    assert opt.u.grad.abs().sum().item() > 0.0


def test_invalid_time_split_rejected():
    state, _ = tumor_prepattern(4, 4, patch=2)
    with pytest.raises(ValueError):
        PatternOptimizer(Tissue(), state.v, state.v, t_stim=0.5, t_relax=0.1)
