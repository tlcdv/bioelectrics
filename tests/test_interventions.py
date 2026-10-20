"""Interventions and their inverse design on the ion-level tissue."""

import pytest
import torch

from bioelectrics.constants import DTYPE
from bioelectrics.interventions import (
    ChannelBlocker,
    ChannelMisexpression,
    GapJunctionBlocker,
    InterventionOptimizer,
    Optogenetic,
    PumpBlocker,
)
from bioelectrics.ionic import CAT_DEP_CAM, K_IR_CAM, ChannelExpression, IonicTissue, Modifiers, leak

V_HEALTHY, V_TUMOR = -0.078, -0.003


def bistable_tissue(extra=()):
    """Ion-level tissue whose cells have two attractors (about -78 and -4 mV)."""
    return IonicTissue(
        channels=[ChannelExpression(K_IR_CAM, 1e-16), ChannelExpression(CAT_DEP_CAM, 1e-16), *extra]
    )


def tumor_state(tissue, n=6, lo=2, hi=4, modifiers=None):
    mask = torch.zeros(n, n, dtype=torch.bool)
    mask[lo:hi, lo:hi] = True
    v0 = torch.full((1, n, n), V_HEALTHY, dtype=DTYPE)
    v0[:, mask] = V_TUMOR
    state = tissue.simulate(tissue.with_vmem(tissue.initial_state((n, n)), v0), 0.5, modifiers=modifiers)
    return state, mask


def test_intervention_values_respect_bounds_and_apply():
    shape = (1, 3, 3)
    blk = ChannelBlocker("Cat_dep(CAM)", max_block=0.8, init=0.5)
    blk.build(shape)
    with torch.no_grad():
        blk.u.fill_(50.0)
    mods = Modifiers()
    blk.apply(mods)
    assert mods.channel_scale["Cat_dep(CAM)"].item() == pytest.approx(0.2)

    dose = ChannelBlocker("Cat_dep(CAM)", ic50=2.0, hill=1.0, init=2.0)
    dose.build(shape)
    assert dose.blocked_fraction().item() == pytest.approx(0.5, rel=1e-6)

    region = torch.zeros(3, 3, dtype=DTYPE)
    region[1, 1] = 1.0
    mis = ChannelMisexpression("K_ir(CAM)", region=region, max_d=1e-15, init=5e-16)
    mis.build(shape)
    pump, gj = PumpBlocker(init=0.25), GapJunctionBlocker(init=0.5)
    pump.build(shape)
    gj.build(shape)
    mods = Modifiers()
    for iv in (mis, pump, gj):
        iv.apply(mods)
    added = mods.channel_add["K_ir(CAM)"]
    assert added[1, 1].item() == pytest.approx(5e-16, rel=1e-6) and added.sum().item() == pytest.approx(5e-16, rel=1e-6)
    assert mods.nak_scale.item() == pytest.approx(0.75, rel=1e-6)
    assert mods.gj_scale.item() == pytest.approx(0.5, rel=1e-6)

    opto = Optogenetic("ClLeak", init=0.3)
    opto.build((3, 3))
    mods = Modifiers()
    opto.apply(mods)
    assert mods.channel_gate["ClLeak"].shape == (3, 3)


def test_tumor_is_a_stable_attractor_without_intervention():
    tissue = bistable_tissue()
    state, mask = tumor_state(tissue)
    later = tissue.simulate(state, 1.0)
    v = tissue.vmem_of(later)[0]
    assert v[mask].min().item() > -0.010
    assert v[~mask].max().item() < -0.070


def test_optimized_blocker_normalizes_tumor_permanently():
    tissue = bistable_tissue()
    state, mask = tumor_state(tissue)
    target = torch.full_like(tissue.vmem_of(state), V_HEALTHY)
    opt = InterventionOptimizer(
        tissue, state, target, [ChannelBlocker("Cat_dep(CAM)", cost=1.0)], t_stim=0.1, t_relax=0.5
    )
    result = opt.optimize(n_iters=30, tol_mV=1.0)
    p = result.protocol
    assert p.error_reduction >= 0.85
    assert 0.0 < p.steps[0]["value"] < 1.0
    # The drug is washed out at T_stim; one more second without it keeps the healthy state.
    with torch.no_grad():
        mid = tissue.simulate(state, 0.1, modifiers=opt.modifiers())
        end = tissue.simulate(mid, 1.4)
    v = tissue.vmem_of(end)[0]
    assert v.max().item() < -0.070
    assert "channel blocker" in p.to_markdown()


def test_optogenetic_light_pattern_targets_the_tumor():
    opsin = ChannelExpression(leak("Cl"), 3e-16)  # anion channelrhodopsin, E_Cl ~ -90 mV
    tissue = bistable_tissue(extra=(opsin,))
    off = Modifiers(channel_gate={"ClLeak": torch.zeros(6, 6, dtype=DTYPE)})
    state, mask = tumor_state(tissue, modifiers=off)  # dark: the tumor persists
    assert tissue.vmem_of(state)[0][mask].min().item() > -0.010
    target = torch.full_like(tissue.vmem_of(state), V_HEALTHY)
    light = Optogenetic("ClLeak", init=0.02, spatial=True, cost=20.0)
    # The opsin is dark (gate = 0) whenever the light is off, including after T_stim.
    opt = InterventionOptimizer(tissue, state, target, [light], t_stim=0.1, t_relax=0.5, background=off)
    result = opt.optimize(n_iters=40, tol_mV=1.0)
    assert result.protocol.error_reduction >= 0.85
    field = light.field().detach()
    assert field[mask].mean().item() > 3.0 * field[~mask].mean().item()

