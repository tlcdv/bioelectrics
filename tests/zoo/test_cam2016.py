"""Reproduce Cervera, Alcaraz & Mafe, Sci. Rep. 6, 20403 (2016)."""

import pytest
import torch

from bioelectrics import find_fixed_points
from bioelectrics.constants import DTYPE
from bioelectrics.tissue import BioelectricState, Tissue
from bioelectrics.zoo import cam2016


def _depolarized_count_after(g_ratio: float, g_out_ratio: float, t_tilde: float) -> tuple[int, int]:
    profile = cam2016.channel_profile(g_out_ratio)
    v_h, v_s, v_d = (fp[0] for fp in find_fixed_points(profile))
    mask = cam2016.cross_region(50)
    v0 = torch.full((1, 50, 50), v_h, dtype=DTYPE)
    v0[:, mask] = v_d
    tissue = Tissue(channels=profile, gap_junctions=cam2016.gap_junctions(g_ratio))
    out = tissue.simulate(BioelectricState(v0), duration=t_tilde * cam2016.TAU, dt=5e-3)
    return int((out.v[0] > v_s).sum()), int(mask.sum())


def test_fig2_single_cell_fixed_points():
    fps = find_fixed_points(cam2016.channel_profile())
    assert [s for _, s in fps] == [True, False, True]
    v = [1e3 * f[0] for f in fps]
    # Paper Fig. 4 colour bar: hyperpolarized ~ -58, unstable ~ -15, depolarized ~ -7 mV.
    assert v[0] == pytest.approx(-58.6, abs=0.5)
    assert v[1] == pytest.approx(-15.5, abs=0.5)
    assert v[2] == pytest.approx(-7.5, abs=0.5)


def test_fig4_normalization_depends_on_coupling():
    counts = {g: _depolarized_count_after(g, 1.0, t_tilde=20.0) for g in (0.05, 0.15, 0.25)}
    n0 = counts[0.05][1]
    n05, n15, n25 = counts[0.05][0], counts[0.15][0], counts[0.25][0]
    # Weak coupling: a large part of the abnormal region remains at t~ = 20.
    assert n05 > 0.25 * n0
    # Stronger coupling normalizes faster; at G/G_in = 0.25 the region is (almost) gone.
    assert n05 > n15 >= n25
    assert n25 < 0.05 * n0


def test_fig5_channel_overexpression_blocks_normalization():
    n_end, n0 = _depolarized_count_after(0.25, 2.5, t_tilde=40.0)
    assert n_end >= n0  # the abnormal region does not shrink
