"""Figure generation smoke tests."""

import matplotlib.pyplot as plt
import torch

from bioelectrics import ChannelProfile, GapJunctionProfile, tumor_prepattern
from bioelectrics.control import OptimizationResult
from bioelectrics.viz import make_reprogramming_figure, plot_iv_curve, plot_vmem_map


def test_iv_curve_and_closed_junctions(tmp_path):
    fig, (a, b) = plt.subplots(1, 2)
    plot_iv_curve(a)
    state, mask = tumor_prepattern(12, 12, patch=4)
    _, n_closed = plot_vmem_map(b, state.v[0], GapJunctionProfile())
    # A 4x4 patch has 16 boundary edges, all closed at a 55 mV transjunctional voltage.
    assert n_closed == 16
    fig.savefig(tmp_path / "a.png")
    plt.close(fig)


def test_reprogramming_figure_written(tmp_path):
    state, _ = tumor_prepattern(8, 8, patch=2)
    result = OptimizationResult(
        baseline_error=0.01,
        final_error=0.001,
        loss_history=[10.0, 1.0, 0.1],
        stimulus=torch.zeros(4, 1, 8, 8, dtype=torch.float64),
        v_final=torch.full((1, 8, 8), -0.07, dtype=torch.float64),
    )
    out = tmp_path / "fig.png"
    fig = make_reprogramming_figure(ChannelProfile(), GapJunctionProfile(), state.v[0],
                                    result, 0.1, 0.5, str(out))
    assert out.exists() and out.stat().st_size > 10_000
    plt.close(fig)
