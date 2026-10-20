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


def test_ionic_plots(tmp_path):
    from bioelectrics.ionic import CAT_DEP_CAM, K_IR_CAM, ChannelExpression, IonicTissue
    from bioelectrics.viz import ionic_iv_curve, plot_field_map, plot_ionic_iv, plot_ionic_vmem_map

    cell = IonicTissue(edges=torch.zeros(2, 0, dtype=torch.long),
                       channels=[ChannelExpression(K_IR_CAM, 1e-16), ChannelExpression(CAT_DEP_CAM, 1e-16)])
    v, i = ionic_iv_curve(cell)
    crossings = int((torch.sign(i[:-1]) != torch.sign(i[1:])).sum())
    assert crossings == 3  # stable, saddle, stable
    v_map = torch.full((6, 6), -0.078, dtype=torch.float64)
    v_map[2:4, 2:4] = -0.003
    fig, (a, b, c) = plt.subplots(1, 3)
    plot_ionic_iv(a, cell)
    _, n_closed = plot_ionic_vmem_map(b, v_map)
    assert n_closed == 8  # boundary of a 2x2 domain
    plot_field_map(c, torch.rand(6, 6))
    fig.savefig(tmp_path / "ionic.png")
    plt.close(fig)


def test_theme_switch_restores_and_colors_axes(tmp_path):
    import matplotlib

    from bioelectrics.viz import CONSOLE, IVORY, theme, theme_context

    assert theme() is IVORY
    with theme_context("console") as t:
        assert t is CONSOLE and theme() is CONSOLE
        fig, ax = plt.subplots()
        plot_iv_curve(ax)
        assert matplotlib.colors.to_hex(ax.get_facecolor()) == CONSOLE.panel
        fig.savefig(tmp_path / "dark.png")
        plt.close(fig)
    assert theme() is IVORY
    assert matplotlib.rcParams["figure.facecolor"] == IVORY.surface
