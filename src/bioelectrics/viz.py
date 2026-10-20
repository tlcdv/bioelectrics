"""Publication-grade Matplotlib figures for bioelectric states.

Colors follow a fixed palette: categorical slots for line series, and a
blue <-> gray <-> red diverging map for ``V_mem`` (centered on the bistability
saddle) and for stimulus currents (centered on 0 A).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

from .channels import ChannelProfile, find_fixed_points, i_dep, i_pol, ionic_current  # noqa: E402
from .constants import DTYPE  # noqa: E402
from .gap_junctions import GapJunctionProfile, grid_conductances  # noqa: E402

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]

DIVERGING = LinearSegmentedColormap.from_list(
    "bioelectric_diverging", ["#184f95", "#3987e5", "#f0efec", "#e34948", "#a32a2a"]
)


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=8)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)
    ax.title.set_color(INK)


def _panel_label(ax, text: str) -> None:
    ax.text(-0.02, 1.04, text, transform=ax.transAxes, fontsize=12, fontweight="bold",
            color=INK, ha="right", va="bottom")


def _np(x) -> np.ndarray:
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def plot_iv_curve(ax, profile: ChannelProfile | None = None, v_range_mV=(-90.0, 5.0)) -> None:
    """Single-cell I-V curve with I_pol, I_dep and their sum; marks the fixed points."""
    profile = profile or ChannelProfile()
    v = torch.linspace(v_range_mV[0], v_range_mV[1], 600, dtype=DTYPE) * 1e-3
    with torch.no_grad():
        curves = [
            ("I_pol", i_pol(v, profile), SERIES[0], 1.5),
            ("I_dep", i_dep(v, profile), SERIES[1], 1.5),
            ("I_ion (total)", ionic_current(v, profile), SERIES[2], 2.2),
        ]
    v_mV = _np(v) * 1e3
    ax.axhline(0.0, color=BASELINE, lw=0.8, zorder=1)
    for name, i, color, lw in curves:
        ax.plot(v_mV, _np(i) * 1e12, color=color, lw=lw, label=name, zorder=2)
    for v_star, stable in find_fixed_points(profile):
        ax.plot(v_star * 1e3, 0.0, "o", ms=8, zorder=4,
                mfc=INK if stable else SURFACE, mec=INK, mew=1.5)
        ax.annotate(f"{v_star * 1e3:.0f} mV\n{'stable' if stable else 'saddle'}",
                    (v_star * 1e3, 0.0), textcoords="offset points",
                    xytext=(0, 10 if stable else -26), ha="center", fontsize=7, color=INK_2)
    ax.set_xlabel("V_mem (mV)")
    ax.set_ylabel("Outward current (pA)")
    ax.set_ylim(-60, 60)
    ax.grid(True, color=GRID, lw=0.5)
    ax.legend(frameon=False, fontsize=7, loc="upper left", labelcolor=INK_2)
    _style(ax)


def _closed_junction_segments(v: torch.Tensor, gj: GapJunctionProfile, threshold: float):
    """Line segments on cell borders where ``G_ij < threshold`` (grid coordinates)."""
    gx, gy = grid_conductances(v, gj)
    segs = []
    rows, cols = np.nonzero(_np(gx) < threshold)
    for r, c in zip(rows, cols):  # plotting only, not ODE evaluation
        segs.append([(c + 0.5, r - 0.5), (c + 0.5, r + 0.5)])
    rows, cols = np.nonzero(_np(gy) < threshold)
    for r, c in zip(rows, cols):
        segs.append([(c - 0.5, r + 0.5), (c + 0.5, r + 0.5)])
    return segs


def plot_vmem_map(
    ax,
    v: torch.Tensor,
    gap_junctions: GapJunctionProfile | None = None,
    v_center_mV: float = -45.0,
    v_lim_mV=(-75.0, -10.0),
    show_closed_junctions: bool = True,
):
    """2D ``V_mem`` map (mV). Optionally draws closed gap junctions as dark lines.

    ``v`` has shape ``(H, W)`` [V].
    """
    gap_junctions = gap_junctions or GapJunctionProfile()
    norm = TwoSlopeNorm(vcenter=v_center_mV, vmin=v_lim_mV[0], vmax=v_lim_mV[1])
    im = ax.imshow(_np(v) * 1e3, cmap=DIVERGING, norm=norm, interpolation="nearest")
    n_closed = 0
    if show_closed_junctions:
        thr = 0.5 * float(gap_junctions.g_max + gap_junctions.g_min)
        segs = _closed_junction_segments(v, gap_junctions, thr)
        n_closed = len(segs)
        if segs:
            ax.add_collection(LineCollection(segs, colors=INK, linewidths=1.6))
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(BASELINE)
    return im, n_closed


def plot_loss(ax, loss_history) -> None:
    """Optimization loss per Adam iteration (log scale)."""
    it = np.arange(1, len(loss_history) + 1)
    ax.semilogy(it, loss_history, color=SERIES[0], lw=2, marker="o", ms=5)
    ax.set_xlabel("Adam iteration")
    ax.set_ylabel("Loss")
    ax.grid(True, color=GRID, lw=0.5, which="major")
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    _style(ax)


def plot_stimulus_map(ax, stimulus: torch.Tensor):
    """Time-averaged stimulus ``mean_t I_ext(t, x, y)`` in pA. ``stimulus``: ``(K, H, W)``."""
    s = _np(stimulus).mean(axis=0) * 1e12
    lim = max(float(np.abs(s).max()), 1e-3)
    im = ax.imshow(s, cmap=DIVERGING, norm=TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim),
                   interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(BASELINE)
    return im


def make_reprogramming_figure(
    profile: ChannelProfile,
    gap_junctions: GapJunctionProfile,
    v_initial: torch.Tensor,
    result,
    t_stim: float,
    t_relax: float,
    path: str | None = None,
):
    """4-panel figure: (A) I-V curve, (B) isolated tumor, (C) loss + stimulus, (D) result.

    ``v_initial`` is the ``(H, W)`` tissue state before intervention (with the
    tumor domain settled in its attractor). ``result`` is an ``OptimizationResult``.
    """
    fig = plt.figure(figsize=(12, 9), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1], hspace=0.35, wspace=0.35)

    ax_a = fig.add_subplot(gs[0, 0:2])
    plot_iv_curve(ax_a, profile)
    ax_a.set_title("Single-cell bistability (CAM model)", fontsize=10, loc="left")
    _panel_label(ax_a, "A")

    ax_b = fig.add_subplot(gs[0, 2])
    im_b, n_closed = plot_vmem_map(ax_b, v_initial, gap_junctions)
    ax_b.set_title(f"Initial tissue: tumor isolated\n{n_closed} closed gap junctions",
                   fontsize=10, loc="left")
    _panel_label(ax_b, "B")
    cb = fig.colorbar(im_b, ax=ax_b, fraction=0.046, pad=0.04)
    cb.set_label("V_mem (mV)", color=INK_2, fontsize=8)
    cb.ax.tick_params(labelsize=7, colors=MUTED, labelcolor=INK_2)

    ax_c1 = fig.add_subplot(gs[1, 0])
    plot_loss(ax_c1, result.loss_history)
    ax_c1.set_title("Inverse control: BPTT loss", fontsize=10, loc="left")
    _panel_label(ax_c1, "C")

    ax_c2 = fig.add_subplot(gs[1, 1])
    stim = result.stimulus[:, 0] if result.stimulus.dim() == 4 else result.stimulus
    im_c = plot_stimulus_map(ax_c2, stim)
    ax_c2.set_title(f"Optimized stimulus, mean over\n[0, {t_stim * 1e3:.0f} ms]",
                    fontsize=10, loc="left")
    cb = fig.colorbar(im_c, ax=ax_c2, fraction=0.046, pad=0.04)
    cb.set_label("I_ext (pA)", color=INK_2, fontsize=8)
    cb.ax.tick_params(labelsize=7, colors=MUTED, labelcolor=INK_2)

    ax_d = fig.add_subplot(gs[1, 2])
    v_final = result.v_final[0] if result.v_final.dim() == 3 else result.v_final
    im_d, n_closed_d = plot_vmem_map(ax_d, v_final, gap_junctions)
    ax_d.set_title(f"Reprogrammed at t = {t_relax * 1e3:.0f} ms\n"
                   f"(stimulus off), {n_closed_d} closed junctions", fontsize=10, loc="left")
    _panel_label(ax_d, "D")
    cb = fig.colorbar(im_d, ax=ax_d, fraction=0.046, pad=0.04)
    cb.set_label("V_mem (mV)", color=INK_2, fontsize=8)
    cb.ax.tick_params(labelsize=7, colors=MUTED, labelcolor=INK_2)

    fig.suptitle(
        f"Bioelectric tumor reprogramming: pattern error reduced by "
        f"{100 * result.error_reduction:.1f}%",
        fontsize=12, color=INK, x=0.06, ha="left",
    )
    if path:
        fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    return fig


# ----------------------------------------------------------------- ion-level model
def ionic_iv_curve(tissue, v_range=(-0.10, 0.03), n: int = 300):
    """Steady-state membrane I-V curve of one cell of an ion-level tissue.

    Each voltage is set with :meth:`IonicTissue.with_vmem` at the profile
    concentrations, gates at steady state. Returns ``(V [V], I_out [A])`` with
    ``I_out = -F vol sum_k z_k dc_k/dt`` (outward positive), gap junctions excluded.
    """
    from .constants import F as FARADAY

    v = torch.linspace(v_range[0], v_range[1], n, dtype=DTYPE)
    state = tissue.with_vmem(tissue.initial_state((n,)), v.unsqueeze(0))
    with torch.no_grad():
        dx = tissue.rhs(state.x, gap_junctions=False)[:, : tissue.n_ions]
        z = tissue.profile.z.reshape(1, -1, 1)
        i_out = -FARADAY * tissue.geometry.volume * (z * dx).sum(dim=1)[0]
    return v, i_out


def plot_ionic_iv(ax, tissue, v_range=(-0.10, 0.03)) -> None:
    """Plot the ion-level single-cell I-V curve and mark its zero crossings."""
    v, i = ionic_iv_curve(tissue, v_range)
    vm, ip = _np(v) * 1e3, _np(i) * 1e12
    ax.axhline(0.0, color=BASELINE, lw=0.8, zorder=1)
    ax.plot(vm, ip, color=SERIES[2], lw=2.2, zorder=2)
    crossings = np.nonzero(np.sign(ip[:-1]) != np.sign(ip[1:]))[0]
    for k in crossings:
        v0 = vm[k] - ip[k] * (vm[k + 1] - vm[k]) / (ip[k + 1] - ip[k])
        stable = ip[k + 1] > ip[k]
        ax.plot(v0, 0.0, "o", ms=8, zorder=4, mfc=INK if stable else SURFACE, mec=INK, mew=1.5)
        ax.annotate(f"{v0:.0f} mV\n{'stable' if stable else 'saddle'}", (v0, 0.0), textcoords="offset points",
                    xytext=(0, 10 if stable else -26), ha="center", fontsize=7, color=INK_2)
    ax.set_xlabel("V_mem (mV)")
    ax.set_ylabel("Net outward current (pA)")
    ax.grid(True, color=GRID, lw=0.5)
    _style(ax)


def ionic_closed_junctions(v: torch.Tensor, threshold: float = 0.33):
    """Line segments for grid edges whose BETSE gap junction open fraction is below ``threshold``."""
    from .ionic.transport import betse_gj_open_fraction

    gx = betse_gj_open_fraction(v[..., :, 1:] - v[..., :, :-1])
    gy = betse_gj_open_fraction(v[..., 1:, :] - v[..., :-1, :])
    segs = []
    for r, c in zip(*np.nonzero(_np(gx) < threshold)):  # plotting only
        segs.append([(c + 0.5, r - 0.5), (c + 0.5, r + 0.5)])
    for r, c in zip(*np.nonzero(_np(gy) < threshold)):
        segs.append([(c - 0.5, r + 0.5), (c + 0.5, r + 0.5)])
    return segs


def plot_ionic_vmem_map(ax, v: torch.Tensor, v_center_mV: float = -45.0, v_lim_mV=(-90.0, 0.0)):
    """``(H, W)`` V_mem map of an ion-level tissue with closed gap junctions drawn as lines."""
    norm = TwoSlopeNorm(vcenter=v_center_mV, vmin=v_lim_mV[0], vmax=v_lim_mV[1])
    im = ax.imshow(_np(v) * 1e3, cmap=DIVERGING, norm=norm, interpolation="nearest")
    segs = ionic_closed_junctions(v)
    if segs:
        ax.add_collection(LineCollection(segs, colors=INK, linewidths=1.6))
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(BASELINE)
    return im, len(segs)


SEQUENTIAL = LinearSegmentedColormap.from_list(
    "bioelectric_sequential", ["#f0efec", "#b7d3f6", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"]
)


def plot_field_map(ax, field: torch.Tensor, vmax: float | None = None):
    """Non-negative per-cell field (for example a light pattern) on a sequential blue scale."""
    f = _np(field)
    im = ax.imshow(f, cmap=SEQUENTIAL, vmin=0.0, vmax=vmax if vmax is not None else max(float(f.max()), 1e-12),
                   interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(BASELINE)
    return im
