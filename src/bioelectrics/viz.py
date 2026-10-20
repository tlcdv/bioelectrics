"""Publication-grade Matplotlib figures for bioelectric states.

Two themes share one design language (the palette of theconsciousness.ai):

- ``"ivory"`` (default): ivory background, slate text, for papers and the README.
- ``"console"``: dark instrument-panel background, for web consoles and slides.

``V_mem`` maps use a diverging scale from slate blue (polarized) to orange
(depolarized), centered on the bistability saddle. Non-negative fields (light,
dose) use a sequential ivory -> gold -> orange scale. Switch with
:func:`use_theme` or the :func:`theme_context` context manager.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import Colormap, LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

from .channels import ChannelProfile, find_fixed_points, i_dep, i_pol, ionic_current  # noqa: E402
from .constants import DTYPE  # noqa: E402
from .gap_junctions import GapJunctionProfile, grid_conductances  # noqa: E402


@dataclass(frozen=True)
class Theme:
    """Colors and fonts of one figure theme.

    Every series color has a contrast ratio of at least 3:1 against ``panel``;
    text colors (``ink``, ``ink_2``, ``muted``) have at least 4.5:1, except
    ``muted`` on ivory (3.5:1), which is used only for tick marks.
    """

    name: str
    surface: str  # figure background
    panel: str  # axes background
    ink: str  # titles, panel labels, closed gap junctions
    ink_2: str  # axis labels, annotations
    muted: str  # ticks
    grid: str
    baseline: str  # zero lines and spines
    shade: str  # shaded time windows (intervention phase)
    series: tuple[str, ...]
    diverging: Colormap
    sequential: Colormap


_SERIF = ["Newsreader", "Georgia", "DejaVu Serif"]
_MONO = ["PT Mono", "Consolas", "DejaVu Sans Mono"]

IVORY = Theme(
    name="ivory",
    surface="#f0eee6",
    panel="#faf9f5",
    ink="#1f1e1d",
    ink_2="#3d3d3a",
    muted="#87867f",
    grid="#e8e6dc",
    baseline="#b0aea5",
    shade="#e8e6dc",
    series=("#2f5f86", "#c4623f", "#4f7f6a", "#a07a2c"),
    diverging=LinearSegmentedColormap.from_list(
        "bioelectric_ivory_diverging", ["#1d3f5e", "#4f7fa6", "#e8e6dc", "#d97757", "#8f3a22"]
    ),
    sequential=LinearSegmentedColormap.from_list(
        "bioelectric_ivory_sequential", ["#faf9f5", "#ecd9b4", "#d6ab71", "#d97757", "#8f3a22"]
    ),
)

CONSOLE = Theme(
    name="console",
    surface="#0b0c0e",
    panel="#101215",
    ink="#f0eee6",
    ink_2="#cfcabd",
    muted="#8b877d",
    grid="#1f2125",
    baseline="#4a4b4e",
    shade="#1c1e22",
    series=("#7fa7c9", "#e08e6f", "#8fbfa6", "#d6ab71"),
    diverging=LinearSegmentedColormap.from_list(
        "bioelectric_console_diverging", ["#a9cbe6", "#4f7fa6", "#1a1c20", "#c4623f", "#f3b08f"]
    ),
    sequential=LinearSegmentedColormap.from_list(
        "bioelectric_console_sequential", ["#16181c", "#6b4a3c", "#b0603f", "#e08e6f", "#f6d2b8"]
    ),
)

THEMES = {"ivory": IVORY, "console": CONSOLE}
_active: Theme = IVORY


def theme() -> Theme:
    """The active :class:`Theme`."""
    return _active


def use_theme(name: str | Theme = "ivory") -> Theme:
    """Activate a theme (``"ivory"`` or ``"console"``) for all later figures."""
    global _active
    _active = THEMES[name] if isinstance(name, str) else name
    t = _active
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": _SERIF,
        "font.monospace": _MONO,
        "figure.facecolor": t.surface,
        "savefig.facecolor": t.surface,
        "axes.facecolor": t.panel,
        "axes.edgecolor": t.baseline,
        "axes.labelcolor": t.ink_2,
        "axes.titlecolor": t.ink,
        "text.color": t.ink,
        "xtick.color": t.muted,
        "ytick.color": t.muted,
        "xtick.labelcolor": t.ink_2,
        "ytick.labelcolor": t.ink_2,
        "legend.labelcolor": t.ink_2,
        "grid.color": t.grid,
    })
    return t


@contextlib.contextmanager
def theme_context(name: str | Theme):
    """Use a theme inside a ``with`` block, then restore the previous one."""
    previous = _active
    try:
        yield use_theme(name)
    finally:
        use_theme(previous)


use_theme(IVORY)


def _style(ax) -> None:
    t = _active
    ax.set_facecolor(t.panel)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t.baseline)
    ax.tick_params(colors=t.muted, labelcolor=t.ink_2, labelsize=8)
    ax.xaxis.label.set_color(t.ink_2)
    ax.yaxis.label.set_color(t.ink_2)
    ax.title.set_color(t.ink)


def _frame(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(_active.baseline)


def _panel_label(ax, text: str) -> None:
    ax.text(-0.02, 1.04, text, transform=ax.transAxes, fontsize=13, fontweight="bold",
            color=_active.ink, ha="right", va="bottom")


def style_colorbar(cb, label: str) -> None:
    """Apply the active theme to a colorbar."""
    t = _active
    cb.set_label(label, color=t.ink_2, fontsize=8)
    cb.ax.tick_params(labelsize=7, colors=t.muted, labelcolor=t.ink_2)
    cb.outline.set_edgecolor(t.baseline)


def _np(x) -> np.ndarray:
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def plot_iv_curve(ax, profile: ChannelProfile | None = None, v_range_mV=(-90.0, 5.0)) -> None:
    """Single-cell I-V curve with I_pol, I_dep and their sum; marks the fixed points."""
    profile = profile or ChannelProfile()
    v = torch.linspace(v_range_mV[0], v_range_mV[1], 600, dtype=DTYPE) * 1e-3
    with torch.no_grad():
        curves = [
            ("I_pol", i_pol(v, profile), _active.series[0], 1.5),
            ("I_dep", i_dep(v, profile), _active.series[1], 1.5),
            ("I_ion (total)", ionic_current(v, profile), _active.series[2], 2.2),
        ]
    v_mV = _np(v) * 1e3
    ax.axhline(0.0, color=_active.baseline, lw=0.8, zorder=1)
    for name, i, color, lw in curves:
        ax.plot(v_mV, _np(i) * 1e12, color=color, lw=lw, label=name, zorder=2)
    for v_star, stable in find_fixed_points(profile):
        ax.plot(v_star * 1e3, 0.0, "o", ms=8, zorder=4,
                mfc=_active.ink if stable else _active.panel, mec=_active.ink, mew=1.5)
        ax.annotate(f"{v_star * 1e3:.0f} mV\n{'stable' if stable else 'saddle'}",
                    (v_star * 1e3, 0.0), textcoords="offset points",
                    xytext=(0, 10 if stable else -26), ha="center", fontsize=7, color=_active.ink_2)
    ax.set_xlabel("V_mem (mV)")
    ax.set_ylabel("Outward current (pA)")
    ax.set_ylim(-60, 60)
    ax.grid(True, color=_active.grid, lw=0.5)
    ax.legend(frameon=False, fontsize=7, loc="upper left", labelcolor=_active.ink_2)
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
    im = ax.imshow(_np(v) * 1e3, cmap=_active.diverging, norm=norm, interpolation="nearest")
    n_closed = 0
    if show_closed_junctions:
        thr = 0.5 * float(gap_junctions.g_max + gap_junctions.g_min)
        segs = _closed_junction_segments(v, gap_junctions, thr)
        n_closed = len(segs)
        if segs:
            ax.add_collection(LineCollection(segs, colors=_active.ink, linewidths=1.6))
    _frame(ax)
    return im, n_closed


def plot_loss(ax, loss_history) -> None:
    """Optimization loss per Adam iteration (log scale)."""
    it = np.arange(1, len(loss_history) + 1)
    ax.semilogy(it, loss_history, color=_active.series[0], lw=2, marker="o", ms=5)
    ax.set_xlabel("Adam iteration")
    ax.set_ylabel("Loss")
    ax.grid(True, color=_active.grid, lw=0.5, which="major")
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    _style(ax)


def plot_stimulus_map(ax, stimulus: torch.Tensor):
    """Time-averaged stimulus ``mean_t I_ext(t, x, y)`` in pA. ``stimulus``: ``(K, H, W)``."""
    s = _np(stimulus).mean(axis=0) * 1e12
    lim = max(float(np.abs(s).max()), 1e-3)
    im = ax.imshow(s, cmap=_active.diverging, norm=TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim),
                   interpolation="nearest")
    _frame(ax)
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
    fig = plt.figure(figsize=(12, 9), facecolor=_active.surface)
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
    style_colorbar(cb, "V_mem (mV)")

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
    style_colorbar(cb, "I_ext (pA)")

    ax_d = fig.add_subplot(gs[1, 2])
    v_final = result.v_final[0] if result.v_final.dim() == 3 else result.v_final
    im_d, n_closed_d = plot_vmem_map(ax_d, v_final, gap_junctions)
    ax_d.set_title(f"Reprogrammed at t = {t_relax * 1e3:.0f} ms\n"
                   f"(stimulus off), {n_closed_d} closed junctions", fontsize=10, loc="left")
    _panel_label(ax_d, "D")
    cb = fig.colorbar(im_d, ax=ax_d, fraction=0.046, pad=0.04)
    style_colorbar(cb, "V_mem (mV)")

    fig.suptitle(
        f"Bioelectric tumor reprogramming: pattern error reduced by "
        f"{100 * result.error_reduction:.1f}%",
        fontsize=12, color=_active.ink, x=0.06, ha="left",
    )
    if path:
        fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=_active.surface)
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
    ax.axhline(0.0, color=_active.baseline, lw=0.8, zorder=1)
    ax.plot(vm, ip, color=_active.series[2], lw=2.2, zorder=2)
    crossings = np.nonzero(np.sign(ip[:-1]) != np.sign(ip[1:]))[0]
    for k in crossings:
        v0 = vm[k] - ip[k] * (vm[k + 1] - vm[k]) / (ip[k + 1] - ip[k])
        stable = ip[k + 1] > ip[k]
        ax.plot(v0, 0.0, "o", ms=8, zorder=4, mfc=_active.ink if stable else _active.panel, mec=_active.ink, mew=1.5)
        ax.annotate(f"{v0:.0f} mV\n{'stable' if stable else 'saddle'}", (v0, 0.0), textcoords="offset points",
                    xytext=(0, 10 if stable else -26), ha="center", fontsize=7, color=_active.ink_2)
    ax.set_xlabel("V_mem (mV)")
    ax.set_ylabel("Net outward current (pA)")
    ax.grid(True, color=_active.grid, lw=0.5)
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
    im = ax.imshow(_np(v) * 1e3, cmap=_active.diverging, norm=norm, interpolation="nearest")
    segs = ionic_closed_junctions(v)
    if segs:
        ax.add_collection(LineCollection(segs, colors=_active.ink, linewidths=1.6))
    _frame(ax)
    return im, len(segs)


def plot_field_map(ax, field: torch.Tensor, vmax: float | None = None):
    """Non-negative per-cell field (for example a light pattern) on a sequential blue scale."""
    f = _np(field)
    im = ax.imshow(f, cmap=_active.sequential, vmin=0.0, vmax=vmax if vmax is not None else max(float(f.max()), 1e-12),
                   interpolation="nearest")
    _frame(ax)
    return im
