"""Pattern memory as attractors: hysteresis, basins, and chronic drug design.

Uses :mod:`bioelectrics.attractors` on the ion-level bistable cell (K+ inward
rectifier and depolarizing cation channel, CAM-type gating).

A. Dose ramp of a blocker of the depolarizing channel. Ramping up from the
   depolarized state and down from the polarized state gives two branches: the
   cell remembers its history (hysteresis).
B. Two coupled cells: which steady state each pair of start voltages reaches.
   Mixed patterns (one cell polarized, one depolarized) are stable because the
   voltage-gated gap junction between them closes.
C, D. Chronic drug design on a 16x16 tissue with a depolarized "tumor" domain:
   the constant dose whose steady state is the polarized pattern, found by
   implicit differentiation through the steady state (no simulation in time).

Run:    python examples/attractor_landscape.py
Output: assets/attractor_landscape.png, assets/attractor_landscape_dark.png
"""

from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap

from bioelectrics.attractors import basin_map, dose_response, find_steady_state
from bioelectrics.constants import DTYPE
from bioelectrics.interventions import ChannelBlocker, ChronicInterventionOptimizer
from bioelectrics.ionic import CAT_DEP_CAM, K_IR_CAM, ChannelExpression, IonicTissue, Modifiers
from bioelectrics.viz import _panel_label, _style, plot_ionic_vmem_map, style_colorbar, theme, theme_context

ASSETS = Path(__file__).resolve().parent.parent / "assets"
CHANNELS = [ChannelExpression(K_IR_CAM, 1e-16), ChannelExpression(CAT_DEP_CAM, 1e-16)]
N = 16


def tumor_mask(n: int = N) -> torch.Tensor:
    y, x = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    c = (n - 1) / 2
    blob = ((x - c) / 3.2) ** 2 + ((y - c) / 2.4) ** 2 <= 1.0
    arm = ((x - c - 2.5).abs() <= 1.0) & ((y - c + 3.0).abs() <= 1.5)
    return blob | arm


def main() -> None:
    # A. Dose ramp.
    cell = IonicTissue(edges=torch.zeros(2, 0, dtype=torch.long), channels=CHANNELS)
    at = lambda b: Modifiers(channel_scale={"Cat_dep(CAM)": torch.tensor(1.0 - b, dtype=DTYPE)})
    doses = np.linspace(0.0, 0.9, 37)
    start = lambda v: cell.with_vmem(cell.initial_state((1,)), torch.full((1, 1), v, dtype=DTYPE))
    up, st_up = dose_response(cell, start(-0.003), at, doses)
    down, st_down = dose_response(cell, start(-0.078), at, doses[::-1])
    up, down = 1e3 * up[:, 0].numpy(), 1e3 * down.flip(0)[:, 0].numpy()
    st_down = st_down[::-1]
    print(f"A: ramp up switches at block {doses[np.argmax(up < -40)]:.3f}")

    # B. Basin map of two coupled cells.
    pair = IonicTissue(channels=CHANNELS, edges=torch.tensor([[0], [1]]))
    v_axis = np.linspace(-0.09, 0.0, 13)
    labels, atts = basin_map(pair, (2,), torch.tensor([True, False]), torch.tensor([False, True]), v_axis)
    print("B: attractors", [f"({1e3 * float(a.vmem[0]):.1f}, {1e3 * float(a.vmem[1]):.1f}) mV" for a in atts],
          "stable", [a.stable for a in atts])
    # Class of each steady state: each cell polarized (P) or depolarized (D), threshold -40 mV.
    # Unstable steady states (saddles, reached only from starts on their stable manifold) get class 4.
    cls_of = [int(float(a.vmem[0]) > -0.04) * 2 + int(float(a.vmem[1]) > -0.04) if a.stable else 4 for a in atts]
    labels = torch.tensor(cls_of)[labels]
    names = ["P, P", "P, D", "D, P", "D, D", "saddle (unstable)"]

    # C, D. Chronic drug design on a 16x16 tissue.
    tissue = IonicTissue(channels=CHANNELS)
    mask = tumor_mask()
    v0 = torch.full((1, N, N), -0.078, dtype=DTYPE)
    v0[:, mask] = -0.003
    initial = tissue.with_vmem(tissue.initial_state((N, N)), v0)
    before = find_steady_state(tissue, initial)
    target = torch.full_like(v0, float(before.vmem[0, 0, 0]))
    opt = ChronicInterventionOptimizer(tissue, initial, target, [ChannelBlocker("Cat_dep(CAM)", cost=1.0)])
    t0 = time.perf_counter()
    res = opt.optimize(n_iters=40, tol_mV=1.0, verbose=True)
    per_iter = (time.perf_counter() - t0) / len(res.error_history_mV)
    print(res.protocol.to_markdown())
    print(f"C: {per_iter:.2f} s per iteration on {N}x{N} cells ({tissue.n_ions + 4} state components per cell)")

    data = dict(doses=doses, up=up, down=down, st_up=st_up, st_down=st_down, labels=labels, names=names,
                v_axis=v_axis, before=before.vmem[0], res=res, mask=mask)
    ASSETS.mkdir(exist_ok=True)
    for name, suffix in (("ivory", ""), ("console", "_dark")):
        with theme_context(name):
            fig = make_figure(**data)
            fig.savefig(ASSETS / f"attractor_landscape{suffix}.png", dpi=180, bbox_inches="tight")
            plt.close(fig)


def make_figure(doses, up, down, st_up, st_down, labels, names, v_axis, before, res, mask):
    t = theme()
    fig = plt.figure(figsize=(14.5, 8.6))
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.55)

    ax = fig.add_subplot(gs[0, 0:2])
    ax.plot(100 * doses, up, color=t.series[1], lw=2.2, label="ramp up from the depolarized state")
    ax.plot(100 * doses, down, color=t.series[0], lw=2.2, ls="--", label="ramp down from the polarized state")
    both = np.abs(up - down) > 5
    if both.any():
        ax.axvspan(100 * doses[both].min(), 100 * doses[both].max(), color=t.shade, zorder=0)
        ax.text(100 * doses[both].mean(), -45, "two stable states:\nthe cell remembers", ha="center",
                fontsize=8, color=t.ink_2)
    ax.set_xlabel("Block of the depolarizing channel (%)")
    ax.set_ylabel("Steady-state V_mem (mV)")
    ax.set_title("Dose ramp: hysteresis (pattern memory)", fontsize=10, loc="left", color=t.ink)
    ax.legend(frameon=False, fontsize=8, loc="center right")
    ax.grid(True, color=t.grid, lw=0.5)
    _style(ax)
    _panel_label(ax, "A")

    ax = fig.add_subplot(gs[0, 2])
    colors = list(t.series[:4]) + [t.muted]
    cmap = ListedColormap(colors)
    ext = [1e3 * v_axis[0], 1e3 * v_axis[-1], 1e3 * v_axis[0], 1e3 * v_axis[-1]]
    ax.imshow(labels.T.numpy(), origin="lower", cmap=cmap, vmin=-0.5, vmax=4.5, extent=ext,
              interpolation="nearest")
    for k in sorted(set(labels.reshape(-1).tolist())):
        ax.plot([], [], "s", color=colors[k], label=names[k])
    ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0),
              title="steady state\n(P polarized,\nD depolarized)", title_fontsize=7)
    ax.set_xlabel("start V_mem, cell 1 (mV)")
    ax.set_ylabel("start V_mem, cell 2 (mV)")
    ax.set_title("Two coupled cells: basins", fontsize=10, loc="left", color=t.ink)
    _style(ax)
    _panel_label(ax, "B")

    ax = fig.add_subplot(gs[1, 0])
    im, n_closed = plot_ionic_vmem_map(ax, before, v_center_mV=-40.0)
    ax.set_title(f"Steady state, no drug\n{n_closed} closed gap junctions", fontsize=10, loc="left", color=t.ink)
    style_colorbar(fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04), "V_mem (mV)")
    ax.contour(mask.numpy().astype(float), levels=[0.5], colors=t.ink, linewidths=0.8, linestyles=":")
    _panel_label(ax, "C")

    ax = fig.add_subplot(gs[1, 1])
    im, n_closed = plot_ionic_vmem_map(ax, res.v_final[0], v_center_mV=-40.0)
    block = res.protocol.steps[0]["value"]
    ax.set_title(f"Steady state with {100 * block:.0f}% constant block\n{n_closed} closed gap junctions",
                 fontsize=10, loc="left", color=t.ink)
    style_colorbar(fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04), "V_mem (mV)")
    ax.contour(mask.numpy().astype(float), levels=[0.5], colors=t.ink, linewidths=0.8, linestyles=":")
    _panel_label(ax, "D")

    ax = fig.add_subplot(gs[1, 2])
    ax.plot(range(1, len(res.error_history_mV) + 1), res.error_history_mV, color=t.series[0], lw=2,
            marker="o", ms=4)
    ax.set_yscale("log")
    ax.set_xlabel("Adam iteration")
    ax.set_ylabel("RMS error of the steady state (mV)")
    ax.set_title("Implicit differentiation\nthrough the steady state", fontsize=10, loc="left", color=t.ink)
    ax.grid(True, color=t.grid, lw=0.5)
    _style(ax)
    _panel_label(ax, "E")
    return fig


if __name__ == "__main__":
    main()
