"""Inverse design of lab interventions that normalize a depolarized tumor domain.

Ion-level tissue (Na+, K+, Cl-, Ca2+ with GHK fluxes, Na/K- and Ca-ATPase,
BETSE voltage-gated gap junctions). Each cell expresses a K+ inward rectifier and a
depolarizing cation channel (CAM-type gating), so it has two attractors:
about -79 mV (healthy) and -3 mV (tumor-like). A tumor domain is electrically
isolated because its boundary gap junctions close.

Two questions are answered by backpropagation through the simulation:

Drug protocol: which dose of a bath-applied drug that blocks the depolarizing channel,
   given for 0.1 s and then washed out, normalizes the tissue?
Light protocol: cells express an anion channelrhodopsin (light-gated Cl- channel). Which light
   pattern, shown for 0.1 s, normalizes the tissue with the least light?

Run:    python examples/ionic_tumor_normalization.py
Output: assets/ionic_tumor_normalization.png, assets/ionic_protocols.md
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from bioelectrics.constants import DTYPE
from bioelectrics.interventions import ChannelBlocker, InterventionOptimizer, Optogenetic
from bioelectrics.ionic import CAT_DEP_CAM, K_IR_CAM, ChannelExpression, IonicState, IonicTissue, Modifiers, leak
from bioelectrics.viz import (
    INK,
    INK_2,
    MUTED,
    SERIES,
    SURFACE,
    _panel_label,
    _style,
    plot_field_map,
    plot_ionic_iv,
    plot_ionic_vmem_map,
)

N = 16
T_STIM, T_RELAX = 0.1, 0.5
V_HEALTHY, V_TUMOR = -0.078, -0.003
ASSETS = Path(__file__).resolve().parent.parent / "assets"


def tumor_mask(n: int = N) -> torch.Tensor:
    y, x = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    c = (n - 1) / 2
    blob = ((x - c) / 3.2) ** 2 + ((y - c) / 2.4) ** 2 <= 1.0
    arm = ((x - c - 2.5).abs() <= 1.0) & ((y - c + 3.0).abs() <= 1.5)
    return blob | arm


def main() -> None:
    torch.manual_seed(0)
    opsin = ChannelExpression(leak("Cl"), 3e-16)  # anion channelrhodopsin, E_Cl ~ -90 mV
    tissue = IonicTissue(
        channels=[ChannelExpression(K_IR_CAM, 1e-16), ChannelExpression(CAT_DEP_CAM, 1e-16), opsin]
    )
    dark = Modifiers(channel_gate={"ClLeak": torch.zeros(N, N, dtype=DTYPE)})

    mask = tumor_mask()
    v0 = torch.full((1, N, N), V_HEALTHY, dtype=DTYPE)
    v0[:, mask] = V_TUMOR
    state = tissue.simulate(tissue.with_vmem(tissue.initial_state((N, N)), v0), 0.5, modifiers=dark)
    state = IonicState(state.x, 0.0)  # t = 0 is the start of the intervention
    v_init = tissue.vmem_of(state)[0]
    target = torch.full_like(tissue.vmem_of(state), V_HEALTHY)
    print(f"tumor cells: {int(mask.sum())}, tumor V {1e3 * v_init[mask].mean():.1f} mV, "
          f"healthy V {1e3 * v_init[~mask].mean():.1f} mV")

    # Drug protocol: bath-applied blocker of the depolarizing channel (one scalar dose).
    drug = ChannelBlocker("Cat_dep(CAM)", cost=1.0)
    opt_a = InterventionOptimizer(tissue, state, target, [drug], T_STIM, T_RELAX, background=dark)
    res_a = opt_a.optimize(n_iters=30, tol_mV=1.0, verbose=True)

    # Light protocol: light pattern for the anion channelrhodopsin (one value per cell, L1 cost).
    light = Optogenetic("ClLeak", init=0.02, spatial=True, cost=20.0)
    opt_b = InterventionOptimizer(tissue, state, target, [light], T_STIM, T_RELAX, background=dark)
    res_b = opt_b.optimize(n_iters=60, tol_mV=1.0, verbose=True)

    # Time course of one tumor cell with and without the drug.
    times = [k * 0.005 for k in range(int(T_RELAX / 0.005) + 1)]
    r, c = (int(i) for i in mask.nonzero()[len(mask.nonzero()) // 2])
    with torch.no_grad():
        mods_a = opt_a.modifiers()
        _, _, tr_on = tissue.simulate(state, T_STIM, modifiers=mods_a, t_eval=[t for t in times if t <= T_STIM])
        mid = tissue.simulate(state, T_STIM, modifiers=mods_a)
        _, _, tr_off = tissue.simulate(mid, T_RELAX - T_STIM, modifiers=dark,
                                       t_eval=[t for t in times if t > T_STIM - 1e-12])
        _, _, tr_ctrl = tissue.simulate(state, T_RELAX, modifiers=dark, t_eval=times)
    vm = lambda tr: (1e3 * tissue.vmem(tr[:, :, : tissue.n_ions].squeeze(1))[:, r, c]).numpy()
    t_drug = [t for t in times if t <= T_STIM] + [t for t in times if t > T_STIM - 1e-12]
    v_drug = list(vm(tr_on)) + list(vm(tr_off))

    # Figure.
    fig = plt.figure(figsize=(14.5, 8.6), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.62)

    ax = fig.add_subplot(gs[0, 0])
    single = IonicTissue(edges=torch.zeros(2, 0, dtype=torch.long), channels=tissue.channels[:2])
    plot_ionic_iv(ax, single)
    ax.set_title("Ion-level cell: two attractors", fontsize=10, loc="left", color=INK)
    _panel_label(ax, "A")

    ax = fig.add_subplot(gs[0, 1])
    im, n_closed = plot_ionic_vmem_map(ax, v_init, v_center_mV=-37.0)
    ax.set_title(f"Tumor domain, {n_closed} closed\ngap junctions (black)", fontsize=10, loc="left", color=INK)
    _panel_label(ax, "B")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("V_mem (mV)", color=INK_2, fontsize=8)
    cb.ax.tick_params(labelsize=7, colors=MUTED, labelcolor=INK_2)

    ax = fig.add_subplot(gs[0, 2])
    ax.axvspan(0, 1e3 * T_STIM, color="#f0efec", zorder=0)
    ax.plot([1e3 * t for t in times], vm(tr_ctrl), color=MUTED, lw=1.8, label="no drug")
    ax.plot([1e3 * t for t in t_drug], v_drug, color=SERIES[0], lw=2.2, label="drug for 0.1 s")
    ax.axhline(-37.0, color=MUTED, lw=0.8, ls=":")
    ax.text(1e3 * T_RELAX, -35.5, "threshold", ha="right", fontsize=7, color=INK_2)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("V_mem of a tumor cell (mV)")
    blocked = res_a.protocol.steps[0]["value"]
    ax.set_title(f"Drug protocol: {100 * blocked:.0f}% block\nfor 0.1 s, then washout", fontsize=10, loc="left", color=INK)
    ax.legend(frameon=False, fontsize=7, labelcolor=INK_2, loc="center right")
    ax.grid(True, color="#e1e0d9", lw=0.5)
    _style(ax)
    _panel_label(ax, "C")

    ax = fig.add_subplot(gs[1, 0])
    im = plot_field_map(ax, light.field().detach())
    ax.contour(mask.numpy().astype(float), levels=[0.5], colors=INK, linewidths=1.0)
    ax.set_title("Light protocol: optimized\npattern (outline = tumor)", fontsize=10, loc="left", color=INK)
    _panel_label(ax, "D")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("opsin open probability", color=INK_2, fontsize=8)
    cb.ax.tick_params(labelsize=7, colors=MUTED, labelcolor=INK_2)

    ax = fig.add_subplot(gs[1, 1])
    im, n_closed = plot_ionic_vmem_map(ax, res_b.v_final[0], v_center_mV=-37.0)
    ax.set_title(f"After light is off (t = {1e3 * T_RELAX:.0f} ms)\n{n_closed} closed gap junctions",
                 fontsize=10, loc="left", color=INK)
    _panel_label(ax, "E")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("V_mem (mV)", color=INK_2, fontsize=8)
    cb.ax.tick_params(labelsize=7, colors=MUTED, labelcolor=INK_2)

    ax = fig.add_subplot(gs[1, 2])
    ax.plot(range(1, len(res_a.error_history_mV) + 1), res_a.error_history_mV, color=SERIES[0], lw=2,
            marker="o", ms=4, label="drug protocol")
    ax.plot(range(1, len(res_b.error_history_mV) + 1), res_b.error_history_mV, color=SERIES[1], lw=2,
            marker="o", ms=4, label="light protocol")
    ax.set_yscale("log")
    ax.set_xlabel("Adam iteration")
    ax.set_ylabel("RMS pattern error at T_relax (mV)")
    ax.legend(frameon=False, fontsize=7, labelcolor=INK_2)
    ax.grid(True, color="#e1e0d9", lw=0.5)
    ax.set_title("Backpropagation through the\nion-level simulation", fontsize=10, loc="left", color=INK)
    _style(ax)
    _panel_label(ax, "F")

    ASSETS.mkdir(exist_ok=True)
    fig.savefig(ASSETS / "ionic_tumor_normalization.png", dpi=180, bbox_inches="tight", facecolor=SURFACE)
    report = "# Optimized protocols\n\n## Drug protocol (bath-applied blocker)\n\n" + res_a.protocol.to_markdown()
    report += "\n\n## Light protocol (anion channelrhodopsin)\n\n" + res_b.protocol.to_markdown() + "\n"
    (ASSETS / "ionic_protocols.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"light in tumor (mean): {light.field()[mask].mean():.3f}, outside: {light.field()[~mask].mean():.4f}")


if __name__ == "__main__":
    main()
