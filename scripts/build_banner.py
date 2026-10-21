"""Generate the README banner.

It matches the banner of the main project (black ground, the project emblem, the
name in a bold serif, an uppercase sans subtitle) and adds one thing that is not
decoration: the map on the right is computed by this library. It is the steady
state of a 22x22 ion-level tissue holding a depolarized domain, and the light
segments are the gap junctions that the domain's own voltage has closed.

Run:    python scripts/build_banner.py
Output: assets/banner.png (2000x476, shown at 1000x238 in the README)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

from bioelectrics.attractors import find_steady_state  # noqa: E402
from bioelectrics.constants import DTYPE  # noqa: E402
from bioelectrics.ionic import CAT_DEP_CAM, K_IR_CAM, ChannelExpression, IonicTissue  # noqa: E402
from bioelectrics.viz import ionic_closed_junctions  # noqa: E402

ASSETS = Path(__file__).resolve().parent.parent / "assets"
OUT = ASSETS / "banner.png"
# The project emblem, on a transparent background. On black its disc reads as a
# very dark brown, so the emblem keeps a visible edge.
LOGO = ASSETS / "logo.png"
N = 22

BLACK = "#000000"
WHITE = "#f0eee6"
DIM = "#8b877d"
CLAY = "#d97757"
# The console palette of theconsciousness.ai: polarized cells read as cool, the
# depolarized domain as clay, and the saddle region as near-black.
VMEM = LinearSegmentedColormap.from_list(
    "banner", ["#7fa7c9", "#3d6a8f", "#16181c", "#b0603f", "#e08e6f"])


def domain(n: int) -> torch.Tensor:
    """An irregular domain, so the boundary reads as grown rather than drawn."""
    y, x = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    c = (n - 1) / 2
    blob = ((x - c - 1.0) / 5.2) ** 2 + ((y - c) / 4.0) ** 2 <= 1.0
    arm = ((x - c + 4.2).abs() <= 1.6) & ((y - c - 2.4).abs() <= 2.6)
    return blob | arm


def compute() -> tuple[torch.Tensor, list]:
    tissue = IonicTissue(channels=[ChannelExpression(K_IR_CAM, 1e-16),
                                   ChannelExpression(CAT_DEP_CAM, 1e-16)])
    mask = domain(N)
    v0 = torch.full((1, N, N), -0.078, dtype=DTYPE)
    v0[:, mask] = -0.003
    ss = find_steady_state(tissue, tissue.with_vmem(tissue.initial_state((N, N)), v0),
                           stability=False)
    v = ss.vmem[0]
    return v, ionic_closed_junctions(v)


def main() -> None:
    v, segs = compute()
    fig = plt.figure(figsize=(10.0, 2.38), dpi=200, facecolor=BLACK)

    # The project emblem, held to the left edge of its box at its own aspect ratio.
    mark = fig.add_axes([0.022, 0.07, 0.20, 0.86])
    mark.imshow(plt.imread(LOGO), interpolation="lanczos")
    mark.set_anchor("W")
    mark.axis("off")

    fig.text(0.245, 0.56, "bioelectrics", fontsize=40, fontweight="bold", color=WHITE,
             family=["Georgia", "DejaVu Serif"], va="center")
    fig.text(0.248, 0.30, "DIFFERENTIABLE SOMATIC BIOELECTRICITY  |  PYTORCH",
             fontsize=10.5, color=WHITE, family=["Arial", "DejaVu Sans"], va="center")
    fig.text(0.248, 0.17, "theConsciousness.ai  ·  open source", fontsize=8.5, color=DIM,
             family=["Arial", "DejaVu Sans"], va="center")

    # A thin clay rule between the words and the data.
    fig.add_artist(plt.Line2D([0.705, 0.705], [0.14, 0.86], color=CLAY, lw=1.2,
                              transform=fig.transFigure))

    ax = fig.add_axes([0.73, 0.05, 0.24, 0.90])
    # vmin below the healthy voltage, so healthy tissue reads as the deeper blue
    # and the clay domain carries the eye rather than a pale block.
    norm = TwoSlopeNorm(vcenter=-40.0, vmin=-105.0, vmax=0.0)
    ax.imshow(1e3 * v.numpy(), cmap=VMEM, norm=norm, interpolation="nearest")
    if segs:
        ax.add_collection(LineCollection(segs, colors=WHITE, linewidths=1.3))
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=200, facecolor=BLACK)
    plt.close(fig)
    print(f"wrote {OUT}  ({len(segs)} closed gap junctions drawn)")


if __name__ == "__main__":
    main()
