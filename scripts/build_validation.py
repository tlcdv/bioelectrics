"""Generate the validation page from real runs.

Every number on the page is recomputed here, so the documentation cannot drift
away from the code. Nothing is copied from a paper except the published values
themselves, which are facts and are cited.

Published figures are not reproduced. Comparing numbers is both the honest check
and the one that avoids copying copyrighted images.

Run:    python scripts/build_validation.py
Output: docs/validation.md, docs/assets/validation.png (+ _dark)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from bioelectrics import find_fixed_points
from bioelectrics.attractors import find_steady_state, steady_state_implicit
from bioelectrics.constants import DTYPE, F, R, T
from bioelectrics.ionic import CAT_DEP_CAM, K_IR_CAM, ChannelExpression, IonicTissue, Modifiers
from bioelectrics.viz import _panel_label, _style, theme, theme_context
from bioelectrics.zoo import cam2016

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"


@dataclass
class Check:
    """One comparison between a reference value and what the library computes."""

    group: str
    quantity: str
    reference: float
    computed: float
    unit: str
    tolerance: float
    source: str
    test: str

    @property
    def difference(self) -> float:
        return self.computed - self.reference

    @property
    def passed(self) -> bool:
        return abs(self.difference) <= self.tolerance


def cam2016_checks() -> list[Check]:
    """Cervera, Alcaraz & Mafe (2016): the three fixed points of Fig. 2 and Fig. 4."""
    src = cam2016.REFERENCE
    test = "tests/zoo/test_cam2016.py::test_fig2_single_cell_fixed_points"
    published = {"polarized": -58.6, "saddle": -15.5, "depolarized": -7.5}
    fps = find_fixed_points(cam2016.channel_profile())
    assert [s for _, s in fps] == [True, False, True], "the model lost its bistability"
    out = []
    for (name, ref), (v, _) in zip(published.items(), fps):
        out.append(Check("Cervera, Alcaraz & Mafe (2016)", f"{name} fixed point", ref,
                         1e3 * v, "mV", 0.5, src, test))
    return out


def analytic_checks() -> list[Check]:
    """Limits where the ion-level model has a closed-form answer."""
    out = []
    # The same construction as tests/test_ionic_physics.py: one uncoupled cell whose
    # membrane passes only K+, with both pumps switched off.
    d = torch.zeros(6, dtype=DTYPE)
    tissue = IonicTissue(edges=torch.zeros(2, 0, dtype=torch.long),
                         membrane_d=d.index_fill(0, torch.tensor([1]), 1e-18),
                         alpha_nak=0.0, alpha_ca=0.0)
    res = tissue.simulate(tissue.initial_state((1,)), 16.0, dt=5e-3)
    c = res.x[0, :, 0]
    e_k = (R * T / F) * math.log(tissue.profile.c_out[1].item() / c[1].item())
    out.append(Check("Analytic limits", "K+-only membrane reaches the Nernst potential",
                     1e3 * e_k, 1e3 * tissue.vmem_of(res).item(), "mV", 0.2,
                     "Nernst equation", "tests/test_ionic_physics.py::test_nernst_potential_with_only_potassium"))

    cell = IonicTissue(edges=torch.zeros(2, 0, dtype=torch.long),
                       channels=[ChannelExpression(K_IR_CAM, 1e-16), ChannelExpression(CAT_DEP_CAM, 1e-16)])
    start = cell.with_vmem(cell.initial_state((1,)), torch.full((1, 1), -0.078, dtype=DTYPE))
    ss = find_steady_state(cell, start)
    settled = cell.simulate(type(ss.state)(ss.state.x.clone(), 0.0), 1.0)
    drift = 1e3 * float((cell.vmem_of(settled) - ss.vmem).abs().max())
    out.append(Check("Analytic limits", "a steady state does not move in a 1 s simulation",
                     0.0, drift, "mV", 1e-3,
                     "definition of a steady state", "tests/test_attractors.py::test_steady_states_are_fixed_points_of_the_simulation"))
    return out


def gradient_check() -> tuple[Check, float]:
    """The implicit gradient against a central finite difference."""
    tissue = IonicTissue(channels=[ChannelExpression(K_IR_CAM, 1e-16),
                                   ChannelExpression(CAT_DEP_CAM, 1e-16)])
    v0 = torch.full((1, 4, 4), -0.078, dtype=DTYPE)
    v0[:, 1:3, 1:3] = -0.003
    state = tissue.with_vmem(tissue.initial_state((4, 4)), v0)
    ramp = torch.linspace(0, 1, 16, dtype=DTYPE).reshape(4, 4)
    block = torch.tensor(0.3, dtype=DTYPE, requires_grad=True)
    mods = lambda b=block: Modifiers(channel_scale={"Cat_dep(CAM)": 1 - b * ramp})

    t0 = time.perf_counter()
    x = steady_state_implicit(tissue, state, mods, [block])
    (tissue.vmem_of(x) ** 2).sum().backward()
    implicit_time = time.perf_counter() - t0

    def loss(b: float) -> float:
        ss = find_steady_state(tissue, state, mods(torch.tensor(b, dtype=DTYPE)), stability=False)
        return float((ss.vmem ** 2).sum())

    eps = 1e-4
    fd = (loss(0.3 + eps) - loss(0.3 - eps)) / (2 * eps)
    rel = abs(float(block.grad) - fd) / abs(fd)
    return Check("Gradients", "implicit gradient against a central finite difference",
                 fd, float(block.grad), "dimensionless", 1e-4 * abs(fd),
                 "implicit function theorem",
                 "tests/test_attractors.py::test_implicit_gradient_matches_finite_difference"), rel


def main() -> None:
    checks = cam2016_checks() + analytic_checks()
    grad_check, rel = gradient_check()
    checks.append(grad_check)

    DOCS.mkdir(exist_ok=True)
    (DOCS / "assets").mkdir(exist_ok=True)
    for name, suffix in (("ivory", ""), ("console", "_dark")):
        with theme_context(name):
            fig = make_figure(checks)
            fig.savefig(DOCS / "assets" / f"validation{suffix}.png", dpi=180, bbox_inches="tight")
            plt.close(fig)

    (DOCS / "validation.md").write_text(page(checks, rel), encoding="utf-8")
    failed = [c for c in checks if not c.passed]
    print(f"{len(checks) - len(failed)}/{len(checks)} checks within tolerance")
    for c in failed:
        print(f"  FAILED {c.quantity}: {c.computed:.4g} vs {c.reference:.4g} {c.unit}")
    print(f"wrote {DOCS / 'validation.md'}")
    # A failing check must stop a release, not be published as if it held. CI runs
    # this script, so a non-zero exit fails the build.
    if failed:
        raise SystemExit(1)


def page(checks: list[Check], rel: float) -> str:
    lines = [
        "# Validation",
        "",
        "Every number on this page is recomputed by `scripts/build_validation.py` from the",
        "code in this repository. None of it is copied from a previous run or edited by hand.",
        "",
        "Published figures are not reproduced here. The check is a comparison of numbers,",
        "which is the part that can be verified, and it avoids copying copyrighted images.",
        "",
    ]
    for group in dict.fromkeys(c.group for c in checks):
        rows = [c for c in checks if c.group == group]
        lines += [f"## {group}", "",
                  "| Quantity | Reference | Computed | Difference | Within tolerance | Test |",
                  "|---|---|---|---|---|---|"]
        for c in rows:
            unit = "" if c.unit == "dimensionless" else f" {c.unit}"
            lines.append(
                f"| {c.quantity} | {c.reference:.4g}{unit} | {c.computed:.4g}{unit} "
                f"| {c.difference:+.3g}{unit} | {'yes' if c.passed else 'NO'} | `{c.test}` |")
        sources = list(dict.fromkeys(c.source for c in rows))
        lines += ["", "Reference: " + "; ".join(sources), ""]

    lines += [
        "![Validation](assets/validation.png#only-light)",
        "![Validation](assets/validation_dark.png#only-dark)",
        "",
        "## What these checks do and do not establish",
        "",
        "They establish that the implementation matches the published values and the analytic",
        "limits it claims to match. They do not establish that the model describes any",
        "particular tissue. The parameters come from the sources named above, measured in",
        "preparations that differ from any tissue a user may have in mind.",
        "",
        f"The implicit gradient agrees with a central finite difference to a relative error of",
        f"{rel:.1e}, which is the check that the attractor derivatives are correct rather than",
        "merely plausible.",
        "",
        "## What is not reproduced, and why",
        "",
        "Two models the library would otherwise include cannot be reproduced from the public",
        "record:",
        "",
        "- **Manicka & Levin (2019), the BEN model.** The equations in its Fig. 8 are",
        "  dimensionally inconsistent as printed: there is no Faraday constant in the V_mem",
        "  expression, the channel flux sign is reversed, and the pump term is about 10^7 times",
        "  the channel flux. Rather than guess, the model is left out.",
        "- **Pai et al. (2020), the frog brain patterning model.** The equations are in the",
        "  paper, but G_pol, G_dep, E_pol, E_dep and the patch geometry are in the",
        "  supplementary material, which the paper states is available from the corresponding",
        "  author on request.",
        "",
        "Both are listed here rather than omitted silently, because an absent reproduction is",
        "information too.",
        "",
    ]
    return "\n".join(lines)


def short(quantity: str) -> str:
    """A label that fits beside a bar."""
    table = {
        "K+-only membrane reaches the Nernst potential": "Nernst potential",
        "a steady state does not move in a 1 s simulation": "steady state holds",
        "implicit gradient against a central finite difference": "implicit gradient",
    }
    return table.get(quantity, quantity)


def make_figure(checks: list[Check]):
    t = theme()
    cam = [c for c in checks if c.group.startswith("Cervera")]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.4), gridspec_kw={"wspace": 0.08})

    y = np.arange(len(cam))
    ax.barh(y - 0.18, [c.reference for c in cam], height=0.34, color=t.muted, label="published")
    ax.barh(y + 0.18, [c.computed for c in cam], height=0.34, color=t.series[0], label="this library")
    ax.set_yticks(y)
    ax.set_yticklabels([c.quantity.replace(" fixed point", "") for c in cam], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("V_mem (mV)")
    ax.set_title("Cervera, Alcaraz & Mafe (2016)\nfixed points", fontsize=10, loc="left", color=t.ink)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(True, axis="x", color=t.grid, lw=0.5)
    _style(ax)
    _panel_label(ax, "A")

    margins = [abs(c.difference) / c.tolerance if c.tolerance else 0.0 for c in checks]
    y2 = np.arange(len(checks))
    ax2.barh(y2, [max(m, 1e-4) for m in margins], height=0.55,
             color=[t.series[0] if c.passed else t.series[1] for c in checks])
    ax2.axvline(1.0, color=t.baseline, lw=1.0, ls="--")
    ax2.text(0.99, 1.02, "tolerance", fontsize=7, color=t.ink_2, ha="right",
             transform=ax2.transAxes)
    # The labels sit on the right, so they cannot run into the left panel.
    ax2.yaxis.tick_right()
    ax2.set_yticks(y2)
    ax2.set_yticklabels([short(c.quantity) for c in checks], fontsize=7)
    ax2.invert_yaxis()
    ax2.set_xscale("log")
    ax2.set_xlabel("difference / tolerance")
    ax2.set_title("Every check, as a fraction\nof its tolerance", fontsize=10, loc="left", color=t.ink)
    ax2.grid(True, axis="x", color=t.grid, lw=0.5)
    _style(ax2)
    _panel_label(ax2, "B")
    return fig


if __name__ == "__main__":
    main()
