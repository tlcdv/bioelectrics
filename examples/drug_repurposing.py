"""An in-silico screen of approved drugs against a depolarized domain.

Every drug in :mod:`bioelectrics.pharmacology` carries a potency value read from
a primary source. This script asks what each one does to a tissue that holds a
depolarized domain, by computing the steady state under a chronic dose
(:func:`bioelectrics.attractors.find_steady_state`), and reports the direction and
size of the effect.

The screen reports what it finds, including when the answer is that nothing in the
library helps. A negative result is the point: it says which potency data exist,
not what bioelectric medicine can do.

These are in-silico hypotheses, not medical advice. Each potency value was
measured in a named species in a heterologous expression system, and off-target
actions are not modelled.

Run:    python examples/drug_repurposing.py
Run again with --reuse to redraw the figures from the cached results without
repeating the steady-state solves.

Output: assets/drug_screen.png (+ _dark), assets/drug_screen.md, assets/drug_screen.json
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from bioelectrics import pharmacology as ph
from bioelectrics.attractors import find_steady_state
from bioelectrics.constants import DTYPE
from bioelectrics.ionic import CAT_DEP_CAM, K_IR_CAM, ChannelExpression, IonicTissue, Modifiers
from bioelectrics.viz import _panel_label, _style, plot_ionic_vmem_map, style_colorbar, theme, theme_context

ASSETS = Path(__file__).resolve().parent.parent / "assets"
N = 10
DOSES = (0.1, 1.0, 10.0, 100.0)  # multiples of the published half effect

# The bistable pair: a K+ inward rectifier and a depolarizing cation channel,
# both with CAM-type gating. Kir2.1 is added so that chloroquine, which is in the
# drug library, has the target it was measured against.
from bioelectrics.ionic import KIR2P1  # noqa: E402

CHANNELS = [ChannelExpression(K_IR_CAM, 1e-16), ChannelExpression(CAT_DEP_CAM, 1e-16),
            ChannelExpression(KIR2P1, 3e-16)]


def modifiers_for(drug: ph.Drug, multiple: float) -> Modifiers:
    """Chronic application of ``drug`` at ``multiple`` times its half effect."""
    blocked = 1.0 / (1.0 + (1.0 / multiple) ** drug.hill_or_default)
    scale = torch.tensor(1.0 - blocked, dtype=DTYPE)
    if drug.target == "nak_scale":
        return Modifiers(nak_scale=scale)
    if drug.target == "gj_scale":
        return Modifiers(gj_scale=scale)
    return Modifiers(channel_scale={drug.target: scale})


def domain_mask(n: int = N) -> torch.Tensor:
    y, x = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    c = (n - 1) / 2
    return ((x - c) / 2.2) ** 2 + ((y - c) / 1.8) ** 2 <= 1.0


def main() -> None:
    tissue = IonicTissue(channels=CHANNELS)
    mask = domain_mask()
    v0 = torch.full((1, N, N), -0.078, dtype=DTYPE)
    v0[:, mask] = -0.003
    initial = tissue.with_vmem(tissue.initial_state((N, N)), v0)

    base = find_steady_state(tissue, initial)
    v_base = base.vmem[0]
    healthy0, domain0 = float(v_base[~mask].median()), float(v_base[mask].mean())
    print(f"untreated steady state: surround {1e3 * healthy0:.1f} mV, "
          f"domain {1e3 * domain0:.1f} mV ({int(mask.sum())} cells)")

    targets = set(c.channel.name for c in CHANNELS) | {"nak_scale", "gj_scale"}
    drugs = [d for d in ph.LIBRARY if d.target in targets]
    print(f"screening {len(drugs)} drugs with a target this tissue has\n")

    cache = ASSETS / "drug_screen.json"
    if "--reuse" in sys.argv and cache.exists():
        saved = json.loads(cache.read_text(encoding="utf-8"))
        print(f"reusing {cache.name}; no steady states recomputed")
        finish(drugs, {k: [tuple(p) for p in v] for k, v in saved["results"].items()},
               v_base, mask, healthy0, domain0)
        return

    results = {}
    for d in drugs:
        row = []
        state = base.state
        for mult in DOSES:
            try:
                ss = find_steady_state(tissue, state, modifiers_for(d, mult), stability=False,
                                       max_iter=800)
                state = ss.state  # continuation: a slow ramp, not a jump
                v = ss.vmem[0]
                row.append((float(v[~mask].median()), float(v[mask].mean())))
            except (RuntimeError, FloatingPointError) as exc:
                print(f"  {d.name} x{mult:g}: solver did not converge ({type(exc).__name__})")
                row.append((float("nan"), float("nan")))
        results[d.name] = row
        h, t = row[-1]
        print(f"  {d.name:26s} at max dose: surround {1e3 * h:7.1f} mV, domain {1e3 * t:7.1f} mV")

    ASSETS.mkdir(exist_ok=True)
    (ASSETS / "drug_screen.json").write_text(json.dumps(
        {"doses": list(DOSES), "surround_V": healthy0, "domain_V": domain0,
         "n_domain_cells": int(mask.sum()),
         "results": {k: [list(p) for p in v] for k, v in results.items()}}, indent=1), encoding="utf-8")
    finish(drugs, results, v_base, mask, healthy0, domain0)


def finish(drugs, results, v_base, mask, healthy0, domain0) -> None:
    """Write the report and both figures from a finished screen."""
    text = report(drugs, results, healthy0, domain0, mask)
    (ASSETS / "drug_screen.md").write_text(text, encoding="utf-8")
    print("\n" + text)
    for name, suffix in (("ivory", ""), ("console", "_dark")):
        with theme_context(name):
            fig = make_figure(drugs, results, v_base, mask, healthy0, domain0)
            fig.savefig(ASSETS / f"drug_screen{suffix}.png", dpi=180, bbox_inches="tight")
            plt.close(fig)


def verdict(h0: float, t0: float, h: float, t: float) -> str:
    """What the drug did to the pattern, in words.

    The surround is judged before the domain. A drug that depolarizes the healthy
    surround has made the tissue worse, even though that shrinks the voltage gap
    between the two regions.
    """
    if not np.isfinite(h) or not np.isfinite(t):
        return "solver did not converge; see the note below"
    d_surround, d_domain = h - h0, t - t0
    if d_surround > 2e-3:
        return f"depolarizes the surround by {1e3 * d_surround:.0f} mV; spreads the domain"
    if d_domain < -20e-3:
        return f"repolarizes the domain by {-1e3 * d_domain:.0f} mV"
    if abs(d_domain) < 2e-3 and abs(d_surround) < 1e-3:
        return "no effect in this model"
    if d_domain > 2e-3:
        return f"depolarizes the domain further by {1e3 * d_domain:.0f} mV"
    return (f"moves the domain by {1e3 * d_domain:.0f} mV and the surround by "
            f"{1e3 * d_surround:.0f} mV; does not remove the domain")


def report(drugs, results, h0, t0, mask) -> str:
    lines = ["# In-silico drug screen", "",
             f"Tissue: {N}x{N} ion-level cells, a {int(mask.sum())}-cell depolarized domain.",
             f"Untreated steady state: surround {1e3 * h0:.1f} mV, domain {1e3 * t0:.1f} mV.",
             "",
             "Doses are multiples of each drug's published half-effect concentration, applied",
             "chronically and raised as a ramp, so each steady state follows from the previous one.",
             "",
             f"| Drug | Status | Target | Surround at {DOSES[-1]:g}x | Domain at {DOSES[-1]:g}x | Effect |",
             "|---|---|---|---|---|---|"]
    for d in drugs:
        h, t = results[d.name][-1]
        lines.append(f"| {d.name} | {d.status} | {d.target} | {1e3 * h:.1f} mV | {1e3 * t:.1f} mV "
                     f"| {verdict(h0, t0, h, t)} |")
    failed = {d.name: [f"{m:g}x" for m, r in zip(DOSES, results[d.name]) if not np.isfinite(r[0])]
              for d in drugs}
    failed = {k: v for k, v in failed.items() if v}
    if failed:
        lines += ["", "## Where the solver did not converge", "",
                  "No steady state was reached within the iteration limit at these doses: "
                  + "; ".join(f"{k} at {', '.join(v)}" for k, v in failed.items()) + ".",
                  "",
                  "Those points are absent from the table and from the curves rather than filled in.",
                  "",
                  "This is a statement about the solver, not proof that no steady state exists.",
                  "Blocking the Na/K-ATPase removes what holds the Na+ and K+ gradients apart, so the",
                  "cell runs down toward an equilibrium far from where it started. The solver limits",
                  "each step to a 25 % concentration change, to stay inside the basin it started in,",
                  "so a long run-down needs more steps than the limit allows. A single uncoupled cell",
                  "does converge under the same block, which supports that reading.", ""]
    lines += ["", "## What this screen does not show", "",
              "No drug in the library normalizes the domain. That is a statement about which",
              "potency values are published, not about bioelectric medicine. The drugs with",
              "verified data act on polarizing elements, the inward rectifier and the Na/K-ATPase,",
              "so blocking them depolarizes. Normalizing a depolarized domain needs a blocker of",
              "the depolarizing conductance that holds it, and this library has no published",
              "potency for such a compound on this target.", "",
              "The gap junction blockers have no effect here because the domain boundary is",
              "already closed by its own transjunctional voltage. Blocking junctions that are",
              "shut changes nothing. Their effect would show on a domain small enough for its",
              "neighbours to repair, which is the regime of Fig. 4 of Cervera et al. (2016).", "",
              "The depolarizing channel in this model is a phenomenological CAM-type cation",
              "channel, not a named protein, so no real drug targets it by construction. Naming",
              "the protein that carries this current in a given tissue is the step that would",
              "make the screen actionable.", "",
              "## Sources", "", ph.sources_markdown(tuple(drugs)), "", "## Caveats", ""]
    for d in drugs:
        lines += [f"**{d.name}**"] + [f"- {c}" for c in d.caveats()] + [""]
    return "\n".join(lines) + "\n"


def make_figure(drugs, results, v_base, mask, h0, t0):
    t = theme()
    fig = plt.figure(figsize=(13.5, 4.8))
    gs = fig.add_gridspec(1, 3, wspace=0.5, width_ratios=[1, 1.5, 1.5])
    centre = 1e3 * 0.5 * (h0 + t0)

    ax = fig.add_subplot(gs[0, 0])
    im, n_closed = plot_ionic_vmem_map(ax, v_base, v_center_mV=centre)
    ax.set_title(f"Untreated steady state\n{n_closed} closed gap junctions",
                 fontsize=10, loc="left", color=t.ink)
    style_colorbar(fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04), "V_mem (mV)")
    _panel_label(ax, "A")

    # Drugs whose curves are identical to the last decimal are drawn once, with a
    # joint label. Five gap junction blockers overlap exactly here, and seven
    # separate lines in four colours would read as a difference that is not there.
    groups: list[tuple[list[str], list[tuple[float, float]]]] = []
    for d in drugs:
        row = results[d.name]
        for names, ref in groups:
            if all(abs(a - c) < 1e-6 and abs(b - e) < 1e-6
                   for (a, b), (c, e) in zip(row, ref)
                   if np.isfinite(a) and np.isfinite(c)):
                names.append(d.name)
                break
        else:
            groups.append(([d.name], row))

    for k, (idx, which, label) in enumerate(((1, 0, "surround"), (2, 1, "domain"))):
        ax = fig.add_subplot(gs[0, idx])
        for j, (names, row) in enumerate(groups):
            ys = [1e3 * r[which] for r in row]
            name = names[0] if len(names) == 1 else f"{len(names)} gap junction blockers"
            ax.plot(DOSES, ys, marker="o", ms=4, lw=1.8,
                    color=t.series[j % len(t.series)], label=name)
        ax.axhline(1e3 * (h0 if which == 0 else t0), color=t.baseline, lw=1.0, ls="--")
        ax.set_xscale("log")
        # Both panels share the full physiological range, so a change of a fraction
        # of a millivolt cannot look like a change of tens.
        ax.set_ylim(1e3 * min(h0, t0) - 12.0, 2.0)
        ax.set_xlabel("dose / published half effect")
        ax.set_ylabel(f"V_mem of the {label} (mV)")
        ax.set_title(f"Effect on the {label}\n(dashed = untreated)", fontsize=10, loc="left", color=t.ink)
        if k == 0:
            ax.legend(frameon=False, fontsize=7, loc="center left")
        ax.grid(True, color=t.grid, lw=0.5)
        _style(ax)
        _panel_label(ax, "BC"[k])
    return fig


if __name__ == "__main__":
    main()
