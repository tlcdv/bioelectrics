"""Tumor reprogramming demo: cancer normalization by a brief bioelectric trigger.

1. A 32x32 polarized tissue (-70 mV) carries a 10x10 depolarized tumor (-15 mV).
   High transjunctional voltage closes the boundary gap junctions, so the tumor is
   electrically isolated: the computational boundary of the self shrinks.
2. ``PatternOptimizer`` finds a stimulus I_ext(t, x, y) in [0, T_stim] by BPTT.
3. The stimulus is switched off for [T_stim, T_relax]. The tissue stays in the
   healthy attractor and the gap junctions re-open.

Run:  python examples/tumor_reprogramming_demo.py
Output: assets/inverse_reprogramming_result.png
"""

from pathlib import Path

import torch

from bioelectrics import ChannelProfile, GapJunctionProfile, Tissue, find_fixed_points, tumor_prepattern
from bioelectrics.control import PatternOptimizer
from bioelectrics.viz import make_reprogramming_figure

GRID, PATCH = 32, 10
T_STIM, T_RELAX, DT = 0.1, 0.5, 5e-4


def main() -> None:
    torch.manual_seed(0)
    profile, gap_junctions = ChannelProfile(), GapJunctionProfile()
    tissue = Tissue(channels=profile, gap_junctions=gap_junctions)
    v_healthy = find_fixed_points(profile)[0][0]

    # Let the tumor settle into its depolarized attractor behind closed junctions.
    raw, mask = tumor_prepattern(GRID, GRID, PATCH, v_healthy_mV=-70.0, v_tumor_mV=-15.0)
    with torch.no_grad():
        initial = tissue.simulate(raw, duration=0.5)

    opt = PatternOptimizer(
        tissue,
        initial.v,
        torch.full_like(initial.v, v_healthy),
        t_stim=T_STIM,
        t_relax=T_RELAX,
        dt=DT,
        n_time_bins=4,
        lr=0.5,
    )
    result = opt.optimize(n_iters=30, tol=1e-5, verbose=True)

    print(f"baseline RMS error : {result.baseline_error * 1e3:8.3f} mV")
    print(f"final RMS error    : {result.final_error * 1e3:8.3f} mV")
    print(f"error reduction    : {100 * result.error_reduction:8.2f} %")
    print(f"max tumor V(T_relax): {result.v_final[0][mask].max().item() * 1e3:.2f} mV")

    out = Path(__file__).resolve().parent.parent / "assets" / "inverse_reprogramming_result.png"
    out.parent.mkdir(exist_ok=True)
    make_reprogramming_figure(profile, gap_junctions, initial.v[0], result, T_STIM, T_RELAX, str(out))
    print(f"saved figure       : {out}")


if __name__ == "__main__":
    main()
