# bioelectrics

**A differentiable engine for somatic bioelectricity, in PyTorch.**

`bioelectrics` simulates the membrane potential of non-neural tissues at the level of ions,
channels, pumps and gap junctions, and computes exact gradients of any outcome with respect
to any parameter. It uses those gradients for the bioelectric inverse problem: which
experimentally realizable intervention, applied for a short time or held constant, moves a
tissue into a target voltage pattern that then persists on its own?

The library builds on the work of Michael Levin's lab, Alexis Pietak (BETSE), and Javier
Cervera and Salvador Mafe. The [README](https://github.com/tlcdv/bioelectrics#related-tools)
states what it takes from each and what it adds.

![Inverse design of interventions on an ion-level tissue](https://raw.githubusercontent.com/tlcdv/bioelectrics/main/assets/ionic_tumor_normalization.png#only-light)
![Inverse design of interventions on an ion-level tissue](https://raw.githubusercontent.com/tlcdv/bioelectrics/main/assets/ionic_tumor_normalization_dark.png#only-dark)

## What it does

| Task | Entry point |
|---|---|
| Simulate V_mem, Na⁺/K⁺/Cl⁻/Ca²⁺ concentrations and channel gates on a grid or any graph | `bioelectrics.ionic.IonicTissue` |
| BETSE-compatible physics: GHK electrodiffusion, Na/K- and Ca-ATPase, voltage-gated electrodiffusive gap junctions | `bioelectrics.ionic` |
| Find steady states, their stability and their slowest time scale, without long simulations | `bioelectrics.attractors.find_steady_state` |
| Differentiate a steady state with respect to any parameter, by the implicit function theorem | `bioelectrics.attractors.steady_state_implicit` |
| Map the attractor landscape: all patterns from many start states, basins, dose ramps with hysteresis | `enumerate_attractors`, `basin_map`, `dose_response` |
| Design an intervention and print it as a lab protocol | `bioelectrics.interventions` |
| Screen drugs whose potency values are published, each with its assay, species and citation | `bioelectrics.pharmacology` |
| Run published models and their reproduction tests | `bioelectrics.zoo` |

## Where to start

- [Getting started](getting-started.md) installs the library and designs a protocol.
- [Validation](validation.md) compares every published and analytic value against what the
  code computes. Each row names the test that asserts it.
- [Physics](physics.md) points at the equations and their sources.
- [Limitations](limitations.md) states what the model does not do. Read it before relying on
  a result.

## Honesty rules this project holds itself to

1. Every biophysical parameter has a published source. No value is estimated to make a
   result come out.
2. A test assertion is never weakened to make it pass. If a test fails, the model, the units
   or the solver is wrong.
3. A model that cannot be reproduced from the public record is left out and the reason is
   stated, rather than guessed at. Two are currently in that position; see
   [Validation](validation.md).
4. A negative result is reported as a result. The drug screen finds that no compound in the
   library normalizes a depolarized domain, and the docs say so.

MIT licensed. Parts of `bioelectrics.ionic` are derived from BETSE (BSD-2-Clause).
