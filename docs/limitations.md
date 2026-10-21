# Limitations

Read this before relying on a result. Each item is a real constraint, not a disclaimer.

## The model

- **The extracellular space is a fixed bath.** There are no extracellular electric fields and
  no extracellular diffusion. This matches BETSE run without its `is_ecm` option.
- **There is no cell volume or osmotic regulation.** A depolarized steady state is reached
  within seconds, but over hours its cells gain Na⁺, up to about 200 mM in the default cell,
  with a slowest time scale near 3 hours. Steady states of depolarized cells are therefore
  not physiological at long times.
- **The channels that produce bistability are phenomenological.** They are CAM-type gating
  functions, not models of a named protein. The BETSE Kir2.1 model alone does not give
  bistability with the default profile.
- **There is no gene regulatory network.** Voltage does not feed back onto expression.

## The drug library

- It contains 10 potency values from 4 primary sources. Each was measured in a heterologous
  expression system or on purified protein, in a named species. A tissue in an embryo is a
  different preparation.
- **Selectivity is not modelled.** A drug acts on one target in the model. Known off-target
  actions are recorded in each entry and reported by `selectivity_warnings`, but not
  simulated. Digoxin is the clearest case: it blocks the Na/K-ATPase about a thousand times
  more potently than Cx43, so at a junction-blocking concentration the pump is blocked too.
- **Hill coefficients are mostly unpublished.** Where none is given, 1 is used and the entry
  says so.
- **The Cx43 values come from a dye-transfer assay**, which measures permeability to calcein
  rather than electrical coupling. These are different measurements.
- **Dihydropyridines are excluded on purpose.** Their block is state-dependent, so one IC50
  is wrong in a tissue whose cells sit at different voltages.
- The screen in `examples/drug_repurposing.py` finds that **no drug in the library normalizes
  a depolarized domain**. Every compound with a verified value acts on a polarizing element,
  so blocking it depolarizes. That is a statement about which potency data are published, not
  about bioelectric medicine.

**These are in-silico hypotheses, not medical advice.**

## Performance

- Backpropagation through time costs more as the simulated interval grows. An 8×8 ion-level
  tissue takes about 4 s per optimization iteration for a 0.5 s rollout.
- Steady-state design avoids that: a 16×16 tissue takes about 9 s per iteration regardless of
  how long the tissue would take to relax.

## Scope of the gradients

`steady_state_implicit` differentiates the attractor the start state relaxes to. It does not
include the jump that happens when a parameter change moves the start state into a different
basin. Near a fold, read the dose response rather than the derivative.
