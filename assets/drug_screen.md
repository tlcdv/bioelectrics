# In-silico drug screen

Tissue: 10x10 ion-level cells, a 12-cell depolarized domain.
Untreated steady state: surround -81.1 mV, domain -12.8 mV.

Doses are multiples of each drug's published half-effect concentration, applied
chronically and raised as a ramp, so each steady state follows from the previous one.

| Drug | Status | Target | Surround at 100x | Domain at 100x | Effect |
|---|---|---|---|---|---|
| chloroquine | approved | Kir2.1 | -77.3 mV | -12.7 mV | depolarizes the surround by 4 mV; spreads the domain |
| ouabain | tool compound | nak_scale | -11.3 mV | -11.3 mV | depolarizes the surround by 70 mV; spreads the domain |
| carbenoxolone | approved | gj_scale | -81.1 mV | -12.4 mV | no effect in this model |
| meclofenamic acid | approved | gj_scale | -81.1 mV | -12.4 mV | no effect in this model |
| heptanol | tool compound | gj_scale | -81.1 mV | -12.4 mV | no effect in this model |
| digoxin | approved | gj_scale | -81.1 mV | -12.4 mV | no effect in this model |
| phorbol 12,13-dibutyrate | research compound | gj_scale | -81.1 mV | -12.4 mV | no effect in this model |

## Where the solver did not converge

No steady state was reached within the iteration limit at these doses: ouabain at 1x, 10x.

Those points are absent from the table and from the curves rather than filled in.

This is a statement about the solver, not proof that no steady state exists.
Blocking the Na/K-ATPase removes what holds the Na+ and K+ gradients apart, so the
cell runs down toward an equilibrium far from where it started. The solver limits
each step to a 25 % concentration change, to stay inside the basin it started in,
so a long run-down needs more steps than the limit allows. A single uncoupled cell
does converge under the same block, which supports that reading.


## What this screen does not show

No drug in the library normalizes the domain. That is a statement about which
potency values are published, not about bioelectric medicine. The drugs with
verified data act on polarizing elements, the inward rectifier and the Na/K-ATPase,
so blocking them depolarizes. Normalizing a depolarized domain needs a blocker of
the depolarizing conductance that holds it, and this library has no published
potency for such a compound on this target.

The gap junction blockers have no effect here because the domain boundary is
already closed by its own transjunctional voltage. Blocking junctions that are
shut changes nothing. Their effect would show on a domain small enough for its
neighbours to repair, which is the regime of Fig. 4 of Cervera et al. (2016).

The depolarizing channel in this model is a phenomenological CAM-type cation
channel, not a named protein, so no real drug targets it by construction. Naming
the protein that carries this current in a given tissue is the step that would
make the screen actionable.

## Sources

| Drug | Target | Half effect | Assay | Species | Source |
|---|---|---|---|---|---|
| chloroquine | Kir2.1 | IC50 8.7 +/- 0.9 uM, Hill 1.0, n = 5 | whole cell, action-potential clamp waveform (Fig. 1D) | human | Rodriguez-Menchaca et al. 2008, PNAS 105:1364-1368, doi:10.1073/pnas.0708153105 |
| ouabain | nak_scale | K0.5 24.6 +/- 6 nM (alpha1-beta1); 102 +/- 14 nM (alpha2-beta1) | purified enzyme expressed in Pichia pastoris, ATPase activity | human (alpha1 isoform) | Lifshitz et al. 2007, Biochemistry 46:14937-14950, doi:10.1021/bi701812c |
| carbenoxolone | gj_scale | IC50 210 uM | parachute dye-transfer assay, hCx43 in HeLa cells, calcein | human (Cx43) | Burnham et al. 2014, Br J Pharmacol 171:4808-4819, doi:10.1111/bph.12554 |
| meclofenamic acid | gj_scale | IC50 264 uM | parachute dye-transfer assay, hCx43 in HeLa cells, calcein | human (Cx43) | Burnham et al. 2014, Br J Pharmacol 171:4808-4819, doi:10.1111/bph.12554 |
| heptanol | gj_scale | IC50 2208 uM | parachute dye-transfer assay, hCx43 in HeLa cells, calcein | human (Cx43) | Burnham et al. 2014, Br J Pharmacol 171:4808-4819, doi:10.1111/bph.12554 |
| digoxin | gj_scale | IC50 0.136 uM (the same paper states 0.15 uM, pIC50 6.8 +/- 0.2, n = 15, in its marketed-drug section) | parachute dye-transfer assay, hCx43 in HeLa cells, calcein | human (Cx43) | Burnham et al. 2014, Br J Pharmacol 171:4808-4819, doi:10.1111/bph.12554 |
| phorbol 12,13-dibutyrate | gj_scale | IC50 0.023 uM | parachute dye-transfer assay, hCx43 in HeLa cells, calcein | human (Cx43) | Burnham et al. 2014, Br J Pharmacol 171:4808-4819, doi:10.1111/bph.12554 |

These are in-silico hypotheses, not medical advice. The potency values come from the assays named above, whose preparation and species differ from any simulated tissue.

## Caveats

**chloroquine**
- Measured in human, whole cell, action-potential clamp waveform (Fig. 1D).
- Also acts on: other Kir channels, hERG. Not modelled here.
- The same paper reports IC50 1.1 +/- 0.2 uM in excised inside-out patches (Fig. 4C). The whole-cell figure is used here because the model represents an intact cell.

**ouabain**
- Measured in human (alpha1 isoform), purified enzyme expressed in Pichia pastoris, ATPase activity.
- The source reports no Hill coefficient. A value of 1 is assumed.
- Strongly species-dependent. Rodent alpha1 is far less sensitive than human alpha1. Do not apply this value to a rodent or amphibian preparation.

**carbenoxolone**
- Measured in human (Cx43), parachute dye-transfer assay, hCx43 in HeLa cells, calcein.
- The source reports no Hill coefficient. A value of 1 is assumed.
- Also acts on: pannexins, Cx26, Cx30. Not modelled here.
- Dye transfer, not electrical coupling. The two are different measurements.

**meclofenamic acid**
- Measured in human (Cx43), parachute dye-transfer assay, hCx43 in HeLa cells, calcein.
- The source reports no Hill coefficient. A value of 1 is assumed.
- Also acts on: cyclooxygenase. Not modelled here.
- Dye transfer, not electrical coupling.

**heptanol**
- Measured in human (Cx43), parachute dye-transfer assay, hCx43 in HeLa cells, calcein.
- The source reports no Hill coefficient. A value of 1 is assumed.
- Also acts on: membrane fluidity, non-specific. Not modelled here.
- Dye transfer, not electrical coupling. A long-chain alkanol acts on the membrane generally, so a clean interpretation is not available.

**digoxin**
- Measured in human (Cx43), parachute dye-transfer assay, hCx43 in HeLa cells, calcein.
- The source reports no Hill coefficient. A value of 1 is assumed.
- Also acts on: Na/K-ATPase (nak_scale), its primary target. Not modelled here.
- Digoxin blocks the Na/K-ATPase as its main action. This entry models only the gap junction effect, so a protocol that uses it is incomplete. The two potencies differ by about 10^3, so at a gap-junction-blocking concentration the pump is blocked too.

**phorbol 12,13-dibutyrate**
- Measured in human (Cx43), parachute dye-transfer assay, hCx43 in HeLa cells, calcein.
- The source reports no Hill coefficient. A value of 1 is assumed.
- Also acts on: protein kinase C, its primary target. Not modelled here.
- Acts through protein kinase C, not directly on the channel, so the effect is indirect and slower than the model's instantaneous block.

