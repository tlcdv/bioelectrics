"""A library of ion channel and pump drugs with published potency values.

Every :class:`Drug` carries the concentration at which it acts, the assay that
measured it, the species, and the citation. **No value in this module was
estimated, interpolated or inferred.** A drug whose potency is not published on
the target this library models is not listed; see :data:`EXCLUDED` for the ones
that were considered and left out, with the reason.

This supports drug repurposing: given a tissue and a target V_mem pattern,
:class:`~bioelectrics.interventions.InterventionOptimizer` selects which of these
approved or well-characterized compounds, at which concentration, moves the tissue
into the target pattern. Use :func:`candidates_for` to build the candidate set.

Read this before using any result
---------------------------------
These are **in-silico hypotheses, not medical advice and not predictions of a
clinical effect**. Three gaps separate a number here from an experiment:

1. *The assay is not the tissue.* Every value was measured in a heterologous
   expression system (HEK293, HeLa, Xenopus oocytes) or on purified protein, at a
   stated temperature and holding potential. A non-neural tissue in an embryo is a
   different preparation.
2. *The species matters.* Potency can differ by orders of magnitude between
   orthologues. Each entry names the species it was measured in.
3. *Selectivity is not modelled.* A drug is represented by its effect on one
   target. Real compounds hit several. :attr:`Drug.other_targets` records the
   known ones, and :func:`selectivity_warnings` reports them for a protocol.

Concentrations are in **mol/m^3**, which is millimolar, to match the SI
convention of the rest of the library (``DOSE_UNIT``). The sources report
micromolar or nanomolar; the conversion is done once, here, and the original
figure is kept in :attr:`Drug.reported`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .interventions import ChannelBlocker, GapJunctionBlocker, Intervention, PumpBlocker

__all__ = [
    "Drug",
    "LIBRARY",
    "EXCLUDED",
    "DOSE_UNIT",
    "get",
    "for_target",
    "candidates_for",
    "selectivity_warnings",
    "sources_markdown",
]

DOSE_UNIT = "mol/m^3"

_UM = 1e-3  # 1 micromolar in mol/m^3
_NM = 1e-6  # 1 nanomolar in mol/m^3


@dataclass(frozen=True)
class Drug:
    """One compound and one published potency value on one target.

    Parameters
    ----------
    name: the compound.
    target: what it acts on, named exactly as the library names it. For a channel
        this is ``GatedChannel.name`` (for example ``"Kir2.1"``); for a pump or
        the junctions it is the :class:`~bioelectrics.ionic.model.Modifiers` field
        (``"nak_scale"`` or ``"gj_scale"``).
    action: ``"block"`` or ``"open"``.
    half_effect: the concentration giving half the maximum effect [mol/m^3]
        (IC50, EC50 or K0.5, whichever the source reports).
    hill: the Hill coefficient. ``None`` when the source does not report one; the
        dose model then uses 1, and :meth:`caveats` says so.
    reported: the value as the source gives it, verbatim, including the units.
    assay: the preparation and the conditions.
    species: the species of the protein studied.
    source: the citation with a DOI or PMID.
    status: regulatory status, for repurposing ("approved", "withdrawn",
        "research compound", "tool compound").
    other_targets: known off-target actions, for :func:`selectivity_warnings`.
    note: anything a user must know before trusting the entry.
    """

    name: str
    target: str
    action: str
    half_effect: float
    hill: float | None
    reported: str
    assay: str
    species: str
    source: str
    status: str
    other_targets: tuple[str, ...] = ()
    note: str = ""

    @property
    def hill_or_default(self) -> float:
        """The Hill coefficient, or 1.0 when the source does not report one."""
        return 1.0 if self.hill is None else self.hill

    def caveats(self) -> list[str]:
        """Everything that limits this entry, as plain statements."""
        out = [f"Measured in {self.species}, {self.assay}."]
        if self.hill is None:
            out.append("The source reports no Hill coefficient. A value of 1 is assumed.")
        if self.other_targets:
            out.append("Also acts on: " + ", ".join(self.other_targets) + ". Not modelled here.")
        if self.note:
            out.append(self.note)
        return out

    def intervention(self, max_dose: float | None = None, **kwargs) -> Intervention:
        """A dose-parameterized intervention for this drug.

        The target decides the class: a pump drug becomes a
        :class:`~bioelectrics.interventions.PumpBlocker`, a junction drug a
        :class:`~bioelectrics.interventions.GapJunctionBlocker`, anything else a
        :class:`~bioelectrics.interventions.ChannelBlocker`. ``max_dose``
        [mol/m^3] bounds the search; the default is 100 times ``half_effect``,
        which reaches 99 % block at a Hill coefficient of 1.
        """
        if self.action != "block":
            raise NotImplementedError(
                f"{self.name} acts by '{self.action}'. Only 'block' is implemented; "
                "an opener needs a channel whose maximum conductance can be raised."
            )
        dose = dict(
            ic50=self.half_effect,
            hill=self.hill_or_default,
            max_dose=max_dose if max_dose is not None else 100.0 * self.half_effect,
            **kwargs,
        )
        if self.target == "nak_scale":
            return PumpBlocker(**dose)
        if self.target == "gj_scale":
            return GapJunctionBlocker(**dose)
        return ChannelBlocker(self.target, **dose)

    def describe(self) -> str:
        return (f"{self.name} ({self.status}), {self.action}s {self.target}, "
                f"half effect {self.reported} [{self.source}]")


# --------------------------------------------------------------------- library
#
# Each entry was read from the primary source named in it. The `reported` field
# is the figure as that source states it. Do not add an entry without reading
# the source, and do not change a number without re-reading it.

LIBRARY: tuple[Drug, ...] = (
    # --- HCN channels -------------------------------------------------------
    Drug(
        name="ivabradine",
        target="HCN2",
        action="block",
        half_effect=2.25 * _UM,
        hill=None,
        reported="IC50 2.25 uM",
        assay="cloned human HCN1-4, whole cell; steady-state block, no subtype selectivity",
        species="human",
        source="Stieber et al. 2006, Mol Pharmacol 69:1328-1337, doi:10.1124/mol.105.020701",
        status="approved",
        note="The value is the mean over HCN1-4. The source reports no HCN2-specific figure.",
    ),
    Drug(
        name="cilobradine",
        target="HCN2",
        action="block",
        half_effect=0.99 * _UM,
        hill=None,
        reported="IC50 0.99 uM",
        assay="cloned human HCN1-4, whole cell; steady-state block",
        species="human",
        source="Stieber et al. 2006, Mol Pharmacol 69:1328-1337, doi:10.1124/mol.105.020701",
        status="research compound",
        note="The value is the mean over HCN1-4.",
    ),
    Drug(
        name="zatebradine",
        target="HCN2",
        action="block",
        half_effect=1.96 * _UM,
        hill=None,
        reported="IC50 1.96 uM",
        assay="cloned human HCN1-4, whole cell; steady-state block",
        species="human",
        source="Stieber et al. 2006, Mol Pharmacol 69:1328-1337, doi:10.1124/mol.105.020701",
        status="research compound",
        note="The value is the mean over HCN1-4.",
    ),
    # --- Inward rectifier ---------------------------------------------------
    Drug(
        name="chloroquine",
        target="Kir2.1",
        action="block",
        half_effect=8.7 * _UM,
        hill=1.0,
        reported="IC50 8.7 +/- 0.9 uM, Hill 1.0, n = 5",
        assay="whole cell, action-potential clamp waveform (Fig. 1D)",
        species="human",
        source="Rodriguez-Menchaca et al. 2008, PNAS 105:1364-1368, doi:10.1073/pnas.0708153105",
        status="approved",
        other_targets=("other Kir channels", "hERG"),
        note="The same paper reports IC50 1.1 +/- 0.2 uM in excised inside-out patches "
             "(Fig. 4C). The whole-cell figure is used here because the model represents "
             "an intact cell.",
    ),
    # --- Na/K-ATPase --------------------------------------------------------
    Drug(
        name="ouabain",
        target="nak_scale",
        action="block",
        half_effect=24.6 * _NM,
        hill=None,
        reported="K0.5 24.6 +/- 6 nM (alpha1-beta1); 102 +/- 14 nM (alpha2-beta1)",
        assay="purified enzyme expressed in Pichia pastoris, ATPase activity",
        species="human (alpha1 isoform)",
        source="Lifshitz et al. 2007, Biochemistry 46:14937-14950, doi:10.1021/bi701812c",
        status="tool compound",
        note="Strongly species-dependent. Rodent alpha1 is far less sensitive than human "
             "alpha1. Do not apply this value to a rodent or amphibian preparation.",
    ),
    # --- Gap junctions (connexin 43) ---------------------------------------
    #
    # Every value below is from a dye-transfer assay, which measures permeability
    # to calcein. The library models the effect on ELECTRICAL coupling. These are
    # different measurements and the note on each entry says so.
    Drug(
        name="carbenoxolone",
        target="gj_scale",
        action="block",
        half_effect=210.0 * _UM,
        hill=None,
        reported="IC50 210 uM",
        assay="parachute dye-transfer assay, hCx43 in HeLa cells, calcein",
        species="human (Cx43)",
        source="Burnham et al. 2014, Br J Pharmacol 171:4808-4819, doi:10.1111/bph.12554",
        status="approved",
        other_targets=("pannexins", "Cx26", "Cx30"),
        note="Dye transfer, not electrical coupling. The two are different measurements.",
    ),
    Drug(
        name="meclofenamic acid",
        target="gj_scale",
        action="block",
        half_effect=264.0 * _UM,
        hill=None,
        reported="IC50 264 uM",
        assay="parachute dye-transfer assay, hCx43 in HeLa cells, calcein",
        species="human (Cx43)",
        source="Burnham et al. 2014, Br J Pharmacol 171:4808-4819, doi:10.1111/bph.12554",
        status="approved",
        other_targets=("cyclooxygenase",),
        note="Dye transfer, not electrical coupling.",
    ),
    Drug(
        name="heptanol",
        target="gj_scale",
        action="block",
        half_effect=2208.0 * _UM,
        hill=None,
        reported="IC50 2208 uM",
        assay="parachute dye-transfer assay, hCx43 in HeLa cells, calcein",
        species="human (Cx43)",
        source="Burnham et al. 2014, Br J Pharmacol 171:4808-4819, doi:10.1111/bph.12554",
        status="tool compound",
        other_targets=("membrane fluidity, non-specific",),
        note="Dye transfer, not electrical coupling. A long-chain alkanol acts on the "
             "membrane generally, so a clean interpretation is not available.",
    ),
    Drug(
        name="digoxin",
        target="gj_scale",
        action="block",
        half_effect=0.136 * _UM,
        hill=None,
        reported="IC50 0.136 uM (the same paper states 0.15 uM, pIC50 6.8 +/- 0.2, n = 15, "
                 "in its marketed-drug section)",
        assay="parachute dye-transfer assay, hCx43 in HeLa cells, calcein",
        species="human (Cx43)",
        source="Burnham et al. 2014, Br J Pharmacol 171:4808-4819, doi:10.1111/bph.12554",
        status="approved",
        other_targets=("Na/K-ATPase (nak_scale), its primary target",),
        note="Digoxin blocks the Na/K-ATPase as its main action. This entry models only the "
             "gap junction effect, so a protocol that uses it is incomplete. The two "
             "potencies differ by about 10^3, so at a gap-junction-blocking concentration "
             "the pump is blocked too.",
    ),
    Drug(
        name="phorbol 12,13-dibutyrate",
        target="gj_scale",
        action="block",
        half_effect=0.023 * _UM,
        hill=None,
        reported="IC50 0.023 uM",
        assay="parachute dye-transfer assay, hCx43 in HeLa cells, calcein",
        species="human (Cx43)",
        source="Burnham et al. 2014, Br J Pharmacol 171:4808-4819, doi:10.1111/bph.12554",
        status="research compound",
        other_targets=("protein kinase C, its primary target",),
        note="Acts through protein kinase C, not directly on the channel, so the effect is "
             "indirect and slower than the model's instantaneous block.",
    ),
)


EXCLUDED: dict[str, str] = {
    "nifedipine, nitrendipine and other dihydropyridines (Cav1.2)":
        "Dihydropyridine block is state-dependent: the potency depends on the holding "
        "potential and differs by orders of magnitude between a hyperpolarized and a "
        "depolarized cell. A single IC50 is therefore meaningless in a tissue whose cells "
        "sit at different voltages, which is exactly the case this library models. Adding "
        "one would be a number that looks right and is wrong. Modelling the state "
        "dependence explicitly would be needed first.",
    "lamotrigine (HCN2 activator, used by Pai et al. 2020)":
        "No published EC50 on HCN2 was found. The 2020 paper reports organism-level "
        "outcomes (the share of tadpoles with brain defects), not a channel potency, so "
        "there is no number to use.",
    "glibenclamide, pinacidil, diazoxide, minoxidil (KATP)":
        "The KATP channel is not in the channel library yet. Add the channel from a cited "
        "primary model first.",
}


# ------------------------------------------------------------------- accessors
def get(name: str) -> Drug:
    """The drug with this name (case-insensitive)."""
    key = name.strip().lower()
    for d in LIBRARY:  # a short fixed table, not a hot path
        if d.name.lower() == key:
            return d
    raise KeyError(f"no drug named {name!r}. Available: {', '.join(d.name for d in LIBRARY)}")


def for_target(target: str) -> tuple[Drug, ...]:
    """Every drug acting on this target."""
    return tuple(d for d in LIBRARY if d.target == target)


def candidates_for(targets: list[str], status: str | None = None) -> tuple[Drug, ...]:
    """Drugs acting on any of ``targets``, optionally filtered by regulatory status.

    ``status="approved"`` restricts the set to compounds already approved for use
    in people, which is the repurposing question.
    """
    out = [d for d in LIBRARY if d.target in targets]
    if status is not None:
        out = [d for d in out if d.status == status]
    return tuple(out)


def selectivity_warnings(drugs: list[Drug]) -> list[str]:
    """One line per drug whose off-target actions are not modelled."""
    return [f"{d.name}: also acts on {', '.join(d.other_targets)}; not modelled."
            for d in drugs if d.other_targets]


def sources_markdown(drugs: tuple[Drug, ...] | None = None) -> str:
    """A citation table for the drugs used, to print under a protocol."""
    drugs = LIBRARY if drugs is None else drugs
    lines = ["| Drug | Target | Half effect | Assay | Species | Source |",
             "|---|---|---|---|---|---|"]
    for d in drugs:
        lines.append(f"| {d.name} | {d.target} | {d.reported} | {d.assay} | {d.species} | {d.source} |")
    lines += ["", "These are in-silico hypotheses, not medical advice. The potency values come "
                  "from the assays named above, whose preparation and species differ from any "
                  "simulated tissue."]
    return "\n".join(lines)
