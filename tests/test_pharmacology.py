"""The drug library: data integrity, dose response, and the three blocker targets."""

import math

import pytest
import torch

from bioelectrics import pharmacology as ph
from bioelectrics.constants import DTYPE
from bioelectrics.interventions import ChannelBlocker, GapJunctionBlocker, Modifiers, PumpBlocker
from bioelectrics.ionic import LIBRARY as CHANNELS


def test_every_entry_carries_a_source_an_assay_and_a_species():
    """No value may enter the library without the record of where it came from."""
    assert len(ph.LIBRARY) > 0
    for d in ph.LIBRARY:
        assert d.source and ("doi:" in d.source or "PMID" in d.source), d.name
        assert d.assay and d.species, d.name
        assert d.reported, d.name
        assert d.action in ("block", "open"), d.name
        assert d.status in ("approved", "withdrawn", "research compound", "tool compound"), d.name
        assert d.half_effect > 0.0, d.name


def test_reported_value_matches_the_stored_concentration():
    """The SI conversion is the only transformation applied to a published number."""
    expected = {  # name: (figure in the source, unit)
        "ivabradine": (2.25, "uM"), "cilobradine": (0.99, "uM"), "zatebradine": (1.96, "uM"),
        "chloroquine": (8.7, "uM"), "ouabain": (24.6, "nM"), "carbenoxolone": (210.0, "uM"),
        "meclofenamic acid": (264.0, "uM"), "heptanol": (2208.0, "uM"),
        "digoxin": (0.136, "uM"), "phorbol 12,13-dibutyrate": (0.023, "uM"),
    }
    assert set(expected) == {d.name for d in ph.LIBRARY}
    for name, (value, unit) in expected.items():
        d = ph.get(name)
        factor = 1e-3 if unit == "uM" else 1e-6  # mol/m^3 is mM
        assert d.half_effect == pytest.approx(value * factor, rel=1e-12)
        assert f"{value:g}" in d.reported


def test_every_target_exists_in_the_library_it_refers_to():
    """A target name that nothing reads would make the drug silently do nothing."""
    channel_names = set(CHANNELS)  # LIBRARY is a dict keyed by channel name
    modifier_fields = {"nak_scale", "gj_scale"}
    for d in ph.LIBRARY:
        assert d.target in channel_names | modifier_fields, f"{d.name} -> {d.target}"


def test_excluded_drugs_are_recorded_with_a_reason():
    assert len(ph.EXCLUDED) >= 3
    for name, reason in ph.EXCLUDED.items():
        assert len(reason) > 80, name  # a real reason, not a label
    joined = " ".join(ph.EXCLUDED).lower()
    assert "dihydropyridine" in joined or "nifedipine" in joined
    assert not any(d.name.lower() in joined.split() for d in ph.LIBRARY)


def test_hill_default_is_one_and_is_declared_in_the_caveats():
    d = ph.get("ivabradine")
    assert d.hill is None and d.hill_or_default == 1.0
    assert any("Hill" in c for c in d.caveats())
    assert ph.get("chloroquine").hill == 1.0


def test_dose_response_is_half_at_the_published_concentration():
    """The Hill equation must give exactly 50 % block at the reported value."""
    for d in ph.LIBRARY:
        iv = d.intervention()
        iv.build(())
        with torch.no_grad():
            iv.u.copy_(torch.logit(torch.tensor((d.half_effect - iv.lo) / (iv.hi - iv.lo), dtype=DTYPE)))
            assert float(iv.blocked_fraction()) == pytest.approx(0.5, abs=1e-9), d.name


def test_dose_response_is_monotone_and_bounded():
    iv = ph.get("chloroquine").intervention()
    iv.build(())
    fractions = []
    for u in (-12.0, -6.0, 0.0, 6.0, 12.0):
        with torch.no_grad():
            iv.u.fill_(u)
            fractions.append(float(iv.blocked_fraction()))
    assert all(0.0 <= f <= 1.0 for f in fractions)
    assert fractions == sorted(fractions)
    # The dose is bounded by max_dose = 100 * ic50, so the reachable range is
    # about 0.06 % to 99 % block, not the full [0, 1].
    assert fractions[0] < 0.01 and fractions[-1] > 0.95


def test_intervention_class_follows_the_target():
    assert isinstance(ph.get("chloroquine").intervention(), ChannelBlocker)
    assert isinstance(ph.get("ouabain").intervention(), PumpBlocker)
    assert isinstance(ph.get("carbenoxolone").intervention(), GapJunctionBlocker)


def test_pump_and_junction_blockers_write_their_own_modifier_field():
    """Regression: a dose on a pump drug must not land in channel_scale."""
    for name, field_name in (("ouabain", "nak_scale"), ("carbenoxolone", "gj_scale")):
        d = ph.get(name)
        iv = d.intervention()
        iv.build(())
        with torch.no_grad():
            iv.u.copy_(torch.logit(torch.tensor((d.half_effect - iv.lo) / (iv.hi - iv.lo), dtype=DTYPE)))
        mods = Modifiers()
        iv.apply(mods)
        assert mods.channel_scale == {} and mods.channel_add == {}
        assert float(getattr(mods, field_name)) == pytest.approx(0.5, abs=1e-9)


def test_dose_is_differentiable():
    """Rule 2: an optimizer must be able to move the dose."""
    iv = ph.get("ivabradine").intervention()
    u = iv.build(())
    iv.blocked_fraction().backward()
    assert u.grad is not None and float(u.grad) > 0.0


def test_an_opener_is_refused_rather_than_silently_treated_as_a_blocker():
    d = ph.Drug(name="x", target="HCN2", action="open", half_effect=1e-3, hill=None,
                reported="EC50 1 uM", assay="a", species="b", source="doi:10/x", status="approved")
    with pytest.raises(NotImplementedError, match="open"):
        d.intervention()


def test_lookup_helpers():
    assert ph.get("OUABAIN").name == "ouabain"
    with pytest.raises(KeyError, match="no drug named"):
        ph.get("aspirin")
    assert len(ph.for_target("gj_scale")) == 5
    approved = ph.candidates_for(["gj_scale", "Kir2.1"], status="approved")
    assert {d.name for d in approved} == {"carbenoxolone", "meclofenamic acid", "digoxin", "chloroquine"}


def test_off_target_actions_are_reported():
    warnings = ph.selectivity_warnings(list(ph.LIBRARY))
    assert any("digoxin" in w and "Na/K-ATPase" in w for w in warnings)
    assert any("chloroquine" in w for w in warnings)


def test_sources_table_states_the_limits():
    md = ph.sources_markdown(ph.for_target("gj_scale"))
    assert md.count("|") > 20 and "carbenoxolone" in md
    assert "not medical advice" in md
    assert "doi:" in md


def test_blocking_a_polarizing_channel_depolarizes_the_cell():
    """The direction of a drug's effect must follow the physics, not the hope.

    Chloroquine blocks Kir2.1, which carries outward K+ current and holds the cell
    polarized. Blocking it must therefore depolarize. This is the finding that makes
    the screen in examples/drug_repurposing.py come out negative; if this assertion
    ever flips, the model changed and the screen's conclusion must be re-read.
    """
    from bioelectrics.attractors import find_steady_state
    from bioelectrics.ionic import CAT_DEP_CAM, KIR2P1, ChannelExpression, IonicTissue

    cell = IonicTissue(edges=torch.zeros(2, 0, dtype=torch.long),
                       channels=[ChannelExpression(KIR2P1, 3e-16), ChannelExpression(CAT_DEP_CAM, 1e-16)])
    start = cell.with_vmem(cell.initial_state((1,)), torch.full((1, 1), -0.078, dtype=DTYPE))
    base = find_steady_state(cell, start, stability=False)

    drug = ph.get("chloroquine")
    blocked = Modifiers(channel_scale={drug.target: torch.tensor(0.1, dtype=DTYPE)})  # 90 % block
    after = find_steady_state(cell, base.state, blocked, stability=False)
    assert float(after.vmem) > float(base.vmem) + 10e-3


def test_blocking_the_sodium_potassium_pump_depolarizes_the_cell():
    """The Na/K-ATPase is electrogenic and polarizing, so blocking it depolarizes."""
    from bioelectrics.attractors import find_steady_state
    from bioelectrics.ionic import CAT_DEP_CAM, K_IR_CAM, ChannelExpression, IonicTissue

    cell = IonicTissue(edges=torch.zeros(2, 0, dtype=torch.long),
                       channels=[ChannelExpression(K_IR_CAM, 1e-16), ChannelExpression(CAT_DEP_CAM, 1e-16)])
    start = cell.with_vmem(cell.initial_state((1,)), torch.full((1, 1), -0.078, dtype=DTYPE))
    base = find_steady_state(cell, start, stability=False)
    after = find_steady_state(cell, base.state, Modifiers(nak_scale=torch.tensor(0.5, dtype=DTYPE)),
                              stability=False)
    assert float(after.vmem) > float(base.vmem) + 10e-3
