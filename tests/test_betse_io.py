"""Reading and writing BETSE configurations.

The fixtures are written here rather than copied from BETSE, so the tests state
exactly which keys they depend on.
"""

import pytest
import torch

yaml = pytest.importorskip("yaml")

from bioelectrics.io import betse  # noqa: E402
from bioelectrics.ionic import CellGeometry, IonicTissue  # noqa: E402

BASE = {
    "general options": {"simulate extracellular spaces": False, "ion profile": "mammal"},
    "world options": {"cell radius": 5.0e-6, "cell height": 10.0e-6, "cell spacing": 26.0e-9},
    "tissue profile definition": {"tissue": {"default": {"diffusion constants": {
        "Dm_Na": 2.0e-18, "Dm_K": 1.0e-18, "Dm_Cl": 1.0e-18,
        "Dm_Ca": 1.0e-18, "Dm_M": 1.0e-18, "Dm_P": 0.0}}}},
    "variable settings": {"temperature": 310.0, "gap junctions": {
        "gap junction surface area": 5.0e-8, "voltage sensitive gj": True}},
    "internal parameters": {"alpha_NaK": 1.0e-7, "alpha_Ca": 5.0e-8,
                            "membrane capacitance": 0.05},
}


def write(tmp_path, cfg, name="sim_config.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def deep(cfg: dict) -> dict:
    import copy
    return copy.deepcopy(cfg)


def test_reads_every_shared_parameter(tmp_path):
    tissue = betse.read_config(write(tmp_path, BASE))
    assert tissue.profile.ions == ("Na", "K", "Cl", "Ca", "M", "P")
    assert torch.allclose(tissue.membrane_d,
                          torch.tensor([2e-18, 1e-18, 1e-18, 1e-18, 1e-18, 0.0], dtype=torch.float64))
    assert float(tissue.alpha_nak) == 1.0e-7
    assert float(tissue.alpha_ca) == 5.0e-8
    assert tissue.c_m == 0.05            # BETSE's default, not this library's 0.01
    assert tissue.temperature == 310.0
    assert tissue.gj_length == 26.0e-9
    assert float(tissue.gj_fraction) == 5.0e-8
    assert tissue.gj_gating == "betse"
    assert tissue.geometry.radius == 5.0e-6 and tissue.geometry.height == 10.0e-6


def test_the_loaded_tissue_runs(tmp_path):
    """A configuration that loads must produce a tissue that actually integrates."""
    tissue = betse.read_config(write(tmp_path, BASE))
    state = tissue.initial_state((3, 3))
    out = tissue.simulate(state, 0.05)
    v = tissue.vmem_of(out)
    assert torch.isfinite(v).all()
    assert v.shape == (1, 3, 3)


@pytest.mark.parametrize("path,value,word", [
    (("general options", "simulate extracellular spaces"), True, "extracellular"),
    (("variable settings", "deformation", "turn on"), True, "deformation"),
    (("variable settings", "pressures", "include osmotic pressure"), True, "osmotic"),
    (("gene regulatory network settings", "run gene regulatory network"), True, "gene regulatory"),
    (("world options", "import from svg", "svg override"), True, "SVG"),
])
def test_an_unmodelled_feature_is_refused_and_named(tmp_path, path, value, word):
    """Never substitute a default: a silent swap makes a comparison meaningless."""
    cfg = deep(BASE)
    node = cfg
    for key in path[:-1]:
        node = node.setdefault(key, {})
    node[path[-1]] = value
    with pytest.raises(betse.UnsupportedConfig) as exc:
        betse.read_config(write(tmp_path, cfg))
    assert word in str(exc.value)
    assert " -> ".join(path) in str(exc.value)


def test_an_unknown_ion_profile_is_refused(tmp_path):
    cfg = deep(BASE)
    cfg["general options"]["ion profile"] = "basic"
    with pytest.raises(betse.UnsupportedConfig, match="basic"):
        betse.read_config(write(tmp_path, cfg))


def test_a_custom_profile_is_read(tmp_path):
    cfg = deep(BASE)
    cfg["general options"]["ion profile"] = "custom"
    cfg["general options"]["customized ion profile"] = {
        "extracellular Na+ concentration": 145.0, "cytosolic Na+ concentration": 12.0,
        "extracellular K+ concentration": 5.0, "cytosolic K+ concentration": 139.0,
        "extracellular Cl- concentration": 115.0, "cytosolic Cl- concentration": 4.0,
        "extracellular Ca2+ concentration": 2.0, "cytosolic Ca2+ concentration": 2.0e-5,
        "extracellular M- concentration": 39.0, "cytosolic M- concentration": 12.0,
        "extracellular protein- concentration": 0.0, "cytosolic protein- concentration": 135.0,
    }
    tissue = betse.read_config(write(tmp_path, cfg))
    assert float(tissue.profile.c_in[0]) == 12.0
    assert float(tissue.profile.c_out[1]) == 5.0


def test_a_custom_profile_that_is_not_charge_neutral_is_refused(tmp_path):
    """V_mem follows from net charge, so a profile with net charge starts at tens of volts."""
    cfg = deep(BASE)
    cfg["general options"]["ion profile"] = "custom"
    cfg["general options"]["customized ion profile"] = {
        "extracellular Na+ concentration": 145.0, "cytosolic Na+ concentration": 12.0,
        "extracellular K+ concentration": 5.0, "cytosolic K+ concentration": 139.0,
        "extracellular Cl- concentration": 115.0, "cytosolic Cl- concentration": 4.0,
        "extracellular Ca2+ concentration": 2.0, "cytosolic Ca2+ concentration": 2.0e-5,
        "extracellular M- concentration": 0.0, "cytosolic M- concentration": 0.0,
        "extracellular protein- concentration": 0.0, "cytosolic protein- concentration": 0.0,
    }
    with pytest.raises(betse.UnsupportedConfig, match="net charge"):
        betse.read_config(write(tmp_path, cfg))


def test_a_custom_profile_missing_an_ion_is_refused(tmp_path):
    cfg = deep(BASE)
    cfg["general options"]["ion profile"] = "custom"
    cfg["general options"]["customized ion profile"] = {
        "extracellular Na+ concentration": 145.0, "cytosolic Na+ concentration": 12.0}
    with pytest.raises(betse.UnsupportedConfig, match="K"):
        betse.read_config(write(tmp_path, cfg))


def test_a_missing_diffusion_constant_is_refused(tmp_path):
    """Leaving one out would silently substitute a value from elsewhere."""
    cfg = deep(BASE)
    del cfg["tissue profile definition"]["tissue"]["default"]["diffusion constants"]["Dm_Cl"]
    with pytest.raises(betse.UnsupportedConfig, match="Dm_Cl"):
        betse.read_config(write(tmp_path, cfg))


def test_gating_can_be_switched_off(tmp_path):
    cfg = deep(BASE)
    cfg["variable settings"]["gap junctions"]["voltage sensitive gj"] = False
    assert betse.read_config(write(tmp_path, cfg)).gj_gating is None


def test_round_trip_preserves_the_shared_parameters(tmp_path):
    tissue = betse.read_config(write(tmp_path, BASE))
    out = tmp_path / "written.yaml"
    betse.write_config(tissue, out)
    again = betse.read_config(out)
    assert torch.allclose(tissue.membrane_d, again.membrane_d)
    assert torch.allclose(tissue.profile.c_in, again.profile.c_in)
    assert torch.allclose(tissue.profile.c_out, again.profile.c_out)
    assert (again.c_m, again.temperature, again.gj_length) == (tissue.c_m, tissue.temperature, tissue.gj_length)
    assert abs(again.geometry.area - tissue.geometry.area) < 1e-20
    assert again.gj_gating == tissue.gj_gating


def test_written_config_says_what_it_leaves_out(tmp_path):
    tissue = betse.read_config(write(tmp_path, BASE))
    out = tmp_path / "written.yaml"
    betse.write_config(tissue, out)
    text = out.read_text(encoding="utf-8")
    assert "Partial BETSE configuration" in text
    assert "Channels expressed in the source tissue are NOT written" in text
    assert yaml.safe_load(text)["general options"]["ion profile"] == "custom"


def test_writing_a_tissue_with_no_cell_radius_is_refused(tmp_path):
    """BETSE's world section needs a radius; an area and volume alone cannot give one."""
    tissue = IonicTissue(geometry=CellGeometry(area=4.7e-10, volume=7.8e-16))
    with pytest.raises(betse.UnsupportedConfig, match="radius"):
        betse.write_config(tissue, tmp_path / "x.yaml")


def test_summary_names_what_it_did_not_read(tmp_path):
    text = betse.config_summary(write(tmp_path, BASE))
    assert "Not read" in text and "extracellular grid" in text
    assert "0.05" in text and "310" in text
