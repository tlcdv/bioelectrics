"""Read and write BETSE simulation configurations.

BETSE (`<https://gitlab.com/betse/betse>`_, Pietak & Levin) is the established
simulator for bioelectric tissue, and the ion-level physics in this library is
ported from it. This module lets an existing BETSE configuration be loaded here,
so the two tools can be compared on the same parameters rather than on two
hand-transcribed sets.

What is read
------------
The parameters that have an equivalent on both sides: the ion profile and its
concentrations, the per-ion membrane diffusion constants, the pump rates, the
membrane capacitance, the temperature, the cell geometry, and the gap junction
settings including their voltage gating.

What is refused
---------------
A configuration that switches on a feature this library does not model raises
:class:`UnsupportedConfig`, naming the feature. It is never silently replaced by a
default: a comparison between the two tools would then be meaningless, which is
worse than a failure to load.

Those features are extracellular spaces, deformation, osmotic pressure, gene
regulatory and metabolic networks, and cell-shape imports. See
:data:`UNSUPPORTED` for the full list and the key each one is read from.

Not a BETSE substitute
----------------------
Loading a configuration reproduces its *parameters*, not BETSE's results. BETSE
solves on a Voronoi mesh of cells with an extracellular grid; this library solves
on a regular grid or an explicit graph. Compare the two with that in mind.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ..constants import DTYPE
from ..ionic.model import CellGeometry, IonicTissue
from ..ionic.profiles import SPECIES, IonProfile

__all__ = ["UnsupportedConfig", "UNSUPPORTED", "read_config", "config_summary", "write_config"]


class UnsupportedConfig(ValueError):
    """A configuration uses a feature this library does not model."""


#: BETSE settings that change the physics and have no equivalent here. Each entry
#: maps the path of keys in the YAML to the value that means "switched on", and the
#: explanation given to the user.
UNSUPPORTED: dict[tuple[str, ...], tuple[Any, str]] = {
    ("general options", "simulate extracellular spaces"): (
        True, "extracellular spaces. This library treats the outside as a fixed bath, with no "
              "extracellular fields or diffusion."),
    ("variable settings", "deformation", "turn on"): (
        True, "cell deformation. Cell geometry here is fixed."),
    ("variable settings", "pressures", "include osmotic pressure"): (
        True, "osmotic pressure. There is no cell volume regulation here."),
    ("gene regulatory network settings", "run gene regulatory network"): (
        True, "a gene regulatory network. Voltage does not feed back onto expression here."),
    ("world options", "import from svg", "svg override"): (
        True, "cell positions imported from an SVG file. Geometry here is a regular grid or an "
              "explicit graph."),
}

#: BETSE ion profile names that map onto an :class:`IonProfile` here.
_PROFILES = {"mammal": IonProfile.mammal, "amphibian": IonProfile.amphibian}

_ION_KEYS = {"Na": "Na", "K": "K", "Cl": "Cl", "Ca": "Ca", "M": "M", "P": "P"}

#: How BETSE spells each ion in its "customized ion profile" block. Chloride is an
#: anion, so it is "Cl-"; deriving these by appending "+" produced "Cl+" and silently
#: failed to find the key.
_LABELS = {"Na": "Na+", "K": "K+", "Cl": "Cl-", "Ca": "Ca2+", "M": "M-", "P": "protein-"}


def _dig(cfg: dict, *keys: str, default: Any = None) -> Any:
    """``cfg[keys[0]][keys[1]]...``, or ``default`` if any level is missing."""
    node: Any = cfg
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node


def _check_supported(cfg: dict) -> None:
    for path, (switched_on, reason) in UNSUPPORTED.items():
        if _dig(cfg, *path) == switched_on:
            raise UnsupportedConfig(
                f"this configuration switches on {reason}\n"
                f"  key: {' -> '.join(path)} = {switched_on}\n"
                "Set it to False to load the rest, and state the difference when you compare "
                "the two tools."
            )


def read_config(path: str | Path) -> IonicTissue:
    """Build an :class:`~bioelectrics.ionic.IonicTissue` from a BETSE ``sim_config.yaml``.

    Parameters
    ----------
    path: the configuration file.

    Returns
    -------
    A tissue carrying the configuration's parameters. The cell layout is not read;
    give the returned tissue a grid shape when you build its state, or set
    ``tissue.edges`` for a graph.

    Raises
    ------
    UnsupportedConfig
        The configuration uses a feature this library does not model. The message
        names the feature and the key it came from.
    """
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("reading a BETSE configuration needs PyYAML: pip install pyyaml") from exc

    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"{path} does not contain a YAML mapping")
    _check_supported(cfg)

    profile = _read_profile(cfg)
    membrane_d = _read_membrane_d(cfg, profile)
    geometry = CellGeometry.cylinder(
        radius=float(_dig(cfg, "world options", "cell radius", default=5.0e-6)),
        height=float(_dig(cfg, "world options", "cell height", default=10.0e-6)),
    )
    gj = _dig(cfg, "variable settings", "gap junctions", default={}) or {}
    gated = bool(gj.get("voltage sensitive gj", True))
    internal = cfg.get("internal parameters", {}) or {}

    return IonicTissue(
        profile=profile,
        geometry=geometry,
        membrane_d=membrane_d,
        alpha_nak=float(internal.get("alpha_NaK", 1.0e-7)),
        alpha_ca=float(internal.get("alpha_Ca", 5.0e-8)),
        c_m=float(internal.get("membrane capacitance", 0.05)),
        temperature=float(_dig(cfg, "variable settings", "temperature", default=310.0)),
        gj_fraction=float(gj.get("gap junction surface area", 5.0e-8)),
        gj_length=float(_dig(cfg, "world options", "cell spacing", default=26.0e-9)),
        gj_gating="betse" if gated else None,
    )


def _read_profile(cfg: dict) -> IonProfile:
    """The ion profile. ``custom`` reads the explicit concentrations."""
    name = str(_dig(cfg, "general options", "ion profile", default="mammal")).strip()
    if name in _PROFILES:
        return _PROFILES[name]()
    if name != "custom":
        raise UnsupportedConfig(
            f"ion profile {name!r} has no equivalent here. Supported: "
            f"{', '.join(sorted(_PROFILES))}, custom.\n"
            "The 'basic' profiles omit Cl- and Ca2+, which this library's transport terms "
            "assume are present."
        )

    block = _dig(cfg, "general options", "customized ion profile", default={}) or {}
    ions, c_in, c_out = [], [], []
    for ion in ("Na", "K", "Cl", "Ca", "M", "P"):
        label = _LABELS[ion]
        inside = block.get(f"cytosolic {label} concentration")
        outside = block.get(f"extracellular {label} concentration")
        if inside is None or outside is None:
            raise UnsupportedConfig(
                f"the custom ion profile does not give both concentrations for {label}. "
                "Every one of Na+, K+, Cl-, Ca2+, M- and protein- is needed here."
            )
        ions.append(ion)
        c_in.append(float(inside))   # BETSE states mmol/L, which is mol/m^3
        c_out.append(float(outside))
    profile = IonProfile(ions=tuple(ions),
                         c_in=torch.tensor(c_in, dtype=DTYPE),
                         c_out=torch.tensor(c_out, dtype=DTYPE))
    _check_charge_neutral(profile)
    return profile


def _check_charge_neutral(profile: IonProfile) -> None:
    """BETSE requires each side to sum to zero charge; so does the voltage here."""
    for side, c in (("cytosolic", profile.c_in), ("extracellular", profile.c_out)):
        net = float((profile.z * c).sum())
        scale = float(c.abs().sum())
        if abs(net) > 1e-6 * max(scale, 1.0):
            raise UnsupportedConfig(
                f"the {side} concentrations carry a net charge of {net:.4g} mol/m^3. "
                "V_mem here follows from net charge, so a profile that is not neutral starts "
                "the cell at a voltage of tens of volts. Balance it with the M- anion."
            )


def _read_membrane_d(cfg: dict, profile: IonProfile) -> torch.Tensor:
    """Per-ion membrane diffusion constants from the default tissue profile."""
    block = _dig(cfg, "tissue profile definition", "tissue", "default", "diffusion constants",
                 default=None)
    if block is None:
        return profile.default_membrane_d()
    values = []
    for ion in profile.ions:
        key = f"Dm_{_ION_KEYS[ion]}"
        if key not in block:
            raise UnsupportedConfig(
                f"the default tissue profile gives no {key}. Every ion in the profile needs one, "
                "because leaving it out would silently substitute a value from elsewhere."
            )
        values.append(float(block[key]))
    return torch.tensor(values, dtype=DTYPE)


def config_summary(path: str | Path) -> str:
    """A markdown table of what was read, what was ignored and what has no equivalent.

    Print this beside any comparison with BETSE, so the reader can see which
    parameters were shared and which were not.
    """
    import yaml

    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    tissue = read_config(path)
    rows = [
        ("ion profile", str(_dig(cfg, "general options", "ion profile", default="mammal"))),
        ("ions", ", ".join(tissue.profile.ions)),
        ("membrane diffusion constants [m^2/s]",
         ", ".join(f"{i}={float(d):.3g}" for i, d in zip(tissue.profile.ions, tissue.membrane_d))),
        ("Na/K-ATPase rate [m/(mol s)]", f"{float(tissue.alpha_nak):.3g}"),
        ("Ca-ATPase rate [m/(mol s)]", f"{float(tissue.alpha_ca):.3g}"),
        ("membrane capacitance [F/m^2]", f"{tissue.c_m:.3g}"),
        ("temperature [K]", f"{tissue.temperature:.4g}"),
        ("cell radius, height [m]", _shape(tissue.geometry)),
        ("cell area, volume [m^2, m^3]",
         f"{tissue.geometry.area:.3g}, {tissue.geometry.volume:.3g}"),
        ("gap junction area fraction", f"{float(tissue.gj_fraction):.3g}"),
        ("gap junction length [m]", f"{tissue.gj_length:.3g}"),
        ("gap junction voltage gating", "on" if tissue.gj_gating else "off"),
    ]
    lines = ["| Parameter | Value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in rows]
    lines += [
        "",
        "Not read, because this library has no equivalent: cell layout and mesh, "
        "tissue profiles other than the default, interventions scheduled in the configuration, "
        "plotting and export settings.",
        "",
        "Loading a configuration reproduces its parameters, not BETSE's results. BETSE solves "
        "on a Voronoi mesh with an extracellular grid; this library solves on a regular grid or "
        "an explicit graph.",
    ]
    return "\n".join(lines)


def write_config(tissue: IonicTissue, path: str | Path) -> None:
    """Write the parameters of ``tissue`` as a partial BETSE configuration.

    The result holds only the keys this library can set. It is a starting point for
    a BETSE run, not a complete configuration: BETSE needs the world, tissue and
    export sections too. Merge it into a full `sim_config.yaml`.
    """
    import yaml

    ions = tissue.profile.ions
    out = {
        "general options": {
            "simulate extracellular spaces": False,
            "ion profile": "custom",
            "customized ion profile": _concentration_block(tissue.profile),
        },
        "world options": _world_options(tissue),
        "tissue profile definition": {
            "tissue": {"default": {"diffusion constants": {
                f"Dm_{_ION_KEYS[i]}": float(d) for i, d in zip(ions, tissue.membrane_d)}}},
        },
        "variable settings": {
            "temperature": float(tissue.temperature),
            "gap junctions": {
                "gap junction surface area": float(tissue.gj_fraction),
                "voltage sensitive gj": tissue.gj_gating is not None,
            },
        },
        "internal parameters": {
            "alpha_NaK": float(tissue.alpha_nak),
            "alpha_Ca": float(tissue.alpha_ca),
            "membrane capacitance": float(tissue.c_m),
        },
    }
    header = (
        "# Partial BETSE configuration written by bioelectrics.io.betse.write_config.\n"
        "# It carries only the parameters both tools share. Merge it into a full\n"
        "# sim_config.yaml; BETSE also needs its world, tissue and export sections.\n"
        "# Channels expressed in the source tissue are NOT written: BETSE declares them\n"
        "# in its own tissue profile format, which has no one-to-one mapping here.\n"
    )
    Path(path).write_text(header + yaml.safe_dump(out, sort_keys=False), encoding="utf-8")


def _shape(geometry: CellGeometry) -> str:
    if geometry.radius is None:
        return "not stated; the area and volume were given directly"
    height = "-" if geometry.height is None else f"{geometry.height:.3g}"
    return f"{geometry.radius:.3g}, {height}"


def _world_options(tissue: IonicTissue) -> dict[str, float]:
    """``cell spacing`` always; the radius and height only if the shape is known."""
    out: dict[str, float] = {"cell spacing": float(tissue.gj_length)}
    g = tissue.geometry
    if g.radius is None:
        raise UnsupportedConfig(
            "this tissue's geometry was given as an area and a volume, so there is no cell "
            "radius to write. BETSE's world section needs a radius and a height. Build the "
            "tissue with CellGeometry.cylinder(...) or .sphere(...) to keep them."
        )
    out["cell radius"] = float(g.radius)
    if g.height is not None:
        out["cell height"] = float(g.height)
    return out


def _concentration_block(profile: IonProfile) -> dict[str, float]:
    block = {}
    for ion, inside, outside in zip(profile.ions, profile.c_in, profile.c_out):
        label = _LABELS[ion]
        block[f"cytosolic {label} concentration"] = float(inside)
        block[f"extracellular {label} concentration"] = float(outside)
    return block
