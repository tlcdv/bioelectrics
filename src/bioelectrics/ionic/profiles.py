"""Ion species and concentration profiles (BETSE-compatible).

Concentrations are in mol/m^3 (numerically equal to mM). The values of the
``mammal`` and ``amphibian`` profiles are taken from BETSE
(``betse/science/parameters.py``, ion profiles ``MAMMAL`` and ``AMPHIBIAN``).
As in BETSE, an anion ``M`` (z = -1) balances the net charge of each compartment,
so the initial membrane potential is 0 V.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..constants import DTYPE

#: (valence, free diffusion constant [m^2/s]) from BETSE ``sim_config.yaml``.
SPECIES: dict[str, tuple[int, float]] = {
    "Na": (1, 1.33e-9),
    "K": (1, 1.96e-9),
    "Cl": (-1, 2.03e-9),
    "Ca": (2, 1.0e-10),
    "M": (-1, 1.0e-9),
    "P": (-1, 0.0),
}

#: Default membrane diffusion constants [m^2/s] from BETSE ``sim_config.yaml``.
DEFAULT_MEMBRANE_D: dict[str, float] = {
    "Na": 2.0e-18,
    "K": 1.0e-18,
    "Cl": 1.0e-18,
    "Ca": 1.0e-18,
    "M": 1.0e-18,
    "P": 0.0,
}


def _balanced(c: dict[str, float]) -> dict[str, float]:
    """Add the anion ``M`` so that ``sum(z * c) == 0``."""
    q = sum(SPECIES[name][0] * value for name, value in c.items())
    if q < 0:
        raise ValueError("compartment has net negative charge; M- cannot balance it")
    out = dict(c)
    out["M"] = q
    return out


@dataclass
class IonProfile:
    """Intracellular and extracellular concentrations of a set of ions [mol/m^3]."""

    ions: tuple[str, ...]
    c_in: torch.Tensor
    c_out: torch.Tensor

    @classmethod
    def from_dicts(cls, c_in: dict[str, float], c_out: dict[str, float]) -> "IonProfile":
        c_in, c_out = _balanced(c_in), _balanced(c_out)
        ions = tuple(name for name in SPECIES if name in c_in or name in c_out)
        return cls(
            ions,
            torch.tensor([c_in.get(n, 0.0) for n in ions], dtype=DTYPE),
            torch.tensor([c_out.get(n, 0.0) for n in ions], dtype=DTYPE),
        )

    @classmethod
    def mammal(cls) -> "IonProfile":
        """BETSE ``MAMMAL`` profile."""
        return cls.from_dicts(
            {"Na": 12.0, "K": 139.0, "Cl": 4.0, "Ca": 5.0e-5, "P": 135.0},
            {"Na": 145.0, "K": 5.0, "Cl": 115.0, "Ca": 2.0, "P": 10.0},
        )

    @classmethod
    def amphibian(cls) -> "IonProfile":
        """BETSE ``AMPHIBIAN`` profile (low-salt extracellular medium)."""
        return cls.from_dicts(
            {"Na": 8.0, "K": 125.0, "Cl": 20.0, "Ca": 1.0e-3, "P": 100.0},
            {"Na": 14.5, "K": 0.5, "Cl": 10.5, "Ca": 0.2, "P": 0.0},
        )

    @property
    def z(self) -> torch.Tensor:
        return torch.tensor([SPECIES[n][0] for n in self.ions], dtype=DTYPE)

    @property
    def d_free(self) -> torch.Tensor:
        return torch.tensor([SPECIES[n][1] for n in self.ions], dtype=DTYPE)

    def index(self, name: str) -> int:
        return self.ions.index(name)

    def default_membrane_d(self) -> torch.Tensor:
        return torch.tensor([DEFAULT_MEMBRANE_D[n] for n in self.ions], dtype=DTYPE)
