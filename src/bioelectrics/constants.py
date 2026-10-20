"""Physical constants in strict SI units and physiological unit converters.

All internal math in ``bioelectrics`` uses SI units (V, A, s, F, S). Convert to
millivolts only at the API boundary and in plots.
"""

from __future__ import annotations

import torch

#: Faraday constant [C/mol].
F: float = 96485.332
#: Molar gas constant [J/(mol K)].
R: float = 8.31446
#: Physiological temperature [K].
T: float = 310.15
#: Thermal voltage RT/F at physiological temperature [V] (about 0.0267 V).
V_T: float = R * T / F

#: Default floating point type for all simulations.
DTYPE = torch.float64


def thermal_voltage(temperature: float = T) -> float:
    """Return the thermal voltage ``RT/F`` [V] at ``temperature`` [K]."""
    return R * temperature / F


def mV_to_V(value):
    """Convert millivolts to volts. Works on floats, arrays and tensors."""
    return value * 1e-3


def V_to_mV(value):
    """Convert volts to millivolts. Works on floats, arrays and tensors."""
    return value * 1e3


def pA_to_A(value):
    """Convert picoamperes to amperes."""
    return value * 1e-12


def nS_to_S(value):
    """Convert nanosiemens to siemens."""
    return value * 1e-9


def pF_to_F(value):
    """Convert picofarads to farads."""
    return value * 1e-12
