# Contributing

Contributions from bioelectricity researchers, students and developers are welcome.

## Set up

```bash
git clone https://github.com/tlcdv/bioelectrics.git
cd bioelectrics
python -m venv .venv
.venv/bin/pip install -e ".[dev]"      # Windows: .venv\Scripts\pip
.venv/bin/python -m pytest
```

## Rules

The main rules:

1. All internal math is in SI units (V, A, s, F, S, mol/m^3).
2. No Python loops over cells or edges inside a right-hand side. Use vectorized tensor operations
   that work for `(batch, H, W)` grids and `(batch, N_cells)` graphs.
3. Every biophysical parameter must accept a tensor with `requires_grad=True`.
4. Never weaken, delete or mock a numerical test assertion to make it pass. Fix the model,
   the units or the solver instead.

## Adding a published model to the zoo

A zoo model (`src/bioelectrics/zoo/`) must have:

- the full citation and DOI in the module docstring;
- the parameter values from the paper, with the figure or table they come from;
- a test in `tests/zoo/` that reproduces a published result, and a statement of whether the
  reproduction is quantitative or qualitative;
- no code copied from sources with an incompatible license (this project is MIT).

## Adding an ion channel

Give the gating functions, ion selectivity and the experimental reference. If you port a model
from BETSE (BSD-2-Clause), add it to `THIRD_PARTY_NOTICES.md`.

## Reporting a bug

Open an issue with a minimal script, the output you get and the output you expect.
