# Validation

Every number on this page is recomputed by `scripts/build_validation.py` from the
code in this repository. None of it is copied from a previous run or edited by hand.

Published figures are not reproduced here. The check is a comparison of numbers,
which is the part that can be verified, and it avoids copying copyrighted images.

## Cervera, Alcaraz & Mafe (2016)

| Quantity | Reference | Computed | Difference | Within tolerance | Test |
|---|---|---|---|---|---|
| polarized fixed point | -58.6 mV | -58.6 mV | +0.0015 mV | yes | `tests/zoo/test_cam2016.py::test_fig2_single_cell_fixed_points` |
| saddle fixed point | -15.5 mV | -15.54 mV | -0.0375 mV | yes | `tests/zoo/test_cam2016.py::test_fig2_single_cell_fixed_points` |
| depolarized fixed point | -7.5 mV | -7.536 mV | -0.0358 mV | yes | `tests/zoo/test_cam2016.py::test_fig2_single_cell_fixed_points` |

Reference: Cervera, Alcaraz & Mafe, Sci. Rep. 6, 20403 (2016), doi:10.1038/srep20403

## Analytic limits

| Quantity | Reference | Computed | Difference | Within tolerance | Test |
|---|---|---|---|---|---|
| K+-only membrane reaches the Nernst potential | -88.87 mV | -88.87 mV | +7.62e-05 mV | yes | `tests/test_ionic_physics.py::test_nernst_potential_with_only_potassium` |
| a steady state does not move in a 1 s simulation | 0 mV | 0.0009851 mV | +0.000985 mV | yes | `tests/test_attractors.py::test_steady_states_are_fixed_points_of_the_simulation` |

Reference: Nernst equation; definition of a steady state

## Gradients

| Quantity | Reference | Computed | Difference | Within tolerance | Test |
|---|---|---|---|---|---|
| implicit gradient against a central finite difference | 0.001626 | 0.001626 | +7.32e-11 | yes | `tests/test_attractors.py::test_implicit_gradient_matches_finite_difference` |

Reference: implicit function theorem

![Validation](assets/validation.png#only-light)
![Validation](assets/validation_dark.png#only-dark)

## What these checks do and do not establish

They establish that the implementation matches the published values and the analytic
limits it claims to match. They do not establish that the model describes any
particular tissue. The parameters come from the sources named above, measured in
preparations that differ from any tissue a user may have in mind.

The implicit gradient agrees with a central finite difference to a relative error of
4.5e-08, which is the check that the attractor derivatives are correct rather than
merely plausible.

## What is not reproduced, and why

Two models the library would otherwise include cannot be reproduced from the public
record:

- **Manicka & Levin (2019), the BEN model.** The equations in its Fig. 8 are
  dimensionally inconsistent as printed: there is no Faraday constant in the V_mem
  expression, the channel flux sign is reversed, and the pump term is about 10^7 times
  the channel flux. Rather than guess, the model is left out.
- **Pai et al. (2020), the frog brain patterning model.** The equations are in the
  paper, but G_pol, G_dep, E_pol, E_dep and the patch geometry are in the
  supplementary material, which the paper states is available from the corresponding
  author on request.

Both are listed here rather than omitted silently, because an absent reproduction is
information too.
