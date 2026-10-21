# Physics

The equations, their parameters and the source of every number are in
[`MATH_REFERENCE.md`](https://github.com/tlcdv/bioelectrics/blob/main/MATH_REFERENCE.md).
This page states what the model is, so you can judge whether it fits your question.

All internal arithmetic is in SI units: volts, amperes, seconds, farads, siemens, mol/m³,
and m²/s for membrane diffusion constants. Millivolts appear only at the API boundary and in
plots.

## The ion-level model

Each cell carries the intracellular concentration of Na⁺, K⁺, Cl⁻, Ca²⁺ and two impermeant
species, plus two gate variables per expressed channel. The membrane potential follows from
net charge,

$$ V_{mem} = \frac{F\,\mathrm{vol}}{C_m A} \sum_k z_k c_k, $$

which is about 16 V per mol/m³, so voltage is a small difference of large numbers. That is
why the solvers use a pure absolute tolerance on concentrations.

Fluxes across the membrane use the Goldman-Hodgkin-Katz electrodiffusion expression. The
Na/K-ATPase and the Ca-ATPase are thermodynamically constrained pumps. Gap junctions carry
ions between neighbours and are gated by the transjunctional voltage, so a large voltage
difference closes them and a domain isolates itself.

These equations are ported from BETSE (BSD-2-Clause); the file and function each one came
from is named in `MATH_REFERENCE.md`, and the notice is in `THIRD_PARTY_NOTICES.md`.

## Channels

`bioelectrics.ionic.LIBRARY` holds Kv1.5, Kir2.1, HCN2 and Cav1.2 ported from BETSE with
their original experimental references, leak channels, and two phenomenological CAM-type
channels used to produce bistability.

## Steady states and their derivatives

A bioelectric pattern memory is a stable steady state. `find_steady_state` solves
$f(x) = 0$ by pseudo-transient continuation, using an exact sparse Jacobian built with
forward-mode autograd and a distance-2 colouring of the cell graph. On a grid that costs
$5n$ Jacobian-vector products regardless of tissue size.

`steady_state_implicit` differentiates the result with the implicit function theorem, so a
gradient costs one sparse linear solve rather than backpropagation through the whole
relaxation.

## Solvers

| Method | Use |
|---|---|
| `dopri5` | the default, adaptive, safe with strong channel expression |
| `rosenbrock` | linearly implicit ROS2, for stiff problems |
| `rk4`, `euler` | fixed step; they raise an error rather than return NaN |

## Model zoo

`bioelectrics.zoo.cam2016` implements Cervera, Alcaraz & Mafe (2016) with reproduction tests
that assert the published fixed points and the behaviour of its figures 4 and 5. See
[Validation](validation.md) for the numbers, and for the two models that cannot be
reproduced from the public record.
