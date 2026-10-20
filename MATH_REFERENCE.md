# Mathematical Reference

All internal math uses SI units: Volts, Amperes, Seconds, Farads, Siemens.

## 1. Physical constants

| Symbol | Value | Unit |
|---|---|---|
| $F$ (Faraday constant) | 96485.332 | C/mol |
| $R$ (gas constant) | 8.31446 | J/(mol K) |
| $T$ (physiological temperature) | 310.15 | K |
| $V_T = RT/F$ (thermal voltage) | 0.026727 | V |

BETSE uses $F = 96485$, $R = 8.314$ (`betse/science/parameters.py`, lines 1106–1107) and
$T = 310\text{ K}$ (`betse/data/yaml/sim_config.yaml`, key `temperature`). This library uses
the CODATA values above.

## 2. Single-cell current balance (Cervera–Alcaraz–Mafe model)

$$C_m \frac{dV_i}{dt} = -I_{pol}(V_i) - I_{dep}(V_i) - I_{leak}(V_i) + \sum_{j \in N(i)} G_{ij}(V_i - V_j) \cdot (V_j - V_i) + I_{ext, i}(t)$$

$I_{ext}$ is positive when it depolarizes the cell (inward positive charge).

### Outward hyperpolarizing current (potassium-like, $E_{pol} = -0.080\text{ V}$)

$$I_{pol}(V) = \frac{G_{pol}^0 (V - E_{pol})}{1 + \exp\left(\frac{z (V - V_{th, pol})}{V_T}\right)}$$

### Inward depolarizing current (sodium/calcium-like, $E_{dep} = 0.0\text{ V}$)

$$I_{dep}(V) = \frac{G_{dep}^0 (V - E_{dep})}{1 + \exp\left(\frac{-z (V - V_{th, dep})}{V_T}\right)}$$

### Leak current

$$I_{leak}(V) = G_{leak} (V - E_{leak})$$

The default is $G_{leak} = 0$. The leak is available for model extensions.

### Fixed points

A fixed point $V^*$ satisfies $I_{pol}(V^*) + I_{dep}(V^*) + I_{leak}(V^*) = 0$.
It is stable when $dI_{ion}/dV > 0$ at $V^*$, and unstable when $dI_{ion}/dV < 0$.

With equal thresholds ($V_{th,pol} = V_{th,dep}$), the ratio $G_{dep}(V)/G_{pol}(V)$ is a
pure exponential in $V$. The cell is then bistable only when its stable points lie
close to $E_{pol}$ and $E_{dep}$. The published CAM parameters (Cervera, Alcaraz & Mafe
2016: $V_{th} = -25$ mV, $z = 3$, $E_{pol} = -60$ mV, $E_{dep} = 0$) give −58.6, −15.5 and
−7.5 mV (`bioelectrics.zoo.cam2016`). Stable points far from the reversal potentials,
such as −70 and −20 mV with $E_{pol} = -80$ mV, need two different thresholds.

### Default `ChannelProfile` (calibrated by bounded Nelder–Mead fit)

These defaults are **not** from a paper. They were fitted so that the stable points are
−70 mV and −20 mV. For published parameters, use `bioelectrics.zoo.cam2016`.

| Parameter | Value |
|---|---|
| $C_m$ | 10 pF ($1.0 \times 10^{-11}$ F) |
| $G_{pol}^0$ | 1.0 nS |
| $G_{dep}^0$ | 1.7645 nS |
| $z$ | 2.1134 |
| $V_{th,pol}$ | −20.96 mV |
| $V_{th,dep}$ | −39.00 mV |
| $G_{leak}$ | 0 |

Resulting fixed points: −70.0 mV (stable, polarized), −45.0 mV (unstable saddle),
−20.0 mV (stable, depolarized). The membrane time constant is about
$C_m / G \approx 10\text{ ms}$.

## 3. Voltage-gated gap junctions

Transjunctional voltage: $V_{ij} = V_i - V_j$.

$$G_{ij}(V_{ij}) = G_{min} + \frac{G_{max} - G_{min}}{1 + \exp\left(\frac{\vert V_{ij}\vert - V_0}{V_{width}}\right)}$$

The conductance is bell-shaped in $V_{ij}$. It is near $G_{max}$ for $\vert V_{ij} \vert \ll V_0$
and near $G_{min}$ for $\vert V_{ij} \vert \gg V_0$.

Default values: $G_{max} = 1.0\text{ nS}$, $G_{min} = 0.02\text{ nS}$,
$V_0 = 0.020\text{ V}$, $V_{width} = 0.004\text{ V}$.

### Comparison with BETSE

BETSE (`betse/science/channels/gap_junction.py`, lines 98–99) uses a double-Boltzmann
open fraction:

$$g_{open} = g_{min} + \frac{1}{1 + e^{(V_{gj} - V_{th})/w}} - \frac{1}{1 + e^{(V_{gj} + V_{th})/w}}$$

with $V_{th} = 15\text{ mV}$, $w = 15\text{ mV}$, $g_{min} = 0.10$
(`sim_config.yaml`, keys `gj voltage threshold`, `gj voltage window`, `gj minimum`).
This library uses the single-Boltzmann form in $\vert V_{ij} \vert$ above. Both forms are
symmetric in $V_{ij}$ and close at high transjunctional voltage.

## 4. Goldman–Hodgkin–Katz (GHK) electrodiffusion

BETSE `electroflux` (`betse/science/sim_toolbox.py`, line 18) computes the molar flux
from outside (A) to inside (B):

$$\alpha = \frac{z F V}{R T}, \qquad
J = -\frac{D \alpha}{d} \cdot \frac{c_{in} - c_{out} e^{-\alpha}}{1 - e^{-\alpha}}$$

The GHK current density for one ion species with permeability $P = D/d$ is

$$I_{GHK} = P z^2 \frac{F^2 V}{R T} \cdot \frac{c_{in} - c_{out} e^{-\alpha}}{1 - e^{-\alpha}}$$

(outward positive). `channels.ghk_current` implements this primitive.

## 5. Numerical integration

Fixed-step RK4 with $dt = 0.5\text{ ms}$ ($\tau / dt \approx 20$). The loop runs over time
steps only. Each step is vectorized over all cells and the batch dimension. The
complete trajectory is differentiable (backpropagation through time).

## 6. Ion-level model (`bioelectrics.ionic`)

State per cell: intracellular concentrations $c_k$ [mol/m³] of $k \in$ {Na⁺, K⁺, Cl⁻,
Ca²⁺, M⁻, P⁻} and the gates $m, h$ of every expressed channel. The bath is fixed.

### Membrane potential from net charge (BETSE `sim.py`, `update_V`)

$$V_{mem} = \frac{F \, \mathrm{vol}}{C_m \, A} \sum_k z_k c_k$$

The profiles are charge-balanced with the anion M⁻, so $V_{mem}(0) = 0$. With
$r = 5\,\mu$m and $C_m = 0.01$ F/m², $F\,\mathrm{vol}/(C_m A) \approx 16$ V per mol/m³:
100 mV is a net charge of only about 6 µM.

### Membrane flux (BETSE `sim_toolbox.electroflux`)

$$J_k = P_k \, \frac{a_k \left(c_k^{out} e^{-a_k} - c_k\right)}{1 - e^{-a_k}}, \qquad
a_k = \frac{z_k F V_{mem}}{R T}, \qquad
P_k = \frac{D_{m,k} + \sum_{ch} P^{ch}_{open} \, \rho^{ch}_k \, D^{ch}_{max}}{d_m}$$

$J_k > 0$ is influx. $P_{open} = m^p h^q$ with $\dot m = (m_\infty(V) - m)/\tau_m(V)$.
$\rho_k$ is the relative permeability of the channel for ion $k$.

### Pumps (BETSE `pumpNaKATP`, `pumpCaATP`)

$$J_{Na} = -3\,\alpha_{NaK}\, \phi_{NaK} \left(1 - \frac{Q}{K_{eq}}\right), \qquad J_K = -\tfrac{2}{3} J_{Na}, \qquad
K_{eq} = \exp\!\left(-\frac{\Delta G_{ATP}}{RT} + \frac{F V_{mem}}{RT}\right)$$

with $\Delta G_{ATP} = -37$ kJ/mol, Michaelis–Menten saturation $\phi_{NaK}$ and reaction
quotient $Q$ as in BETSE. The Ca-ATPase has the same form with charge 2.

### Gap junctions (BETSE `update_gj`, `gap_junction.py`)

Flux into cell $i$ from neighbour $j$ over contact area $A_{ij}$: GHK flux with
$c^{out} = c_j$, $V = V_i - V_j$ and

$$P^{gj}_k = \frac{D^{free}_k \, f_{gj} \, g_{open}(V_i - V_j)}{L_{gj}}, \qquad
g_{open} = g_{min} + \frac{1}{1 + e^{(V - V_{th})/w}} - \frac{1}{1 + e^{(V + V_{th})/w}}$$

Defaults (BETSE): $f_{gj} = 5\times10^{-8}$, $L_{gj} = 26$ nm, $V_{th} = w = 15$ mV,
$g_{min} = 0.1$. Electrical coupling is fast (about 30 s⁻¹ per junction); salt
diffusion through junctions is slow (time constant about 45 min).

### Time scales and solvers

The membrane relaxes at up to about $10^3$ s⁻¹ with strong channel expression, while
concentrations change over minutes. The default solver is adaptive Dormand–Prince
5(4) with a pure absolute tolerance of $10^{-7}$ mol/m³ on concentrations (about
2 µV per step on $V_{mem}$). A linearly implicit ROS2 solver (Verwer et al. 1999)
with a per-cell Jacobian is available as `method="rosenbrock"`.

### Validation (tests/test_ionic_physics.py)

- K⁺-only membrane, no pumps: $V_{mem}$ converges to the Nernst potential (0.2 mV).
- Monovalent ions, no pumps: $V_{mem}$ converges to the GHK voltage equation (0.2 mV).
- Gap junction transport conserves every ion species (relative error $< 10^{-12}$).
- Grid and graph topologies give identical right-hand sides.
