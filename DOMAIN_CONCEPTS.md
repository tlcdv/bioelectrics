# Domain Concepts

These rules are distilled from Levin & Martyniuk (2018), *The Bioelectric Code: An
Ancient Computational Medium for Dynamic Control of Growth and Form*, and Levin (2019),
*The Computational Boundary of a "Self": Developmental Bioelectricity Drives
Multicellularity and Scale-Free Cognition*. Use this exact terminology in docstrings,
class names, and the README.

## 1. Hardware vs. Software Independence

The genome specifies the **"hardware"**: ion channel and gap junction protein expression.
The real-time electrochemical dynamics ($V_{mem}$ distributions) form the **"software"**.
The software stores pattern memory and can be rewritten without altering DNA.

In code:

- `ChannelProfile` holds the fixed hardware parameters ($G_{pol}^0$, $G_{dep}^0$,
  thresholds, gap junction $G_{max}$, $G_{min}$).
- `BioelectricState` holds the dynamic $V_{mem}$ state tensor (the software).

## 2. Closed-Loop Pattern Homeostasis & Target Morphology

Morphogenesis acts as an error-minimization circuit. A multicellular network stores a
**Target Morphology** (a spatial $V_{mem}$ prepattern) as a dynamical attractor, like a
memory in a Hopfield network.

In code, `PatternOptimizer` in `control.py` is an error-minimizing homeostatic
controller. It moves a tissue from a corrupted basin of attraction into the target
morphogenetic attractor.

## 3. Subroutine Calls & Modular Triggering

A bioelectric intervention is a brief trigger. It flips a tissue circuit into a stable
attractor that persists after the stimulus ends.

Every inverse control simulation MUST split time into two phases:

| Phase | Interval | External stimulus |
|---|---|---|
| Active intervention | $[0, T_{stim}]$ | $I_{ext}(t, x, y)$ optimized |
| Relaxation | $[T_{stim}, T_{relax}]$ | zero |

The relaxation phase proves that the intervention triggers a permanent attractor
switch. It does not clamp cells by force.

## 4. Computational Boundary of the Self & Cancer Normalization

Voltage-gated gap junctions merge cells into an isopotential collective.

Oncogenic depolarization (about $-15\text{ mV}$, versus healthy about $-70\text{ mV}$)
creates a high transjunctional voltage $\vert V_i - V_j \vert$. This closes the
surrounding gap junctions ($G_{ij} \to G_{min}$) and electrically isolates the tumor.

Re-polarizing the tumor cells re-opens the gap junctions and normalizes the tissue.
`examples/tumor_reprogramming_demo.py` uses this mechanism.
