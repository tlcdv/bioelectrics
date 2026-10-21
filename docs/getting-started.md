# Getting started

## Install

```bash
pip install bioelectrics
```

Or, to run the tests and the examples:

```bash
git clone https://github.com/tlcdv/bioelectrics.git
cd bioelectrics
python -m venv .venv
.venv/bin/pip install -e ".[dev]"      # Windows: .venv\Scripts\pip
.venv/bin/python -m pytest             # 89 tests, about 5 minutes on a laptop CPU
```

Python 3.10 or later and PyTorch 2.0 or later. A GPU is not required.

## A cell with two stable states

```python
import torch
from bioelectrics.ionic import IonicTissue, ChannelExpression, K_IR_CAM, CAT_DEP_CAM
from bioelectrics.attractors import find_steady_state

cell = IonicTissue(edges=torch.zeros(2, 0, dtype=torch.long),
                   channels=[ChannelExpression(K_IR_CAM, 1e-16),
                             ChannelExpression(CAT_DEP_CAM, 1e-16)])

starts = cell.with_vmem(cell.initial_state((1,), batch=3),
                        torch.tensor([[-0.078], [-0.003], [-0.037]], dtype=torch.float64))
ss = find_steady_state(cell, starts)
print(1e3 * ss.vmem[:, 0])   # about -77.3, -12.3, -55.3 mV
print(ss.stable)             # [True, True, False]
```

The third start lands on the saddle between the two stable states. `find_steady_state`
reports stability from the eigenvalues of the Jacobian, and `slowest_time_s` gives the time
scale of the slowest approach.

## Design a drug protocol

Which constant dose of a blocker of the depolarizing channel puts every cell of a tissue
into the polarized state at steady state?

```python
import torch
from bioelectrics.ionic import IonicTissue, ChannelExpression, K_IR_CAM, CAT_DEP_CAM
from bioelectrics.interventions import ChannelBlocker, ChronicInterventionOptimizer

tissue = IonicTissue(channels=[ChannelExpression(K_IR_CAM, 1e-16),
                               ChannelExpression(CAT_DEP_CAM, 1e-16)])

v0 = torch.full((1, 6, 6), -0.078, dtype=torch.float64)
v0[:, 2:4, 2:4] = -0.003                       # a depolarized domain
state = tissue.with_vmem(tissue.initial_state((6, 6)), v0)

opt = ChronicInterventionOptimizer(
    tissue, state, torch.full_like(v0, -0.0773),
    [ChannelBlocker("Cat_dep(CAM)", cost=1.0)])
print(opt.optimize(n_iters=40, tol_mV=1.0).protocol.to_markdown())
```

The gradients come from the implicit function theorem, so the cost does not grow with how
long the tissue takes to relax.

## Use a drug whose potency is published

```python
from bioelectrics import pharmacology as ph

drug = ph.get("ivabradine")
print(drug.describe())
for line in drug.caveats():
    print("-", line)

intervention = drug.intervention(cost=1.0)   # the dose becomes the optimized variable
```

Every entry stores the figure as its source prints it, the assay, the species and the
citation. `ph.sources_markdown()` prints the citation table to put under a protocol.
`ph.EXCLUDED` lists compounds that were considered and left out, with the reason.

## Examples

| Script | What it shows |
|---|---|
| `examples/ionic_tumor_normalization.py` | An optimized drug dose and an optimized light pattern, each applied for 0.1 s |
| `examples/attractor_landscape.py` | Hysteresis under a dose ramp, basins of two coupled cells, chronic drug design |
| `examples/drug_repurposing.py` | The drug library screened against a depolarized domain |
| `examples/tumor_reprogramming_demo.py` | The faster voltage-level model with a current stimulus |
