# Tutorials

Four notebooks, each runnable in the browser with no installation. They execute in
continuous integration on every push, so a tutorial that stops working fails the build.

| Notebook | What it covers | |
|---|---|---|
| 1. A cell with two stable states | Finding steady states and their stability directly, and the current-voltage curve | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tlcdv/bioelectrics/blob/main/notebooks/01-single-cell-attractors.ipynb) |
| 2. A tissue, and the boundary a domain draws for itself | Voltage-gated gap junctions closing around a depolarized domain | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tlcdv/bioelectrics/blob/main/notebooks/02-tissue-and-gap-junctions.ipynb) |
| 3. Pattern memory, as hysteresis | A dose ramp up and back down, and the two branches it leaves | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tlcdv/bioelectrics/blob/main/notebooks/03-pattern-memory.ipynb) |
| 4. Designing an intervention | Optimizing a chronic dose through the steady state, and the drug library | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tlcdv/bioelectrics/blob/main/notebooks/04-designing-an-intervention.ipynb) |

Running them locally:

```bash
pip install -e ".[notebooks]"
pytest --nbmake notebooks/ -q
```
