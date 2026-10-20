# Third-party notices

## BETSE

The following parts of `bioelectrics` are derived from BETSE, the BioElectric
Tissue Simulation Engine (https://gitlab.com/betse/betse):

- `src/bioelectrics/ionic/transport.py`: GHK electroflux, Na/K-ATPase and Ca-ATPase
  rate laws, voltage-sensitive gap junction gating (`betse/science/sim_toolbox.py`,
  `betse/science/channels/gap_junction.py`).
- `src/bioelectrics/ionic/channel_library.py`: Hodgkin-Huxley gating functions of
  Kv1.5, Kir2.1, HCN2 and Cav1.2 (`betse/science/channels/vg_*.py`).
- `src/bioelectrics/ionic/profiles.py` and `model.py`: ion profiles, diffusion
  constants and default parameters (`betse/science/parameters.py`,
  `betse/data/yaml/sim_config.yaml`).

The equations were re-implemented as vectorized PyTorch code. BETSE license:

```text
Copyright 2014-2025 by Alexis Pietak & Cecil Curry.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
