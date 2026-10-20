"""Model zoo: published bioelectric models implemented with the library's engine.

Each module documents its source paper, the parameters taken from it, and the
published results that its tests reproduce.

* :mod:`bioelectrics.zoo.cam2016` — Cervera, Alcaraz & Mafe, Sci. Rep. (2016).
"""

from . import cam2016

__all__ = ["cam2016"]
