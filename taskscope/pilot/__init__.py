"""Feature preservation across the residual blocks of a frozen ImageNet ResNet-50.

Stages: source preparation (:mod:`taskscope.pilot.data`), RGB inverses of the exits
(:mod:`taskscope.pilot.inversion`), the species-pair readers and their assessment
(:mod:`taskscope.pilot.pair`), and the wing regions (:mod:`taskscope.pilot.retention`). The
frozen network, its exits and the reader families live in :mod:`taskscope.pilot.features`.
"""

__all__ = ["common", "data", "features", "inversion", "pair", "retention"]
