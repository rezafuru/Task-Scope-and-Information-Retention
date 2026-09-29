"""ImageNet ENTITY-30: one layer-2 feature code under coarse-only and added-fine losses.

Stages, in the order they run:

  :mod:`taskscope.imagenet.archive`   index the official ILSVRC training tar in place
  :mod:`taskscope.imagenet.data`      BREEDS ENTITY-30 mapping, the split, and the readers
  :mod:`taskscope.imagenet.models`    frozen ResNet-50 observation, feature codec, readouts
  :mod:`taskscope.imagenet.codec`     shared feature initializer and the matched task fits
  :mod:`taskscope.imagenet.assess`    sealed messages, adapted readouts, selection, assessment
  :mod:`taskscope.imagenet.analysis`  the reported validation and held-out comparison

The entropy models come from CompressAI, which is an optional dependency. Importing this
package does not require it, and neither does the analysis. Building or loading a codec does.
"""

__all__ = ["analysis", "archive", "assess", "codec", "data", "models"]
