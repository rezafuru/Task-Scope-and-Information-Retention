#!/usr/bin/env python3
"""Seal messages and fit readouts, freeze the validation selection, assess the held-out split.

Modes:

  cache   seal the training and validation splits for one codec fit (or, with no
          ``--checkpoint``, for the uncompressed reference), capture the frozen suffix's
          penultimate features, and fit the two adapted readouts
  freeze  compare every cached candidate against the cached reference and fix, per
          fitting objective, the lowest mean complete rate meeting both allowances
  test    score one selected checkpoint on the held-out split through the frozen route

The reported allowances are three coarse points and five fine points. ``cache`` and
``test`` need a GPU, the ImageNet root, and the codec checkpoints. ``freeze`` reads only
the cached records and the checkpoints they name.

Example:

  python scripts/imagenet_assess.py cache --checkpoint runs/imagenet/warm_added_fine_b0.3_s17_5000/last.pt \
      --data-root /path/to/imagenet --mapping runs/imagenet/mapping.json \
      --output runs/imagenet/coded_warm_added_fine_b0.3_s17_5000

  python scripts/imagenet_assess.py freeze --mapping runs/imagenet/mapping.json \
      --reference runs/imagenet/reference_suffix96 \
      --candidates runs/imagenet/coded_warm_* --output runs/imagenet/selection_frozen
"""

from __future__ import annotations

import argparse
from pathlib import Path

from taskscope.imagenet.assess import READOUT_SEEDS, dispatch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("cache", "freeze", "test"))
    parser.add_argument("--data-root", type=Path, help="ImageNet root, needed by cache and test")
    parser.add_argument("--mapping", type=Path, required=True, help="ENTITY-30 mapping.json")
    parser.add_argument("--output", type=Path, required=True, help="run directory, must not exist")
    parser.add_argument("--checkpoint", type=Path,
                        help="codec checkpoint, omitted for the uncompressed reference")
    parser.add_argument("--reference", type=Path, help="cached uncompressed reference directory")
    parser.add_argument("--candidates", type=Path, nargs="+", help="cached candidate directories")
    parser.add_argument("--selection", type=Path, help="frozen selection.json, required by test")
    parser.add_argument("--coarse-drop", type=float, default=0.03, help="coarse accuracy allowance")
    parser.add_argument("--fine-drop", type=float, default=0.05, help="fine accuracy allowance")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--train-per-class", type=int, default=96,
                        help="training images per fine class behind the adapted readouts")
    parser.add_argument("--readout-seeds", type=int, nargs="+", default=list(READOUT_SEEDS))
    parser.add_argument("--readout-epochs", type=int, default=20)
    parser.add_argument("--readout-batch-size", type=int, default=256)
    parser.add_argument("--readout-lr", type=float, default=1e-3)
    arguments = parser.parse_args()
    if arguments.mode != "freeze" and arguments.data_root is None:
        parser.error("Sealing and assessment require the ImageNet root")
    if arguments.mode == "freeze" and (arguments.reference is None or not arguments.candidates):
        parser.error("Freeze requires the raw reference and all candidate directories")
    if arguments.mode == "test" and arguments.selection is None:
        parser.error("Test requires a frozen selection")
    dispatch(arguments)


if __name__ == "__main__":
    main()
