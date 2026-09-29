#!/usr/bin/env python3
"""Fit the shared feature initializer, or one matched task objective starting from it.

Run once with ``--objective feature_initialization`` and no ``--initialization``. Every
reported task fit then starts from that single checkpoint and passes it as
``--initialization``, which is what holds the architecture, the observation, the rate
term, the image order, and the fitting exposure fixed across the two losses.

The reported fits use 5000 updates of 128 images, giving 639,976 image exposures, and
cover seeds 17 and 23 at rate weights 0.03 and 0.3 for each of the two objectives. Fitting
needs a GPU and the indexed ImageNet training archive. CompressAI supplies the entropy
models.

Example:

  python scripts/imagenet_fit.py --objective feature_initialization \
      --data-root /path/to/imagenet --mapping runs/imagenet/mapping.json \
      --output runs/imagenet/feature_initialization_s17_5000 --lr 1e-3

  python scripts/imagenet_fit.py --objective added_fine --seed 17 --beta 0.3 --lr 1e-4 \
      --initialization runs/imagenet/feature_initialization_s17_5000/last.pt \
      --data-root /path/to/imagenet --mapping runs/imagenet/mapping.json \
      --output runs/imagenet/warm_added_fine_b0.3_s17_5000
"""

from __future__ import annotations

import argparse
from pathlib import Path

from taskscope.imagenet.codec import OBJECTIVES, run_fit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--objective", choices=OBJECTIVES, required=True)
    parser.add_argument("--initialization", type=Path,
                        help="shared initializer checkpoint, required by a task fit")
    parser.add_argument("--data-root", type=Path, required=True,
                        help="ImageNet root holding the training index, validation, and devkit")
    parser.add_argument("--mapping", type=Path, required=True, help="ENTITY-30 mapping.json")
    parser.add_argument("--output", type=Path, required=True, help="run directory, must not exist")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--calibration-batches", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="1e-3 for the initializer, 1e-4 for the reported task fits")
    parser.add_argument("--beta", type=float, default=0.003,
                        help="rate weight, 0.003 for the initializer")
    parser.add_argument("--fine-weight", type=float, default=1.0)
    arguments = parser.parse_args()
    if (arguments.objective == "feature_initialization") == (arguments.initialization is not None):
        parser.error("Only matched task fitting requires the shared feature initializer")
    budgets = (arguments.epochs, arguments.max_steps, arguments.batch_size,
               arguments.calibration_batches, arguments.lr, arguments.fine_weight)
    if min(budgets) <= 0 or arguments.beta < 0 or arguments.workers < 0:
        parser.error("Invalid fitting budget")
    run_fit(arguments)


if __name__ == "__main__":
    main()
