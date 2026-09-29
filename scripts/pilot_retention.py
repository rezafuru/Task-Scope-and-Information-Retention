#!/usr/bin/env python3
"""Locate the wing regions of the pair photographs and measure the cue colour inside them.

For every development photograph of both species: a wing box centred on the visible wing parts
and a body box centred on breast, back and belly, both 96 by 48 pixels inside the 224 frame. The
percentage of chestnut and of blue pixels inside each box is measured in the photograph and in
the inversion at each exit, and written to ``cue_retention.json``. The figure reads the wing
boxes from that file and enlarges them.
"""

import argparse
import json
from pathlib import Path

from taskscope.pilot import retention
from taskscope.pilot.common import DEFAULT_RUN


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cub-root", type=Path, required=True,
                        help="CUB data root holding CUB_200_2011/parts and the prepared 224 inputs")
    parser.add_argument("--inventory", type=Path,
                        default=DEFAULT_RUN / "data/cub_development/inventory.json")
    parser.add_argument("--inversions", type=Path, default=DEFAULT_RUN / "inversions/development",
                        help="directory holding one inversion inventory per exit")
    parser.add_argument("--output", type=Path, default=DEFAULT_RUN / "pair/cue_retention.json")
    arguments = parser.parse_args()
    records = retention.measure(arguments.output, inventory=arguments.inventory,
                                cub_root=arguments.cub_root, inversions=arguments.inversions)
    print(json.dumps({"written": str(arguments.output), "photographs": len(records),
                      "mean_percent": retention.summary(records)}, indent=1))


if __name__ == "__main__":
    main()
