#!/usr/bin/env python3
"""Fit the 200-way CUB readers and restrict them to Indigo Bunting against Blue Grosbeak.

Stages, in order:

  readouts  fit the two readers of one observation on the CUB development population
  heldout   restrict every fitted reader to the pair on the official CUB test photographs

An observation is either the native feature map at one exit or the RGB reconstruction of that
exit, eight in all. ``readouts`` fits one observation per invocation and needs the extracted
exits and the exported inversions of the complete development population, all 200 categories.
``heldout`` reads the eight fitted pairs and writes ``evaluation.json`` in the form the figure
script reads, using the test exits and inversions of the two species only.

Every reader runs 50 epochs of AdamW at learning rate and weight decay 1e-4, batch 64, with
class-weighted cross entropy and the epoch chosen by validation macro accuracy then macro cross
entropy. The reported accuracies are the two fitted readers and their arithmetic mean on 30
official test photographs per species.
"""

import argparse
from pathlib import Path

from taskscope.pilot import pair
from taskscope.pilot.common import DEFAULT_RUN, deterministic, resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["readouts", "heldout"])
    parser.add_argument("--observation", choices=pair.OBSERVATIONS,
                        help="observation to fit, required by readouts")
    parser.add_argument("--output", type=Path, help="stage output, default under the run directory")
    parser.add_argument("--development-inventory", type=Path,
                        default=DEFAULT_RUN / "data/cub_development/inventory.json")
    parser.add_argument("--test-inventory", type=Path,
                        default=DEFAULT_RUN / "data/cub_test/inventory.json")
    parser.add_argument("--development-features", type=Path, default=DEFAULT_RUN / "features/development")
    parser.add_argument("--test-features", type=Path, default=DEFAULT_RUN / "features/test")
    parser.add_argument("--development-inversions", type=Path,
                        default=DEFAULT_RUN / "inversions/development",
                        help="directory holding one inversion inventory per exit")
    parser.add_argument("--test-inversions", type=Path, default=DEFAULT_RUN / "inversions/test")
    parser.add_argument("--readouts", type=Path, default=DEFAULT_RUN / "pair/readouts",
                        help="directory holding one fitted observation per subdirectory")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    arguments = parser.parse_args()

    deterministic(True)
    if arguments.stage == "readouts":
        if arguments.observation is None:
            parser.error("readouts requires --observation")
        output = arguments.output or arguments.readouts / arguments.observation
        pair.readouts(output, observation=arguments.observation,
                      inventory=arguments.development_inventory,
                      cache=arguments.development_features,
                      inversions=arguments.development_inversions,
                      device=resolve_device(arguments.device))
    else:
        output = arguments.output or DEFAULT_RUN / "pair/heldout"
        pair.heldout(output, readouts_root=arguments.readouts,
                     test_inventory=arguments.test_inventory, test_cache=arguments.test_features,
                     inversions=arguments.test_inversions, device=resolve_device(arguments.device))


if __name__ == "__main__":
    main()
