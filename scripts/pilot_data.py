#!/usr/bin/env python3
"""Prepare the pilot sources and extract the frozen exits.

Stages, in order:

  cub-development  prepare the official CUB training partition and choose the validation split
  cub-test         prepare the official CUB test partition against the development hash audit
  cub-features     extract the complete exits of the unchanged network for chosen photographs

External inputs are required arguments and are never inferred. The CUB data root holds the
extracted ``CUB_200_2011`` tree and the official ``CUB_200_2011.tgz``. The family mapping gives
the 200-to-36 category grouping. The 200-way readers are fitted on every category, so the
development extraction runs with ``--fine-labels`` and no values.
"""

import argparse
from pathlib import Path

from taskscope.pilot import data
from taskscope.pilot.common import DEFAULT_RUN, deterministic, resolve_device

PAIR_LABELS = (13, 53)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["cub-development", "cub-test", "cub-features"])
    parser.add_argument("--cub-root", type=Path, required=True,
                        help="CUB data root holding CUB_200_2011/ and CUB_200_2011.tgz")
    parser.add_argument("--family-mapping", type=Path,
                        help="200-to-36 category mapping, required by cub-development")
    parser.add_argument("--output", type=Path,
                        help="stage output directory, default under the pilot run directory")
    parser.add_argument("--development", type=Path, default=DEFAULT_RUN / "data/cub_development",
                        help="prepared CUB development directory")
    parser.add_argument("--test", type=Path, default=DEFAULT_RUN / "data/cub_test",
                        help="prepared CUB test directory")
    parser.add_argument("--population", choices=("development", "test"), default="development")
    parser.add_argument("--fine-labels", type=int, nargs="*", default=list(PAIR_LABELS),
                        help="CUB fine labels to extract, empty for every category")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    arguments = parser.parse_args()

    if arguments.stage == "cub-development":
        if arguments.family_mapping is None:
            parser.error("cub-development requires --family-mapping")
        output = arguments.output or DEFAULT_RUN / "data/cub_development"
        data.prepare_cub_development(output, cub_root=arguments.cub_root,
                                     mapping=arguments.family_mapping, workers=arguments.workers)
    elif arguments.stage == "cub-test":
        output = arguments.output or DEFAULT_RUN / "data/cub_test"
        data.prepare_cub_test(output, cub_root=arguments.cub_root, development=arguments.development,
                              workers=arguments.workers)
    else:
        deterministic(True)
        source = arguments.development if arguments.population == "development" else arguments.test
        records = data.load_inventory(source / "inventory.json", arguments.population)
        if arguments.fine_labels:
            records = [row for row in records if row["fine_label"] in set(arguments.fine_labels)]
        records.sort(key=lambda row: row["observation_id"])
        output = arguments.output or DEFAULT_RUN / f"features/{arguments.population}"
        data.extract_features(output, records, arguments.cub_root, resolve_device(arguments.device),
                              batch_size=arguments.batch_size)


if __name__ == "__main__":
    main()
