#!/usr/bin/env python3
"""Fit the RGB inverses on ImageNet and apply them to the prepared CUB photographs.

Stages, in order:

  fit      fit one inverse per exit on ImageNet training photographs
  sources  write the explicit source list an export reads
  apply    invert the listed photographs and export 8-bit PNGs
  convert  bind the exported PNGs back to their CUB records

The fit needs the official ImageNet training archive through a byte-offset index, plus the
complete official validation directory. Both are required arguments. ``fit`` runs 100000 updates
of 64 photographs per exit and needs a CUDA device with bfloat16 support.
"""

import argparse
from pathlib import Path

from taskscope.pilot import data, inversion
from taskscope.pilot.common import file_hash, read_json, resolve_device, write_json
from taskscope.pilot.features import EXITS

PAIR_LABELS = (13, 53)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["fit", "sources", "apply", "convert"])
    parser.add_argument("--view", choices=EXITS, help="exit the inverse reads, required by fit")
    parser.add_argument("--imagenet-train-index", type=Path,
                        help="byte-offset index of the official ImageNet training archive")
    parser.add_argument("--imagenet-validation-root", type=Path,
                        help="directory holding the 50,000 official validation JPEGs")
    parser.add_argument("--cub-root", type=Path, help="CUB data root, required by sources and convert")
    parser.add_argument("--inventory", type=Path, help="prepared CUB inventory, required by sources")
    parser.add_argument("--population", choices=("development", "test"), default="development")
    parser.add_argument("--fine-labels", type=int, nargs="*", default=list(PAIR_LABELS),
                        help="CUB fine labels to export, empty for every category")
    parser.add_argument("--sources", type=Path, help="source list written by the sources stage")
    parser.add_argument("--run", type=Path, help="fitted inverse directory, required by apply")
    parser.add_argument("--preview", type=Path, help="preview.json written by apply")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="fitting batch, and export batch for apply")
    parser.add_argument("--iterations", type=int, default=100000)
    parser.add_argument("--validation-every", type=int, default=2500)
    parser.add_argument("--validation-count", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--memory-format", choices=("contiguous", "channels_last"),
                        default="channels_last")
    parser.add_argument("--deterministic-training", action="store_true",
                        help="the retained fits used cudnn benchmark instead")
    arguments = parser.parse_args()

    if arguments.stage == "fit":
        if not (arguments.view and arguments.imagenet_train_index
                and arguments.imagenet_validation_root):
            parser.error("fit requires --view, --imagenet-train-index and --imagenet-validation-root")
        inversion.fit(arguments.output, view=arguments.view,
                      train_index=arguments.imagenet_train_index,
                      validation_root=arguments.imagenet_validation_root,
                      device=resolve_device(arguments.device), iterations=arguments.iterations,
                      batch_size=arguments.batch_size, validation_every=arguments.validation_every,
                      validation_count=arguments.validation_count, workers=arguments.workers,
                      memory_format=arguments.memory_format,
                      fast_training=not arguments.deterministic_training)
    elif arguments.stage == "sources":
        if not (arguments.cub_root and arguments.inventory):
            parser.error("sources requires --cub-root and --inventory")
        records = data.load_inventory(arguments.inventory, arguments.population)
        if arguments.fine_labels:
            records = [row for row in records if row["fine_label"] in set(arguments.fine_labels)]
        records.sort(key=lambda row: row["observation_id"])
        rows = data.source_rows(records, arguments.cub_root)
        for row, source in zip(records, rows):
            data.check_png(Path(source["path"]), source["sha256"], row["pixel_sha256"])
        write_json(arguments.output, rows)
        write_json(arguments.output.with_suffix(".metadata.json"),
                   {"population": arguments.population, "count": len(rows),
                    "fine_labels": sorted(set(arguments.fine_labels)) if arguments.fine_labels else None,
                    "inventory": str(Path(arguments.inventory).resolve()),
                    "inventory_sha256": file_hash(arguments.inventory),
                    "source_list_sha256": file_hash(arguments.output)})
    elif arguments.stage == "apply":
        if not (arguments.run and arguments.sources):
            parser.error("apply requires --run and --sources")
        inversion.apply_to_sources(arguments.output, run_directory=arguments.run,
                                   sources=arguments.sources, device=resolve_device(arguments.device),
                                   batch_size=arguments.batch_size)
    else:
        if not (arguments.preview and arguments.sources and arguments.cub_root):
            parser.error("convert requires --preview, --sources and --cub-root")
        metadata = read_json(Path(arguments.sources).with_suffix(".metadata.json"))
        if file_hash(arguments.sources) != metadata["source_list_sha256"]:
            raise ValueError("Source list changed after it was written")
        inventory = Path(metadata["inventory"])
        records = data.load_inventory(inventory, metadata["population"],
                                      metadata["inventory_sha256"])
        if metadata["fine_labels"]:
            records = [row for row in records if row["fine_label"] in set(metadata["fine_labels"])]
        records.sort(key=lambda row: row["observation_id"])
        inversion.convert_export(arguments.output, preview=arguments.preview, records=records,
                                 sources=read_json(arguments.sources),
                                 population=metadata["population"], cub_root=arguments.cub_root,
                                 inventory=inventory)


if __name__ == "__main__":
    main()
