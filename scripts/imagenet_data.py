#!/usr/bin/env python3
"""Build the BREEDS ENTITY-30 mapping and, when the images are present, the split record.

``--hierarchy`` is a directory holding the three BREEDS files ``dataset_class_info.json``,
``class_hierarchy.txt``, and ``node_names.txt``. A copy of those files is retained under
``results/imagenet/hierarchy``, which is enough to rebuild ``mapping.json`` without the
dataset.

Passing ``--data-root`` additionally opens the three splits and writes ``data_audit.json``
and one membership file per split. That needs the indexed training archive
(``scripts/imagenet_archive.py``), the official validation JPEGs under ``validation/``,
and the devkit under ``ILSVRC2012_devkit_t12/``.

Example:

  python scripts/imagenet_data.py --hierarchy results/imagenet/hierarchy \
      --output runs/imagenet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from taskscope.imagenet.data import build_mapping
from taskscope.paths import REPO_ROOT

DEFAULT_HIERARCHY = REPO_ROOT / "results/imagenet/hierarchy"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hierarchy", type=Path, default=DEFAULT_HIERARCHY,
                        help="directory holding the three BREEDS hierarchy files")
    parser.add_argument("--output", type=Path, required=True,
                        help="directory that receives mapping.json and the split record")
    parser.add_argument("--data-root", type=Path,
                        help="ImageNet root, required only for the split record")
    arguments = parser.parse_args()
    mapping = build_mapping(arguments.hierarchy, arguments.output, arguments.data_root)
    print(json.dumps({"coarse_classes": mapping["coarse_classes"],
                      "fine_classes": mapping["fine_classes"],
                      "mapping": str(arguments.output / "mapping.json")}), flush=True)


if __name__ == "__main__":
    main()
