#!/usr/bin/env python3
"""Fetch the official ImageNet training tar and index its JPEG byte ranges.

Two commands:

  download  write the official ILSVRC-2012 training archive, verified against its size
            and MD5, using parallel HTTP byte ranges (needs `pip install requests` and an
            ImageNet account whose terms cover the download)
  index     record the offset and length of every training image inside that archive

Indexing writes ``train_index.npz`` and ``train_index.json`` beside each other. The
dataset reader takes both and reads images straight out of the archive, so no extracted
copy of ImageNet is created. The archive is 147.9 GB.

Example:

  python scripts/imagenet_archive.py index --archive /path/to/ILSVRC2012_img_train.tar \
      --index /path/to/imagenet/train_index.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

from taskscope.imagenet import archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("download", "index"))
    parser.add_argument("--archive", type=Path, required=True,
                        help="path of the official ILSVRC2012_img_train.tar")
    parser.add_argument("--workers", type=int, default=8, help="parallel download ranges")
    parser.add_argument("--index", type=Path, help="index written by the index command")
    arguments = parser.parse_args()
    if not 1 <= arguments.workers <= 16:
        parser.error("Require between 1 and 16 download workers")
    if arguments.command == "download":
        archive.download(arguments.archive, arguments.workers)
    elif arguments.index is None:
        parser.error("Index output is required")
    else:
        archive.index_archive(arguments.archive, arguments.index)


if __name__ == "__main__":
    main()
