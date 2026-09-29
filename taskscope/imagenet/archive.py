"""Fetch the official ImageNet training tar and index its JPEG byte ranges.

The index records the absolute offset and length of every training image inside the
verified archive, so :class:`taskscope.imagenet.data.ImageNetImages` reads images with
``pread`` and no extracted copy of the dataset is written. Both stages refuse to proceed
unless the archive matches the official size and MD5.

The download uses ``requests``, imported lazily so that indexing an archive obtained by
other means needs no extra package.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .data import ARCHIVE_BYTES, ARCHIVE_CLASSES, ARCHIVE_IMAGES, ARCHIVE_MD5, ARCHIVE_URL

_REQUESTS_MISSING = ("Downloading the ImageNet training archive needs the requests package "
                     "(pip install requests). Indexing an archive you already hold does not.")

IMAGE_SUFFIXES = (".jpeg", ".jpg", ".png")


def _require_requests():
    try:
        return importlib.import_module("requests")
    except ImportError as error:
        raise ImportError(_REQUESTS_MISSING) from error


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "md5").hexdigest()


def download(path: Path, workers: int) -> None:
    """Write disjoint HTTP ranges into one file and verify the full archive."""
    requests = _require_requests()
    if path.exists():
        if path.stat().st_size != ARCHIVE_BYTES or digest(path) != ARCHIVE_MD5:
            raise ValueError("Existing archive fails official size or MD5")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    descriptor = os.open(partial, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o644)
    started = time.monotonic()
    try:
        os.posix_fallocate(descriptor, 0, ARCHIVE_BYTES)

        def fetch(worker: int) -> None:
            first = ARCHIVE_BYTES * worker // workers
            last = ARCHIVE_BYTES * (worker + 1) // workers - 1
            position = first
            with requests.Session() as session:
                for attempt in range(3):
                    try:
                        with session.get(ARCHIVE_URL,
                                         headers={"Range": f"bytes={position}-{last}",
                                                  "Accept-Encoding": "identity"},
                                         stream=True, timeout=(30, 120)) as response:
                            expected = f"bytes {position}-{last}/{ARCHIVE_BYTES}"
                            if (response.status_code != 206
                                    or response.headers.get("Content-Range") != expected):
                                raise ValueError("Server did not honor the exact requested byte range")
                            for block in response.iter_content(1024 * 1024):
                                if position + len(block) > last + 1:
                                    raise ValueError("Response exceeds requested range")
                                view = memoryview(block)
                                while view:
                                    written = os.pwrite(descriptor, view, position)
                                    if written <= 0:
                                        raise OSError("Archive write did not advance")
                                    position += written
                                    view = view[written:]
                        if position != last + 1:
                            raise requests.ConnectionError("Incomplete byte range")
                        print(json.dumps({"range": worker, "bytes": position - first,
                                          "seconds": time.monotonic() - started}), flush=True)
                        return
                    except requests.RequestException:
                        if attempt == 2:
                            raise
                        time.sleep(2)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(fetch, range(workers)))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    actual = digest(partial)
    if actual != ARCHIVE_MD5:
        raise ValueError(f"Archive MD5 mismatch: {actual}")
    partial.rename(path)
    report = {"url": ARCHIVE_URL, "bytes": ARCHIVE_BYTES, "md5": actual,
              "seconds": time.monotonic() - started, "workers": workers}
    path.with_suffix(".download.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


def index_archive(path: Path, output: Path) -> None:
    """Index uncompressed class tars without creating extracted image copies."""
    if path.stat().st_size != ARCHIVE_BYTES or digest(path) != ARCHIVE_MD5:
        raise ValueError("Indexing requires the verified official archive")
    offsets, lengths, classes, names, class_names = [], [], [], [], []
    with tarfile.open(path, mode="r:") as archive:
        for outer in archive:
            if not outer.isfile() or not outer.name.endswith(".tar"):
                continue
            class_index = len(class_names)
            class_names.append(Path(outer.name).stem)
            with archive.extractfile(outer) as source, tarfile.open(fileobj=source, mode="r:") as images:
                for member in images:
                    if not member.isfile():
                        continue
                    if not member.name.lower().endswith(IMAGE_SUFFIXES):
                        raise ValueError(f"Unexpected training member: {member.name}")
                    offset = outer.offset_data + member.offset_data
                    if member.size <= 0 or offset + member.size > outer.offset_data + outer.size:
                        raise ValueError("Invalid nested image bounds")
                    offsets.append(offset)
                    lengths.append(member.size)
                    classes.append(class_index)
                    names.append(member.name)
    if (len(class_names) != ARCHIVE_CLASSES or len(offsets) != ARCHIVE_IMAGES
            or len(set(names)) != len(names)):
        raise ValueError("Unexpected ImageNet training class, image or unique-name count")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        np.savez(stream, offsets=np.asarray(offsets, dtype=np.int64),
                 lengths=np.asarray(lengths, dtype=np.int64),
                 classes=np.asarray(classes, dtype=np.int16),
                 names=np.asarray(names), class_names=np.asarray(class_names))
    report = {"archive": str(path.resolve()), "archive_bytes": ARCHIVE_BYTES,
              "archive_md5": ARCHIVE_MD5, "images": len(offsets), "classes": len(class_names),
              "index_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
              "offset_convention": "Absolute byte offset of image data within the original outer tar"}
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
