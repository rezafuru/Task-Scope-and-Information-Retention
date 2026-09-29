"""Fixed BREEDS ENTITY-30 mapping and streaming labeled ImageNet datasets.

The mapping is rebuilt from the three BREEDS hierarchy files rather than imported, so the
selection of eight fine classes per superclass is reproduced exactly, including the
``RandomState(2)`` draw and the level-4 node ordering the upstream generator uses. Every
constructed mapping is checked against torchvision's lexicographic synset order, which is
what makes the retained fine indices valid for the pretrained classifier.

Training images are read from the official archive through byte ranges recorded by
:mod:`taskscope.imagenet.archive`, so no extracted copy of ImageNet is created. Official
validation images are read as flat JPEGs and split into 25 validation and 25 test images
per fine class under a fixed seed.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import Dataset
from torchvision.models import ResNet50_Weights

BREEDS_REVISION = "f83be509107e89f41dde0e4e8e7f8be051fbb891"
ROBUSTNESS_REVISION = "a9541241defd9972e9334bfcdb804f6aefe24dc7"
SPLIT_SEED = 20260912
COARSE_CLASSES = 30
FINE_CLASSES = 240

ARCHIVE_URL = "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar"
ARCHIVE_BYTES = 147897477120
ARCHIVE_MD5 = "1d675b47d978889d74fa0da5fadfb00e"
ARCHIVE_IMAGES = 1281167
ARCHIVE_CLASSES = 1000

HIERARCHY_FILES = ("dataset_class_info.json", "class_hierarchy.txt", "node_names.txt")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def entity30(directory: Path) -> dict:
    """Reproduce make_entity30(split=None), including RandomState(2) selection."""
    info = json.loads((directory / "dataset_class_info.json").read_text())
    leaves = {row[1]: row for row in info}
    children, parents = {}, {}
    for line in (directory / "class_hierarchy.txt").read_text().splitlines():
        parent, child = line.split()
        children.setdefault(parent, set()).add(child)
        parents.setdefault(child, set()).add(parent)
    relevant, pending = set(leaves), list(leaves)
    while pending:
        for parent in parents.get(pending.pop(), ()):
            if parent not in relevant:
                relevant.add(parent)
                pending.append(parent)
    depths = {"n00001740": 0}
    pending = ["n00001740"]
    while pending:
        parent = pending.pop()
        for child in children.get(parent, ()):
            if child in relevant and depths.get(child, -1) < depths[parent] + 1:
                depths[child] = depths[parent] + 1
                pending.append(child)
    names = dict(line.split("\t", 1) for line in
                 (directory / "node_names.txt").read_text().splitlines())
    rng, groups = np.random.RandomState(2), []
    for node in sorted(node for node, depth in depths.items() if depth == 4):
        descendants, pending = set(), [node]
        while pending:
            current = pending.pop()
            if current in descendants:
                continue
            descendants.add(current)
            pending.extend(children.get(current, ()))
        candidates = sorted(descendants & leaves.keys())
        if len(candidates) >= 8:
            selected = rng.choice(candidates, 8, replace=False).tolist()
            groups.append({"coarse": len(groups), "wnid": node, "name": names[node],
                           "fine": [{"wnid": w, "imagenet_index": leaves[w][0],
                                     "name": leaves[w][2]} for w in selected]})
    all_fine = [row["wnid"] for group in groups for row in group["fine"]]
    if len(groups) != COARSE_CLASSES or len(set(all_fine)) != FINE_CLASSES:
        raise ValueError("ENTITY-30 requires 30 disjoint groups of eight classes")
    # Torchvision's ImageNet classifier uses lexicographically ordered synsets.
    if any(row[0] != index for index, row in enumerate(sorted(info, key=lambda r: r[1]))):
        raise ValueError("BREEDS class indices disagree with torchvision synset order")
    return {"name": "BREEDS ENTITY-30, split=None", "coarse_classes": COARSE_CLASSES,
            "fine_classes": FINE_CLASSES, "groups": groups, "breeds_revision": BREEDS_REVISION,
            "construction_revision": ROBUSTNESS_REVISION,
            "source_sha256": {name: sha256(directory / name) for name in HIERARCHY_FILES},
            "construction": "level=4, Nsubclasses=8, balanced=True, split=None, random_seed=2"}


def class_maps(mapping: dict) -> tuple[dict, list[int], list[int]]:
    classes, original, coarse = {}, [], []
    for group in mapping["groups"]:
        for row in group["fine"]:
            classes[row["wnid"]] = (len(original), group["coarse"])
            original.append(row["imagenet_index"])
            coarse.append(group["coarse"])
    return classes, original, coarse


def validation_labels(devkit: Path) -> tuple[list[str], dict]:
    """Join official numeric validation IDs to WNIDs through the official meta.mat."""
    meta_path = devkit / "data/meta.mat"
    target_path = devkit / "data/ILSVRC2012_validation_ground_truth.txt"
    meta = loadmat(meta_path, squeeze_me=True, struct_as_record=False)["synsets"]
    labels = {int(row.ILSVRC2012_ID): str(row.WNID) for row in meta
              if int(row.num_children) == 0}
    if len(labels) != ARCHIVE_CLASSES or set(labels) != set(range(1, ARCHIVE_CLASSES + 1)):
        raise ValueError("Require the official 1000 leaf IDs")
    targets = np.loadtxt(target_path, dtype=np.int64)
    if targets.shape != (50000,) or not set(targets).issubset(labels):
        raise ValueError("Require 50,000 official validation targets")
    return [labels[int(target)] for target in targets], {
        "meta_sha256": sha256(meta_path), "ground_truth_sha256": sha256(target_path)}


def split_validation(wnids: list[str], classes: dict) -> dict[str, np.ndarray]:
    """Allocate 25 official validation images per selected fine class to each split."""
    labels = np.asarray(wnids)
    rng = np.random.default_rng(SPLIT_SEED)
    result = {"validation": [], "test": []}
    for wnid in sorted(classes):
        rows = np.flatnonzero(labels == wnid)
        if len(rows) != 50:
            raise ValueError(f"Expected 50 validation images for {wnid}")
        rows = rng.permutation(rows)
        result["validation"].extend(rows[:25])
        result["test"].extend(rows[25:])
    return {key: np.asarray(sorted(value), dtype=np.int64) for key, value in result.items()}


def validate_class_counts(fine: np.ndarray, fine_classes: int, per_class: int | None) -> None:
    counts = np.bincount(fine, minlength=fine_classes)
    if len(counts) != fine_classes or (counts == 0).any():
        raise ValueError("Selected dataset must contain every declared fine class")
    if per_class is not None and not np.all(counts == per_class):
        raise ValueError("Available images do not satisfy the exact requested per-class limit")


class ImageNetImages(Dataset):
    """Load one image at a time from indexed training bytes or flat validation JPEGs."""

    def __init__(self, root: Path, mapping: dict, split: str,
                 per_class: int | None = None) -> None:
        if split not in {"train", "validation", "test"}:
            raise ValueError("Unknown split")
        if per_class is not None and per_class < 1:
            raise ValueError("per_class must be positive")
        self.root, self.split, self.fd = root, split, None
        self.transform = ResNet50_Weights.IMAGENET1K_V2.transforms()
        classes, _, _ = class_maps(mapping)
        if split == "train":
            index = root / "train_index.npz"
            identity = json.loads(index.with_suffix(".json").read_text())
            if (identity["archive_bytes"] != ARCHIVE_BYTES or identity["archive_md5"] != ARCHIVE_MD5
                    or identity["images"] != ARCHIVE_IMAGES
                    or identity["classes"] != ARCHIVE_CLASSES
                    or sha256(index) != identity["index_sha256"]):
                raise ValueError("Training index identity failed")
            self.archive = Path(identity["archive"])
            stat = self.archive.stat()
            self.archive_stat = (stat.st_size, stat.st_mtime_ns)
            if stat.st_size != ARCHIVE_BYTES:
                raise ValueError("Archive size differs from official verified archive")
            with np.load(index, allow_pickle=False) as saved:
                names = saved["class_names"].astype(str)
                fine_map = np.asarray([classes.get(w, (-1, -1))[0] for w in names])
                coarse_map = np.asarray([classes.get(w, (-1, -1))[1] for w in names])
                fine = fine_map[saved["classes"]]
                rows = np.flatnonzero(fine >= 0)
                if per_class:
                    rng = np.random.default_rng(SPLIT_SEED)
                    rows = np.sort(np.concatenate([rng.permutation(np.flatnonzero(fine == k))[:per_class]
                                                   for k in range(len(classes))]))
                self.fine = fine[rows]
                self.coarse = coarse_map[saved["classes"][rows]]
                self.offsets, self.lengths = saved["offsets"][rows], saved["lengths"][rows]
                self.ids = saved["names"][rows].astype(str).tolist()
            if ((self.offsets < 0).any() or (self.lengths <= 0).any()
                    or (self.offsets + self.lengths > ARCHIVE_BYTES).any()):
                raise ValueError("Invalid archive byte ranges")
            self.identity = identity
        else:
            wnids, identity = validation_labels(root / "ILSVRC2012_devkit_t12")
            rows = split_validation(wnids, classes)[split]
            if per_class:
                rows = np.asarray(sorted(row for wnid in sorted(classes) for row in
                                         [i for i in rows if wnids[i] == wnid][:per_class]))
            self.fine = np.asarray([classes[wnids[i]][0] for i in rows])
            self.coarse = np.asarray([classes[wnids[i]][1] for i in rows])
            self.ids = [f"ILSVRC2012_val_{int(i) + 1:08d}.JPEG" for i in rows]
            if any(not (root / "validation" / name).is_file() for name in self.ids):
                raise ValueError("Missing selected official validation images")
            self.identity = identity
        validate_class_counts(self.fine, len(classes), per_class)
        self.identity = {**self.identity, "split": split, "split_seed": SPLIT_SEED,
                         "images": len(self.ids), "per_class_limit": per_class,
                         "membership_sha256": hashlib.sha256("\n".join(self.ids).encode()).hexdigest()}

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> tuple:
        if self.split == "train":
            if self.fd is None:
                self.fd = os.open(self.archive, os.O_RDONLY)
                stat = os.fstat(self.fd)
                if (stat.st_size, stat.st_mtime_ns) != self.archive_stat:
                    raise ValueError("Training archive changed before worker access")
            content = os.pread(self.fd, int(self.lengths[index]), int(self.offsets[index]))
            if len(content) != int(self.lengths[index]):
                raise OSError("Short training image read")
            source = io.BytesIO(content)
        else:
            source = self.root / "validation" / self.ids[index]
        with Image.open(source) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, int(self.fine[index]), int(self.coarse[index]), self.ids[index]

    def __getstate__(self) -> dict:
        return {**self.__dict__, "fd": None}


def build_mapping(hierarchy: Path, output: Path, data_root: Path | None = None) -> dict:
    """Write the ENTITY-30 mapping, and the split memberships when the images are present."""
    mapping = entity30(hierarchy)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "mapping.json", mapping)
    if data_root is None:
        return mapping
    datasets = {split: ImageNetImages(data_root, mapping, split)
                for split in ("train", "validation", "test")}
    write_json(output / "data_audit.json", {split: data.identity for split, data in datasets.items()})
    for split, data in datasets.items():
        write_json(output / f"{split}_membership.json", {"ids": data.ids, "fine": data.fine.tolist(),
                                                         "coarse": data.coarse.tolist()})
    return mapping
