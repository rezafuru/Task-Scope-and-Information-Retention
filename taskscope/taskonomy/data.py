"""Taskonomy Tiny building splits, aligned domain reading, and the manifest cache.

Buildings are partitioned once into train, validation, and test groups, so no image of a
building ever appears in two splits. Within a split each building contributes a
midpoint-stratified stride of its images and the buildings are interleaved round robin,
which keeps every prefix of the returned list balanced across buildings.

The manifest is a cache of RGB stems per building. It is rebuilt by scanning the data root
whenever the file is absent, and it is not an input the experiment depends on.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

TRAIN_BUILDINGS = (
    "hanson",
    "merom",
    "klickitat",
    "onaga",
    "leonardo",
    "marstons",
    "newfields",
    "pinesdale",
    "lakeville",
    "cosmos",
    "benevolence",
    "pomaria",
    "tolstoy",
    "shelbyville",
    "allensville",
    "wainscott",
    "beechwood",
    "coffeen",
    "stockman",
    "hiteman",
    "woodbine",
    "lindenwood",
    "forkland",
    "mifflinburg",
    "ranchester",
)
VAL_BUILDINGS = ("wiconisco", "corozal", "collierville", "markleeville", "darden")
TEST_BUILDINGS = ("ihlen", "muleshoe", "uvalda", "noxapater", "mcdade")
BUILDING_SPLITS = {
    "train": TRAIN_BUILDINGS,
    "val": VAL_BUILDINGS,
    "test": TEST_BUILDINGS,
}

# Excluded from every split, in both the manifest scan and the sampler.
CORRUPT_KEYS = {
    "muleshoe/point_399_view_5",
    "newfields/point_1070_view_8",
}
# Listed in the split above but absent from the Tiny release, so it is skipped silently.
MISSING_TINY_ARCHIVES = {"woodbine"}

MANIFEST_VERSION = 1
SAMPLE_POLICY = "balanced-building-midpoint-stratified-round-robin-v1"
IMAGE_SIZE = 256
NUM_SEMANTIC_CLASSES = 17
SEMANTIC_IGNORE_INDEX = 255
DEPTH_UNITS_PER_METRE = 512.0
SEMANTIC_CLASS_NAMES = (
    "background",
    "bottle",
    "chair",
    "couch",
    "potted_plant",
    "bed",
    "dining_table",
    "toilet",
    "tv",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
)


@dataclass(frozen=True)
class SamplePaths:
    key: str
    rgb: Path
    semantic: Path
    depth: Path
    mask: Path


def _key_from_rgb_path(rgb_path: Path) -> str:
    suffix = "_domain_rgb.png"
    if not rgb_path.name.endswith(suffix):
        raise ValueError(f"unexpected Taskonomy RGB filename {rgb_path.name!r}")
    return rgb_path.name[: -len(suffix)]


def _load_manifest(cache: Path, root: Path) -> dict[str, Any]:
    expected_root = str(root.resolve())
    if not cache.is_file():
        return {
            "version": MANIFEST_VERSION,
            "data_root": expected_root,
            "splits": {},
        }
    value = json.loads(cache.read_text(encoding="utf-8"))
    if value.get("version") != MANIFEST_VERSION:
        raise ValueError(f"unsupported dataset manifest version in {cache}")
    if value.get("data_root") != expected_root:
        raise ValueError(f"dataset manifest {cache} belongs to another data root")
    if not isinstance(value.get("splits"), dict):
        raise ValueError(f"dataset manifest {cache} has no split mapping")
    return value


def _save_manifest(cache: Path, value: Mapping[str, Any]) -> None:
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_name(f"{cache.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, cache)


def _rgb_stems_by_building(
    root: Path,
    split: str,
    manifest_cache: Path | None,
) -> dict[str, list[str]]:
    manifest = _load_manifest(manifest_cache, root) if manifest_cache is not None else None
    if manifest is not None and split in manifest["splits"]:
        cached = manifest["splits"][split]
        if not isinstance(cached, dict):
            raise ValueError(f"cached split {split!r} is malformed")
        expected_buildings = tuple(
            building
            for building in BUILDING_SPLITS[split]
            if building not in MISSING_TINY_ARCHIVES
        )
        if set(cached) != set(expected_buildings):
            raise ValueError(
                f"cached split {split!r} has unexpected buildings: {sorted(cached)}"
            )
        return {building: list(cached[building]) for building in expected_buildings}

    result: dict[str, list[str]] = {}
    for building in BUILDING_SPLITS[split]:
        rgb_dir = root / "rgb" / "taskonomy" / building
        if not rgb_dir.is_dir():
            if building in MISSING_TINY_ARCHIVES:
                continue
            raise FileNotFoundError(f"missing Taskonomy directory {rgb_dir}")
        result[building] = [
            _key_from_rgb_path(rgb_path)
            for rgb_path in sorted(rgb_dir.glob("*_domain_rgb.png"))
        ]
    if manifest is not None:
        manifest["splits"][split] = result
        _save_manifest(manifest_cache, manifest)
    return result


def enumerate_taskonomy_samples(
    root: Path,
    split: str,
    max_samples: int | None = None,
    manifest_cache: Path | None = None,
) -> list[SamplePaths]:
    """Aligned four-domain sample paths for one split under the balanced sampling policy.

    The per-building quota splits ``max_samples`` as evenly as the building count allows.
    Each building keeps the stems at midpoints of equal-length intervals, then buildings are
    interleaved so truncating the list to any length preserves the balance.
    """
    if split not in BUILDING_SPLITS:
        raise ValueError(f"unknown split {split!r}")
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive")
    stems_by_building = _rgb_stems_by_building(root, split, manifest_cache)
    samples_by_building: list[list[SamplePaths]] = []
    building_count = len(stems_by_building)
    if max_samples is None:
        limits: dict[str, int | None] = {building: None for building in stems_by_building}
    else:
        quotient, remainder = divmod(max_samples, max(1, building_count))
        limits = {
            building: quotient + int(index < remainder)
            for index, building in enumerate(stems_by_building)
        }
    for building, all_stems in stems_by_building.items():
        building_samples: list[SamplePaths] = []
        available_stems = [
            stem for stem in all_stems if f"{building}/{stem}" not in CORRUPT_KEYS
        ]
        per_building_limit = limits[building]
        if per_building_limit is None or per_building_limit >= len(available_stems):
            stems = available_stems
        else:
            count = per_building_limit
            stems = [
                available_stems[((2 * index + 1) * len(available_stems)) // (2 * count)]
                for index in range(count)
            ]
        for stem in stems:
            key = f"{building}/{stem}"
            sample = SamplePaths(
                key=key,
                rgb=root / "rgb" / "taskonomy" / building / f"{stem}_domain_rgb.png",
                semantic=(
                    root
                    / "segment_semantic"
                    / "taskonomy"
                    / building
                    / f"{stem}_domain_segmentsemantic.png"
                ),
                depth=(
                    root
                    / "depth_zbuffer"
                    / "taskonomy"
                    / building
                    / f"{stem}_domain_depth_zbuffer.png"
                ),
                mask=(
                    root
                    / "mask_valid"
                    / "taskonomy"
                    / building
                    / f"{stem}_domain_depth_zbuffer.png"
                ),
            )
            missing = [
                str(path)
                for path in (sample.semantic, sample.depth, sample.mask)
                if not path.is_file()
            ]
            if missing:
                raise FileNotFoundError(
                    f"sample {sample.key} has missing aligned domains: {missing}"
                )
            building_samples.append(sample)
        samples_by_building.append(building_samples)
    samples = [
        building_samples[index]
        for index in range(max(map(len, samples_by_building)))
        for building_samples in samples_by_building
        if index < len(building_samples)
    ]
    if max_samples is not None:
        samples = samples[:max_samples]
    if not samples:
        raise RuntimeError(f"no Taskonomy samples found for split {split!r}")
    return samples


def _read_png(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image).copy()


def _resize_rgb(array: np.ndarray) -> Tensor:
    image = Image.fromarray(array).resize(
        (IMAGE_SIZE, IMAGE_SIZE), resample=Image.Resampling.BOX
    )
    value = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)
    return value.float().div_(255.0)


def _resize_semantic(array: np.ndarray) -> Tensor:
    """Nearest-neighbour resize, with class zero and anything above the label range ignored."""
    if array.ndim == 3:
        array = array[..., 0]
    image = Image.fromarray(array).resize(
        (IMAGE_SIZE, IMAGE_SIZE), resample=Image.Resampling.NEAREST
    )
    raw = torch.from_numpy(np.asarray(image).copy()).long()
    target = torch.full_like(raw, SEMANTIC_IGNORE_INDEX)
    valid = (raw >= 1) & (raw <= NUM_SEMANTIC_CLASSES)
    target[valid] = raw[valid] - 1
    return target


def _resize_depth_and_mask(depth_raw: np.ndarray, mask_raw: np.ndarray) -> tuple[Tensor, Tensor]:
    """Z-buffer counts to metres, with invalid pixels zeroed and their mask returned."""
    if depth_raw.ndim == 3:
        depth_raw = depth_raw[..., 0]
    if mask_raw.ndim == 3:
        mask_raw = mask_raw[..., 0]
    depth = torch.from_numpy(depth_raw.astype(np.float32))[None, None]
    valid = torch.from_numpy((mask_raw > 0).astype(np.float32))[None, None]
    resized_depth = F.interpolate(depth, size=(IMAGE_SIZE, IMAGE_SIZE), mode="nearest")
    resized_valid = F.interpolate(valid, size=(IMAGE_SIZE, IMAGE_SIZE), mode="nearest") > 0.5
    depth_metres = resized_depth.div(DEPTH_UNITS_PER_METRE)
    depth_metres = torch.where(resized_valid, depth_metres, torch.zeros_like(depth_metres))
    return depth_metres[0], resized_valid[0]


class TaskonomyRequirements(Dataset[dict[str, Any]]):
    """Aligned RGB, semantic, z-buffer depth, and validity observations."""

    def __init__(
        self,
        root: Path,
        split: str,
        max_samples: int | None = None,
        manifest_cache: Path | None = None,
    ) -> None:
        self.samples = enumerate_taskonomy_samples(root, split, max_samples, manifest_cache)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        rgb = _resize_rgb(_read_png(sample.rgb))
        semantic = _resize_semantic(_read_png(sample.semantic))
        depth, valid = _resize_depth_and_mask(_read_png(sample.depth), _read_png(sample.mask))
        return {
            "key": sample.key,
            "rgb": rgb,
            "semantic": semantic,
            "depth": depth,
            "depth_valid": valid,
        }


def move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }
