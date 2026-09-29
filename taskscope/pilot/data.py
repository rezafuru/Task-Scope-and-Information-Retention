"""Source preparation: the CUB development and test populations and their frozen exits.

Every photograph becomes one 224 by 224 RGB PNG. The crop is square with side
``ceil(1.2 * max(bbox width, bbox height))``, placed at ``floor(centre - side/2)``, padded with
RGB 127 where it leaves the photograph, and resized to 224 with bilinear interpolation.

Duplicate photographs are grouped by original file bytes, by original RGB pixels, and by prepared
RGB pixels, and a group never crosses a split. CUB validation is six whole groups per category in
the order of SHA256 of ``pilot_scope_2026:cub_categories:validation:<class>:<group>``.

Feature extraction writes one float32 array per exit for the requested photographs, after
checking that the unchanged network suffix reproduces the original logits exactly at every exit.

External inputs, both required arguments: the CUB data root (the extracted ``CUB_200_2011``
tree plus the official archive) and the 200-to-36 family mapping.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from PIL import __version__ as pillow_version

from taskscope.pilot import features as feat
from taskscope.pilot.common import file_hash, log, read_json, stable_hash, write_json

NAMESPACE = 1_000_000_000_000
CUB_ARCHIVE_SHA256 = "0c685df5597a8b24909f6a7c9db6d11e008733779a671760afef78feb49bf081"
FAMILY_MAPPING_SHA256 = "9bcba8fad1db479f45849fa1d099949ae9c4c01cac45db8529239de92533bf91"
VALIDATION_IMAGES = 6
CUB_METADATA = ("images.txt", "classes.txt", "image_class_labels.txt", "train_test_split.txt",
                "bounding_boxes.txt")
CROP_RULE = ("square ceil(1.2*max(bbox width,height)), floor(center-side/2), RGB127 padding, "
             "bilinear224")

# The inventories behind the retained evidence. A regenerated inventory records its own
# configuration, so its hash differs. Pass one of these to --expect-inventory-sha256 to pin.
PUBLISHED_INVENTORY_SHA256 = {
    "development": "47638e997edec794533cce94723501e9148993436d0d5ede17c6ac0793f8d168",
    "test": "b459c8c4acf13556cd8379330851ef05459ff75485666a5c2f5a3a0b2f1ec8d3",
}
POPULATION_COUNTS = {"development": {"train": 4794, "val": 1200}, "test": {"test": 5794}}


def source_crop(image: Image.Image, bbox: list[float]) -> tuple[Image.Image, dict]:
    """Bird-box-centred square crop with grey padding, resized to 224."""
    x, y, width, height = bbox
    if width <= 0 or height <= 0:
        raise ValueError("Nonpositive source bounding box")
    side = math.ceil(1.2 * max(width, height))
    left = math.floor(x + width / 2 - side / 2)
    top = math.floor(y + height / 2 - side / 2)
    square = Image.new("RGB", (side, side), (127, 127, 127))
    square.paste(image.convert("RGB"), (-left, -top))
    output = square.resize((224, 224), Image.Resampling.BILINEAR)
    scale = 224 / side
    metadata = {
        "bird_bbox_source_xywh": bbox, "square_source_xyxy": [left, top, left + side, top + side],
        "square_side": side,
        "padding_ltrb": [max(0, -left), max(0, -top),
                         max(0, left + side - image.width), max(0, top + side - image.height)],
        "bird_bbox_input224_xyxy": [(x - left) * scale, (y - top) * scale,
                                    (x + width - left) * scale, (y + height - top) * scale]}
    return output, metadata


def check_png(path: Path, byte_hash: str, pixel_hash: str) -> None:
    if file_hash(path) != byte_hash:
        raise ValueError(f"Image bytes changed: {path}")
    with Image.open(path) as image:
        if image.format != "PNG" or image.mode != "RGB" or image.size != (224, 224):
            raise ValueError("Require an exact 224-by-224 RGB PNG")
        if hashlib.sha256(image.tobytes()).hexdigest() != pixel_hash:
            raise ValueError(f"Image pixels changed: {path}")


def source_rows(records: list[dict], data_root: Path) -> list[dict]:
    """The explicit source list the inversion stage reads."""
    rows = []
    for row in records:
        path = (Path(data_root) / row["input_path"]).resolve()
        if Path(row["input_path"]).is_absolute() or not path.is_relative_to(Path(data_root).resolve()):
            raise ValueError("Image path must remain inside its recorded data root")
        rows.append({"id": row["observation_id"], "path": str(path), "sha256": row["input_sha256"]})
    return rows


def load_inventory(path: Path, population: str, expected_sha256: str | None = None) -> list[dict]:
    """Read a prepared CUB inventory and check its structure, labels and split integrity."""
    counts = POPULATION_COUNTS[population]
    if expected_sha256 is not None and file_hash(path) != expected_sha256:
        raise ValueError("CUB inventory differs from the pinned population")
    records = read_json(path)["records"]
    if Counter(row["split"] for row in records) != counts:
        raise ValueError("CUB split counts changed")
    if len({row["observation_id"] for row in records}) != len(records):
        raise ValueError("Duplicate CUB observation IDs")
    groups: dict[int, str] = {}
    for row in records:
        if (row["fine_label"] != row["cub_class_id"] - 1 or "imagenet_label" in row
                or row["official_split"] != ("test" if population == "test" else "train")):
            raise ValueError("Original CUB labels or official split changed")
        if groups.setdefault(row["group_id"], row["split"]) != row["split"]:
            raise ValueError("A source group crosses CUB splits")
    for split in counts:
        if {row["cub_class_id"] for row in records if row["split"] == split} != set(range(1, 201)):
            raise ValueError("Every CUB class is required in every population split")
    return records


def _cub_tables(data: Path) -> dict[str, dict]:
    image_paths = {int(i): path for i, path in
                   (line.split(maxsplit=1) for line in (data / "images.txt").read_text().splitlines())}
    official = {int(i): int(flag) for i, flag in
                (line.split() for line in (data / "train_test_split.txt").read_text().splitlines())}
    classes = {int(i): int(label) for i, label in
               (line.split() for line in (data / "image_class_labels.txt").read_text().splitlines())}
    boxes = {int(fields[0]): list(map(float, fields[1:])) for fields in
             (line.split() for line in (data / "bounding_boxes.txt").read_text().splitlines())}
    if len(image_paths) != 11788 or sum(official.values()) != 5994 or set(official.values()) != {0, 1}:
        raise ValueError("Unexpected official CUB inventory")
    if not set(image_paths) == set(official) == set(classes) == set(boxes):
        raise ValueError("CUB metadata image IDs differ")
    return {"paths": image_paths, "official": official, "classes": classes, "boxes": boxes}


def cub_source_hashes(data: Path, image_paths: dict[int, str]) -> tuple[dict, dict]:
    """Original file and pixel hashes for every CUB photograph, with duplicate groups."""
    parents = {image_id: image_id for image_id in image_paths}

    def find(image_id: int) -> int:
        while parents[image_id] != image_id:
            parents[image_id] = parents[parents[image_id]]
            image_id = parents[image_id]
        return image_id

    def unite(first: int, second: int) -> None:
        first, second = find(first), find(second)
        parents[max(first, second)] = min(first, second)

    seen_file, seen_pixels, hashes = {}, {}, {}
    for index, (image_id, relative) in enumerate(sorted(image_paths.items()), 1):
        path = data / "images" / relative
        digest = file_hash(path)
        with Image.open(path) as image:
            orientation = image.getexif().get(274, 1)
            rgb = image.convert("RGB")
            pixel = hashlib.sha256(f"{rgb.width}x{rgb.height}:".encode() + rgb.tobytes()).hexdigest()
        hashes[image_id] = {"image_sha256": digest, "source_pixel_sha256": pixel,
                            "source_bytes": path.stat().st_size, "width": rgb.width,
                            "height": rgb.height, "exif_orientation": orientation}
        for table, value in ((seen_file, digest), (seen_pixels, pixel)):
            if value in table:
                unite(image_id, table[value])
            else:
                table[value] = image_id
        if index % 2000 == 0:
            log(stage="cub_source_hashes", hashed=index, total=len(image_paths))
    return hashes, {image_id: find(image_id) for image_id in image_paths}


def merged_groups(source_groups: dict[int, int], pixels: dict[int, str]) -> dict[int, int]:
    """Extend the duplicate groups by equality of the prepared 224 pixels."""
    parents = dict(source_groups)

    def find(image_id: int) -> int:
        while parents[image_id] != image_id:
            parents[image_id] = parents[parents[image_id]]
            image_id = parents[image_id]
        return image_id

    seen: dict[str, int] = {}
    for image_id, digest in sorted(pixels.items()):
        if digest in seen:
            first, second = find(image_id), find(seen[digest])
            parents[max(first, second)] = min(first, second)
        else:
            seen[digest] = image_id
    return {image_id: find(image_id) for image_id in parents}


def validation_groups(groups: dict[int, list[int]], class_id: int) -> set[int]:
    """Six whole duplicate groups per category, in fixed hash order."""
    ordered = sorted(groups, key=lambda group: stable_hash(
        f"pilot_scope_2026:cub_categories:validation:{class_id}:{group}"))
    selected, count = set(), 0
    for group in ordered:
        if count + len(groups[group]) <= VALIDATION_IMAGES:
            selected.add(group)
            count += len(groups[group])
        if count == VALIDATION_IMAGES:
            return selected
    raise ValueError(f"Cannot select six whole-group validation images for class {class_id}")


def prepare_cub_development(output: Path, *, cub_root: Path, mapping: Path, workers: int = 8) -> dict:
    """Prepare the official CUB training partition and choose the validation split."""
    root, output = Path(cub_root).resolve(), Path(output).resolve()
    data = root / "CUB_200_2011"
    if file_hash(mapping) != FAMILY_MAPPING_SHA256:
        raise ValueError("The pinned category mapping differs")
    if file_hash(root / "CUB_200_2011.tgz") != CUB_ARCHIVE_SHA256:
        raise ValueError("The official source archive differs")
    family = read_json(mapping)
    labels = {row["cub_class_id"]: row for row in family["records"]}
    if set(labels) != set(range(1, 201)) or len(family["coarse_class_names"]) != 36:
        raise ValueError("Expected 200 supplied categories in 36 groups")
    for class_id, row in labels.items():
        if row["fine_label"] != class_id - 1:
            raise ValueError("Fine indices must retain the official category order")
        if family["coarse_class_names"][row["coarse_label"]] != row["coarse_class_name"]:
            raise ValueError("Inconsistent coarse mapping")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "inventory.json").exists():
        raise FileExistsError("A complete category inventory already exists")
    tables = _cub_tables(data)
    metadata_hashes = {name: file_hash(data / name) for name in CUB_METADATA}
    hashes, source_groups = cub_source_hashes(data, tables["paths"])
    inputs = root / "inputs224_categories"
    inputs.mkdir(exist_ok=True)

    def prepare(image_id: int) -> tuple[int, dict]:
        with Image.open(data / "images" / tables["paths"][image_id]) as image:
            prepared, geometry = source_crop(image, tables["boxes"][image_id])
        row = {"pixel_sha256": hashlib.sha256(prepared.tobytes()).hexdigest(), **geometry}
        if tables["official"][image_id]:
            path = inputs / f"{NAMESPACE + image_id}.png"
            if path.exists():
                with Image.open(path) as existing:
                    if (existing.mode != "RGB" or existing.size != (224, 224)
                            or existing.tobytes() != prepared.tobytes()):
                        raise ValueError(f"Existing input differs for image {image_id}")
            else:
                temporary = path.with_suffix(".tmp")
                prepared.save(temporary, format="PNG")
                temporary.replace(path)
            row.update(input_path=str(path.relative_to(root)), input_sha256=file_hash(path),
                       input_bytes=path.stat().st_size)
        return image_id, row

    prepared = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for index, (image_id, row) in enumerate(executor.map(prepare, sorted(tables["paths"])), 1):
            prepared[image_id] = row
            if index % 2000 == 0:
                log(stage="cub_development", prepared=index, total=len(tables["paths"]))
    groups = merged_groups(source_groups, {i: row["pixel_sha256"] for i, row in prepared.items()})
    members = defaultdict(list)
    for image_id, group in groups.items():
        members[group].append(image_id)
    for identifiers in members.values():
        if len({tables["official"][i] for i in identifiers}) != 1:
            raise ValueError(f"Duplicate group crosses official partitions: {identifiers}")
        if len({tables["classes"][i] for i in identifiers}) != 1:
            raise ValueError(f"Duplicate group has conflicting supplied labels: {identifiers}")
    validation = set()
    for class_id in range(1, 201):
        candidates = {group: identifiers for group, identifiers in members.items()
                      if tables["official"][identifiers[0]] and tables["classes"][identifiers[0]] == class_id}
        validation.update(validation_groups(candidates, class_id))
    records = []
    for image_id in sorted(tables["paths"]):
        if not tables["official"][image_id]:
            continue
        label = labels[tables["classes"][image_id]]
        records.append({
            "observation_id": NAMESPACE + image_id, "group_id": NAMESPACE + groups[image_id],
            "cub_image_id": image_id, "cub_class_id": tables["classes"][image_id],
            "official_split": "train",
            "split": "val" if groups[image_id] in validation else "train", "status": "downloaded",
            "source_group": f"cub_image_duplicate_group_{groups[image_id]}",
            "source_path": str(Path("CUB_200_2011/images") / tables["paths"][image_id]),
            "coarse_label": label["coarse_label"], "coarse_class_name": label["coarse_class_name"],
            "fine_label": label["fine_label"], "fine_class_name": label["fine_class_name"],
            "fine_eligible": True, **hashes[image_id], **prepared[image_id]})
    counts = {split: Counter(row["fine_label"] for row in records if row["split"] == split)
              for split in ("train", "val")}
    if set(counts["train"]) != set(range(200)) or set(counts["val"].values()) != {6}:
        raise ValueError("Unexpected development class coverage")
    for key in ("group_id", "input_sha256", "pixel_sha256", "image_sha256", "source_pixel_sha256"):
        sides = [{row[key] for row in records if row["split"] == split} for split in ("train", "val")]
        if sides[0] & sides[1]:
            raise ValueError(f"Development split overlap in {key}")
    config = {"archive_sha256": CUB_ARCHIVE_SHA256, "mapping_sha256": FAMILY_MAPPING_SHA256,
              "pillow_version": pillow_version, "data_root": str(root), "official_partition": "train",
              "population": "all 200 supplied bird categories", "crop": CROP_RULE,
              "validation": "six images per category, whole duplicate groups in "
                            "SHA256(pilot_scope_2026:cub_categories:validation:{class_id}:{group}) order, "
                            "skipping groups that exceed six",
              "duplicates": "union of original file, original RGB pixels, and prepared RGB pixels "
                            "across both official partitions",
              "test_policy": "compute source and prepared pixel hashes only, do not save "
                             "official-test inputs"}
    summary = {"prepared_images": len(records),
               "input_bytes": sum(row["input_bytes"] for row in records),
               "split_counts": dict(Counter(row["split"] for row in records)),
               "fine_class_counts": {split: dict(sorted(values.items())) for split, values in counts.items()},
               "duplicate_groups": [ids for ids in members.values() if len(ids) > 1],
               "heldout_sources_hash_checked": sum(flag == 0 for flag in tables["official"].values()),
               "heldout_inputs_saved": 0}
    write_json(output / "source_hash_inventory.json", {
        "metadata_sha256": metadata_hashes,
        "records": [{"cub_image_id": i, "official_train": bool(tables["official"][i]),
                     "group_min_image_id": groups[i], **hashes[i],
                     "prepared_pixel_sha256": prepared[i]["pixel_sha256"]}
                    for i in sorted(tables["paths"])]})
    inventory = {"acquisition_status": "complete", "config": config, "summary": summary,
                 "coarse_class_names": family["coarse_class_names"],
                 "fine_class_names": family["fine_class_names"],
                 "grouping": "Numeric CUB image or exact-duplicate group.",
                 "label_mapping_sha256": FAMILY_MAPPING_SHA256, "records": records}
    write_json(output / "inventory.json", inventory)
    log(stage="complete", inventory_sha256=file_hash(output / "inventory.json"), **summary["split_counts"])
    return inventory


def prepare_cub_test(output: Path, *, cub_root: Path, development: Path, workers: int = 8) -> dict:
    """Prepare the official CUB test partition after the development split is fixed."""
    root, output = Path(cub_root).resolve(), Path(output).resolve()
    data = root / "CUB_200_2011"
    development = Path(development)
    manifest = read_json(development / "inventory.json")
    source_inventory = read_json(development / "source_hash_inventory.json")
    if manifest["acquisition_status"] != "complete" or manifest["config"]["data_root"] != str(root):
        raise ValueError("Development preparation is incomplete or used a different data root")
    if manifest["config"]["pillow_version"] != pillow_version:
        raise ValueError("Development and test preparation must use one Pillow version")
    if file_hash(root / "CUB_200_2011.tgz") != CUB_ARCHIVE_SHA256:
        raise ValueError("The official archive differs")
    for name, digest in source_inventory["metadata_sha256"].items():
        if file_hash(data / name) != digest:
            raise ValueError(f"Source metadata changed: {name}")
    tables = _cub_tables(data)
    sources = {row["cub_image_id"]: row for row in source_inventory["records"]}
    labels = {row["cub_class_id"]: row for row in manifest["records"]}
    allowed = sorted(i for i, flag in tables["official"].items() if flag == 0)
    if len(allowed) != 5794 or set(labels) != set(range(1, 201)):
        raise ValueError("Unexpected official test population or category mapping")
    for key, values in {"observation_id": {NAMESPACE + i for i in allowed},
                        "group_id": {NAMESPACE + sources[i]["group_min_image_id"] for i in allowed},
                        "pixel_sha256": {sources[i]["prepared_pixel_sha256"] for i in allowed}}.items():
        if values & {row[key] for row in manifest["records"]}:
            raise ValueError(f"Official test overlaps development through {key}")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "inventory.json").exists():
        raise FileExistsError("A prepared category test inventory already exists")
    inputs = root / "inputs224_categories"
    inputs.mkdir(exist_ok=True)

    def prepare(image_id: int) -> dict:
        source = data / "images" / tables["paths"][image_id]
        known = sources[image_id]
        if file_hash(source) != known["image_sha256"]:
            raise ValueError(f"Original source bytes changed for {image_id}")
        with Image.open(source) as image:
            prepared, geometry = source_crop(image, tables["boxes"][image_id])
        if hashlib.sha256(prepared.tobytes()).hexdigest() != known["prepared_pixel_sha256"]:
            raise ValueError(f"Prepared pixels differ from the development hash audit for {image_id}")
        path = inputs / f"{NAMESPACE + image_id}.png"
        if path.exists():
            with Image.open(path) as previous:
                if (previous.mode != "RGB" or previous.size != (224, 224)
                        or previous.tobytes() != prepared.tobytes()):
                    raise ValueError(f"Existing heldout input differs for {image_id}")
        else:
            temporary = path.with_suffix(".tmp")
            prepared.save(temporary, format="PNG")
            temporary.replace(path)
        label = labels[tables["classes"][image_id]]
        return {"observation_id": NAMESPACE + image_id,
                "group_id": NAMESPACE + known["group_min_image_id"], "cub_image_id": image_id,
                "cub_class_id": tables["classes"][image_id], "split": "test", "official_split": "test",
                "status": "downloaded",
                "source_group": f"cub_image_duplicate_group_{known['group_min_image_id']}",
                "source_path": str(Path("CUB_200_2011/images") / tables["paths"][image_id]),
                "input_path": str(path.relative_to(root)), "input_sha256": file_hash(path),
                "input_bytes": path.stat().st_size, "pixel_sha256": known["prepared_pixel_sha256"],
                "fine_eligible": True,
                **{key: label[key] for key in
                   ("coarse_label", "coarse_class_name", "fine_label", "fine_class_name")},
                **{key: known[key] for key in ("image_sha256", "source_pixel_sha256", "source_bytes",
                                               "width", "height", "exif_orientation")},
                **geometry}

    records = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for index, row in enumerate(executor.map(prepare, allowed), 1):
            records.append(row)
            if index % 1000 == 0:
                log(stage="cub_test", prepared=index, total=len(allowed))
    if {row["input_sha256"] for row in records} & {row["input_sha256"] for row in manifest["records"]}:
        raise ValueError("Saved heldout PNG hashes overlap development")
    config = {**manifest["config"], "official_partition": "test",
              "development_inventory_sha256": file_hash(development / "inventory.json"),
              "source_inventory_sha256": file_hash(development / "source_hash_inventory.json"),
              "validation": "none, all records are official test",
              "test_policy": "prepare every official-test source after the development split is fixed"}
    summary = {"prepared_images": len(records), "input_bytes": sum(r["input_bytes"] for r in records),
               "split_counts": {"test": len(records)},
               "fine_class_counts": dict(sorted(Counter(r["fine_label"] for r in records).items())),
               "source_groups": len({r["group_id"] for r in records})}
    inventory = {"acquisition_status": "complete", "config": config, "summary": summary,
                 "coarse_class_names": manifest["coarse_class_names"],
                 "fine_class_names": manifest["fine_class_names"], "grouping": manifest["grouping"],
                 "label_mapping_sha256": manifest["label_mapping_sha256"], "records": records}
    write_json(output / "inventory.json", inventory)
    log(stage="complete", inventory_sha256=file_hash(output / "inventory.json"), **summary["split_counts"])
    return inventory


def load_images(records: list[dict], data_root: Path) -> torch.Tensor:
    """Stack prepared PNGs as uint8 CHW, checking every file hash against the inventory."""
    values = []
    for row in records:
        path = Path(data_root) / row["input_path"]
        if file_hash(path) != row["input_sha256"]:
            raise ValueError(f"Prepared image changed: {path}")
        with Image.open(path) as image:
            if image.mode != "RGB" or image.size != (224, 224):
                raise ValueError("Unexpected prepared image geometry")
            values.append(np.asarray(image).copy().transpose(2, 0, 1))
    return torch.from_numpy(np.stack(values))


@torch.inference_mode()
def extract_features(output: Path, records: list[dict], data_root: Path, device: torch.device,
                     exits=feat.EXITS, batch_size: int = 32) -> dict:
    """Write one float32 array per exit, checking exact logit identity of the unchanged suffix."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    teacher = feat.FrozenTeacher(device)
    count = len(records)
    arrays = {name: np.lib.format.open_memmap(output / f"{name}.npy", mode="w+", dtype=np.float32,
                                              shape=(count, *feat.EXIT_SHAPES[name])) for name in exits}
    logits_all = np.zeros((count, 1000), np.float32)
    images = load_images(records, data_root)
    blocks = list(teacher.network.layer4.children())
    started = time.monotonic()
    for start in range(0, count, batch_size):
        rgb = images[start:start + batch_size].to(device).float() / 255
        result = teacher(rgb, depth="all", logits=True)
        value = result["layer3"]
        maps = {"layer3": result["layer3"]}
        for index, block in enumerate(blocks):
            value = block(value)
            maps[f"block{index + 1}"] = value
        for name in exits:
            suffix = maps[name]
            if name == "layer3":
                suffix = teacher.network.layer4(suffix)
            else:
                for block in blocks[int(name[-1]):]:
                    suffix = block(suffix)
            after = teacher.network.fc(teacher.network.avgpool(suffix).flatten(1))
            if not torch.equal(after, result["logits"]):
                raise ValueError(f"Unchanged suffix does not recover the exact teacher logits at {name}")
            arrays[name][start:start + len(rgb)] = maps[name].cpu().numpy()
        logits_all[start:start + len(rgb)] = result["logits"].cpu().numpy()
    for array in arrays.values():
        array.flush()
    np.save(output / "logits.npy", logits_all)
    top1 = logits_all.argmax(1)
    membership_keys = ("observation_id", "group_id", "cub_image_id", "fine_label", "split",
                       "input_path", "input_sha256", "pixel_sha256")
    extraction = {
        "status": "complete", "teacher": teacher.provenance(),
        "operation": "Complete native exits of the unchanged network on prepared 224 RGB",
        "logit_identity": "Exact equality of the unchanged suffix on every complete exit with the "
                          "original network logits",
        "membership": [{key: row[key] for key in membership_keys if key in row} for row in records],
        "artifacts": {name: {"sha256": file_hash(output / f"{name}.npy"), "shape": list(value.shape)}
                      for name, value in arrays.items()}
                     | {"logits": {"sha256": file_hash(output / "logits.npy")}},
        "teacher_top1": top1.tolist(), "runtime": feat.runtime_provenance(),
        "seconds": time.monotonic() - started}
    write_json(output / "extraction.json", extraction)
    log(stage="extract", images=count, exits=list(exits), seconds=round(extraction["seconds"], 1))
    return extraction
