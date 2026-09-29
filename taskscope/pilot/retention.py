"""Wing regions located from the CUB part annotation, with the cue colour measured inside them.

Two 96 by 48 boxes are placed on every development photograph of the pair. The wing box is
centred on the mean of the visible wing parts, the body box on the mean of breast, back and
belly, and both are clamped to the 224 by 224 frame. The figure enlarges the wing box, using the
same coordinates in the photograph and in the inversion at each exit.

Inside each box the record also carries the percentage of pixels that are chestnut (HSV hue 8 to
40 degrees) and the percentage that are blue (hue 190 to 260), in both cases with saturation at
least 0.30 and value at least 0.20. Those percentages stay in the record. No colour curve is
reported, and the figure reads the wing boxes only.

Part locations come from the CUB annotation, mapped into the prepared 224 frame with the square
crop recorded for that photograph.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from taskscope.pilot.common import read_json, write_json

EXITS = ("layer3", "block1", "block2", "block3")
BOX_W, BOX_H = 96, 48
WING_PARTS = ("left wing", "right wing")
BODY_PARTS = ("breast", "back", "belly")
SATURATION_FLOOR, VALUE_FLOOR = 0.30, 0.20
CHESTNUT_HUE = (8, 40)
BLUE_HUE = (190, 260)
SPECIES = {13: "Indigo Bunting", 53: "Blue Grosbeak"}


def fractions(image: Image.Image, box) -> tuple[float, float]:
    """Percentage of chestnut and of blue pixels inside one box."""
    x0, y0, x1, y1 = box
    hsv = np.asarray(image.convert("RGB").crop((x0, y0, x1, y1)).convert("HSV")).astype(float)
    hue, saturation, value = hsv[..., 0] * 360 / 255, hsv[..., 1] / 255, hsv[..., 2] / 255
    eligible = (saturation >= SATURATION_FLOOR) & (value >= VALUE_FLOOR)
    chestnut = eligible & (hue >= CHESTNUT_HUE[0]) & (hue <= CHESTNUT_HUE[1])
    blue = eligible & (hue >= BLUE_HUE[0]) & (hue <= BLUE_HUE[1])
    return float(100 * chestnut.mean()), float(100 * blue.mean())


def box_from(points) -> list[int]:
    """Box of the fixed size centred on the mean of ``points``, clamped to the 224 frame."""
    cx, cy = np.mean(np.array(points), axis=0)
    x0 = int(round(min(max(cx - BOX_W / 2, 0), 224 - BOX_W)))
    y0 = int(round(min(max(cy - BOX_H / 2, 0), 224 - BOX_H)))
    return [x0, y0, x0 + BOX_W, y0 + BOX_H]


def part_locations(cub_root: Path, rows: list[dict]) -> dict[int, dict[str, list[float]]]:
    """Visible CUB part locations of the selected photographs, in prepared 224 coordinates."""
    meta = Path(cub_root) / "CUB_200_2011"
    names = {int(line.split()[0]): " ".join(line.split()[1:])
             for line in (meta / "parts/parts.txt").read_text().splitlines()}
    by_image = {row["cub_image_id"]: row for row in rows}
    located: dict[int, dict[str, list[float]]] = {key: {} for key in by_image}
    for line in (meta / "parts/part_locs.txt").read_text().splitlines():
        image, part, x, y, visible = line.split()
        image = int(image)
        if image in located and int(visible) == 1:
            x0, y0, x1, _ = by_image[image]["square_source_xyxy"]
            scale = 224 / (x1 - x0)
            located[image][names[int(part)]] = [(float(x) - x0) * scale, (float(y) - y0) * scale]
    return located


def measure(output: Path, *, inventory: Path, cub_root: Path, inversions: Path) -> list[dict]:
    """Write one record per development photograph of the pair.

    ``inversions`` holds one subdirectory per exit, each with the inventory written by the
    inversion stage.
    """
    development = read_json(inventory)["records"]
    exits = {exit_name: {row["observation_id"]: row
                         for row in read_json(Path(inversions) / exit_name / "inventory.json")["records"]}
             for exit_name in EXITS}
    rows = [row for row in development if row["fine_label"] in SPECIES]
    located = part_locations(cub_root, rows)
    records = []
    for row in rows:
        parts = located[row["cub_image_id"]]
        wing_points = [parts[key] for key in WING_PARTS if key in parts]
        body_points = [parts[key] for key in BODY_PARTS if key in parts]
        if not wing_points or not body_points:
            continue
        wing, body = box_from(wing_points), box_from(body_points)
        entry = {"cub_image_id": row["cub_image_id"], "fine_label": row["fine_label"],
                 "class": SPECIES[row["fine_label"]], "split": row["split"],
                 "wing_box": wing, "body_box": body, "values": {}}
        images = {"original": Image.open(Path(cub_root) / row["input_path"])}
        for exit_name in EXITS:
            record = exits[exit_name][row["observation_id"]]
            images[exit_name] = Image.open(Path(inversions) / exit_name / record["input_path"])
        for name, image in images.items():
            wing_chestnut, wing_blue = fractions(image, wing)
            body_chestnut, body_blue = fractions(image, body)
            entry["values"][name] = {"wing_chestnut": wing_chestnut, "wing_blue": wing_blue,
                                     "body_chestnut": body_chestnut, "body_blue": body_blue}
            image.close()
        records.append(entry)
    write_json(output, records, indent=1)
    return records


def summary(records: list[dict]) -> dict:
    """Mean percentage per class and exit for the two cues, printed by the stage script."""
    keys = ("original", *EXITS)
    out = {}
    for name in SPECIES.values():
        subset = [row for row in records if row["class"] == name]
        out[name] = {"n": len(subset),
                     **{key: {exit_name: float(np.mean([row["values"][exit_name][key] for row in subset]))
                              for exit_name in keys} for key in ("wing_chestnut", "body_blue")}}
    return out
