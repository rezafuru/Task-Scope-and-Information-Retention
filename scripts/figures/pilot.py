#!/usr/bin/env python3
"""Render the motivating feature-preservation figure.

Two species rows show the original photograph and the RGB inversions of the
ImageNet ResNet-50 residual blocks 13 to 16, each with the annotated wing region
boxed in the frame and enlarged in the strip beneath it. The plot gives
species-pair accuracy from readers fitted on the complete features at each exit
and from readers fitted on the reconstructions, two fits per observation.

Marks follow the retained protocol: the line and open marker give the arithmetic
mean of the two fits, the small filled markers the individual fits offset by
0.07 blocks. These are fitted-reader accuracies, not a probability ensemble, so
they carry no intervals.

The canvas is fixed at 6.5 by 2.02 inches and every element is placed in inches,
so the figure keeps its proportions at the manuscript width.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from PIL import Image

from taskscope.figures import save_figure
from taskscope.paths import ASSETS, FIGURES, REPO_ROOT, RESULTS

EVIDENCE = RESULTS / "pilot/pair/asset_evidence.json"
INVENTORY = RESULTS / "pilot/pair/sources.json"
EXITS = ("layer3", "block1", "block2", "block3")
BLOCKS = (13, 14, 15, 16)
PHOTOGRAPHS = ((754, "Indigo Bunting"), (3092, "Blue Grosbeak"))
VIEWS = (("original", "Original"), ("layer3", "Block 13"),
         ("block1", "Block 14"), ("block2", "Block 15"), ("block3", "Block 16"))
WIDTH, HEIGHT = 6.5, 2.02
COLORS = {"Native feature": "#2C786C", "Reconstructed RGB": "#8B4976"}
MARKERS = {"Native feature": "o", "Reconstructed RGB": "s"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def readouts(original: dict) -> dict:
    """Recompute both fits per observation from the retained confusion counts."""
    measurements = {}
    for name, prefix in (("Native feature", ""), ("Reconstructed RGB", "inverse_")):
        values = []
        for exit_name in EXITS:
            candidates = original[prefix + exit_name]["candidates"]
            if len(candidates) != 2:
                raise ValueError("Require the two fitted readers per observation")
            row = []
            for candidate in candidates:
                metric = candidate["restricted"]
                if metric["counts"] != [30, 30]:
                    raise ValueError("Pair assessment requires 30 photographs per species")
                confusion = np.asarray(metric["confusion"])
                if confusion.shape != (2, 2) or not np.array_equal(confusion.sum(1), [30, 30]):
                    raise ValueError("Confusion counts disagree with the assessment population")
                accuracy = confusion.trace() / 60
                if not np.isclose(accuracy, metric["macro_accuracy"], rtol=0, atol=1e-12):
                    raise ValueError("Accuracy disagrees with the source confusion counts")
                row.append(100 * accuracy)
            values.append(row)
        measurements[name] = {"per_fit_percent": values,
                              "mean_percent": np.mean(values, axis=1).tolist()}
    return measurements


def collect() -> tuple[dict, dict]:
    """Recheck the retained evidence and the illustrations, then read both off."""
    evidence = json.loads(EVIDENCE.read_text())
    evaluation = REPO_ROOT / evidence["evaluation_source"]["path"]
    if sha(evaluation) != evidence["evaluation_source"]["sha256"]:
        raise ValueError("Retained evaluation differs from the verified evidence")
    original = json.loads(evaluation.read_text())["cub_test"]["restricted_200way"]
    if original != evidence["restricted_200way_test"]:
        raise ValueError("Copied accuracy evidence differs from its source")
    measurements = readouts(original)
    cue_rows = json.loads((REPO_ROOT / evidence["cue_measurement_source"]).read_text())
    boxes = {str(ident): next(row["wing_box"] for row in cue_rows if row["cub_image_id"] == ident)
             for ident, _ in PHOTOGRAPHS}
    images = {}
    for record in evidence["verified_images"]:
        path = REPO_ROOT / record["path"]
        if sha(path) != record["sha256"] or record["split"] not in ("train", "val"):
            raise ValueError("Illustration source changed or is not a development photograph")
        with Image.open(path) as source:
            if source.size != (224, 224) or source.mode != "RGB":
                raise ValueError("Require the unchanged 224-pixel RGB exports")
            images[(record["cub_image_id"], record["view"])] = source.copy()
    inventory = json.loads(INVENTORY.read_text())
    illustrations = []
    for ident, _ in PHOTOGRAPHS:
        record = inventory[str(ident)]
        if record["split"] not in ("train", "val"):
            raise ValueError("Illustrations must use development photographs")
        for view, _ in VIEWS:
            path = ASSETS / "pilot" / f"cub{ident}_{view}.png"
            digest = sha(path)
            if digest != record["files"][view]["sha256"]:
                raise ValueError(f"Illustration differs from its source inventory: {path}")
            with Image.open(path) as source:
                if source.size != (224, 224) or source.mode != "RGB":
                    raise ValueError("Require the unchanged 224-pixel RGB exports")
                if (ident, view) in images and not np.array_equal(source, images[(ident, view)]):
                    raise ValueError("The expanded illustration differs from the verified images")
                images[(ident, view)] = source.copy()
            illustrations.append({"path": str(path.relative_to(REPO_ROOT)), "sha256": digest,
                                  "cub_image_id": ident, "split": record["split"], "view": view})
    numbers = {
        "evidence_sha256": sha(EVIDENCE),
        "evaluation_source": evidence["evaluation_source"],
        "blocks": list(BLOCKS),
        "species": [name for _, name in PHOTOGRAPHS],
        "test_photographs_per_species": 30,
        "fitted_readers_per_observation": 2,
        "readouts": measurements,
        "statistic": "Individual fitted-reader accuracies and their arithmetic mean, "
                     "not a probability ensemble",
        "accuracy_marks": {"filled": "Individual fitted readers, offset horizontally by 0.07 blocks",
                           "open": "Arithmetic mean at the stated block",
                           "line": "Arithmetic means"},
        "imagenet_native_decision_agreement_percent": 100,
        "imagenet_agreement_basis": "Unchanged suffix on complete feature, separate from species "
                                    "accuracy and reconstructed-RGB predictions",
        "illustration_images": illustrations,
        "boxes_xyxy_224": boxes,
        "illustration_inventory": {"path": str(INVENTORY.relative_to(REPO_ROOT)),
                                   "sha256": sha(INVENTORY)},
        "context_crop_xyxy_224": [0, 0, 224, 224],
        "figure_size_inches": [WIDTH, HEIGHT],
        "cue_color_curve": "Not displayed",
    }
    return images, numbers


def axes_inches(figure, x, y, width, height):
    return figure.add_axes([x / WIDTH, y / HEIGHT, width / WIDTH, height / HEIGHT])


def configure_style() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "font.weight": "normal", "axes.labelsize": 8,
                         "axes.titlesize": 8, "pdf.fonttype": 42,
                         "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                         "ytick.major.width": 0.6, "xtick.major.size": 3,
                         "ytick.major.size": 3})


def illustration(figure, images, numbers) -> None:
    """Place the column headings, both species rows, and their wing strips."""
    frame, pitch, left_edge = 0.59, 0.615, 0.16
    for column, (_, title) in enumerate(VIEWS):
        figure.text((left_edge + column * pitch + frame / 2) / WIDTH, 1.95 / HEIGHT,
                    title, ha="center", va="center", fontsize=8)
    for index, (ident, name) in enumerate(PHOTOGRAPHS):
        main_bottom, strip_bottom = ((1.30, 0.985), (0.335, 0.020))[index]
        figure.text(0.053 / WIDTH, (strip_bottom + (frame + 0.315) / 2) / HEIGHT,
                    name, ha="center", va="center", rotation=90, fontsize=8)
        box = numbers["boxes_xyxy_224"][str(ident)]
        x0, y0, x1, y1 = box
        for column, (view, _) in enumerate(VIEWS):
            left = left_edge + column * pitch
            image = images[(ident, view)]
            main = axes_inches(figure, left, main_bottom, frame, frame)
            main.imshow(image, interpolation="nearest")
            main.add_patch(Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0, y1 - y0,
                                     fill=False, edgecolor="white", linewidth=0.65))
            main.set_axis_off()
            detail = axes_inches(figure, left, strip_bottom, frame, frame / 2)
            detail.imshow(image.crop(box), interpolation="nearest")
            detail.set_xticks([])
            detail.set_yticks([])
            for spine in detail.spines.values():
                spine.set_linewidth(0.5)
                spine.set_color("#333333")


def accuracy(figure, numbers) -> None:
    """Draw the two-fit species-pair accuracies against the residual block.

    The axes sit 0.03 inch above the position used for the manuscript render.
    Tick and label metrics grew after matplotlib 3.9, and at the original height
    the axis label fell off the fixed canvas, which has no tight bounding box to
    absorb it. The extra margin keeps the label inside across the supported range.
    """
    plot = axes_inches(figure, 3.69, 0.36, 2.72, 1.27)
    handles = []
    for name, measurements in numbers["readouts"].items():
        color, marker = COLORS[name], MARKERS[name]
        values = np.asarray(measurements["per_fit_percent"])
        plot.plot(BLOCKS, measurements["mean_percent"], color=color, linewidth=1.1,
                  marker=marker, markersize=4.3, markerfacecolor="white",
                  markeredgewidth=0.8, zorder=2)
        for fit, offset in enumerate((-0.07, 0.07)):
            plot.scatter(np.asarray(BLOCKS) + offset, values[:, fit], s=7, color=color,
                         marker=marker, edgecolor="white", linewidth=0.25, zorder=3)
        handles.append(Line2D([], [], color=color, marker=marker, markersize=4.3,
                              markerfacecolor="white", markeredgewidth=0.8,
                              linewidth=1.1, label=name))
    plot.set(xlim=(12.82, 16.18), ylim=(78, 100), xticks=BLOCKS,
             yticks=(80, 85, 90, 95, 100), xlabel="Residual block",
             ylabel="Species-pair accuracy (%)")
    plot.spines[["top", "right"]].set_visible(False)
    plot.tick_params(labelsize=8, pad=2)
    plot.xaxis.labelpad = 4
    plot.yaxis.labelpad = 3
    plot.grid(axis="y", color="#e4e4e4", linewidth=0.4, zorder=0)
    plot.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2,
                frameon=False, borderaxespad=0, handlelength=1.4, columnspacing=0.8,
                handletextpad=0.4, fontsize=8)


def check_text_within_canvas(figure) -> None:
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    canvas = figure.bbox
    outside = []
    for text in figure.findobj(match=matplotlib.text.Text):
        if not (text.get_visible() and text.get_text()):
            continue
        box = text.get_window_extent(renderer)
        if (box.x0 < canvas.x0 - 1 or box.y0 < canvas.y0 - 1
                or box.x1 > canvas.x1 + 1 or box.y1 > canvas.y1 + 1):
            outside.append(text.get_text())
    if outside:
        raise ValueError(f"Text outside canvas: {outside}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=FIGURES)
    arguments = parser.parse_args()
    images, numbers = collect()
    configure_style()
    figure = plt.figure(figsize=(WIDTH, HEIGHT), facecolor="white")
    illustration(figure, images, numbers)
    figure.text(4.94 / WIDTH, 1.95 / HEIGHT, "Native ImageNet decisions unchanged",
                ha="center", va="center", fontsize=8)
    accuracy(figure, numbers)
    check_text_within_canvas(figure)
    numbers["text_within_canvas"] = True
    arguments.output.mkdir(parents=True, exist_ok=True)
    base = arguments.output / "paper_pilot_bunting_grosbeak"
    save_figure(figure, base)
    base.with_suffix(".json").write_text(json.dumps(numbers, indent=2, allow_nan=False) + "\n")
    print(base.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
