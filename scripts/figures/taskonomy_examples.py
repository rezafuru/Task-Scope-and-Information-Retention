#!/usr/bin/env python3
"""Render the decoded Taskonomy examples and compose them into one figure.

The panels show semantic and edge predictions from decoded messages on five
preselected validation images, one per building, for the depth, depth-semantic,
and depth-edge focal codes. The composed figure places both panels side by side
against shared RGB references.

Reads the exported prediction arrays, so it needs neither the dataset nor the
codec checkpoints.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pymupdf

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from taskscope.figures import configure_style, save_figure
from taskscope.paths import FIGURES, RESULTS

CANDIDATES = (("d30", "D", "d_r30"), ("ds3", "DS", "ds_r3"), ("de3", "DE", "de_r3"))
BUILDINGS = ("wiconisco", "corozal", "collierville", "markleeville", "darden")
IGNORED_LABEL = 255


def semantic_colormap() -> tuple[ListedColormap, np.ndarray]:
    colors = np.vstack([np.zeros((1, 3)), plt.get_cmap("tab20")(np.linspace(0, 1, 16))[:, :3]])
    colors[12] = (0.90, 0.0, 0.60)
    colormap = ListedColormap(colors)
    colormap.set_bad("#808080")
    return colormap, colors


def load(evidence: Path, heads: Path):
    arrays = np.load(evidence / "focal_readout_example_predictions.npz")
    metadata = json.loads((evidence / "focal_readout_example_predictions.json").read_text())
    if arrays["keys"].tolist() != metadata["keys"]:
        raise ValueError("Example arrays and their fixed image keys differ")
    evaluations = [json.loads((heads / folder / "evaluation.json").read_text())
                   for _, _, folder in CANDIDATES]
    if any(row["images"] != 500 or row["split"] != "val" for row in evaluations):
        raise ValueError("Example rate labels require the matching 500-image validation results")
    for (prefix, _, _), row in zip(CANDIDATES, evaluations):
        if row["actual_bpp"] != metadata["sources"][prefix]["actual_mean_bpp"]:
            raise ValueError("Example rate and exported readout source differ")
    return arrays, evaluations


def panels(arrays, evaluations, output: Path) -> None:
    colormap, colors = semantic_colormap()
    class_names = evaluations[0]["metrics"]["semantic"]["class_names"]
    configure_style()
    for task in ("edge", "semantic"):
        fields = [task] + [f"{prefix}_{task}" for prefix, _, _ in CANDIDATES]
        labels = ["RGB", f"{task.capitalize()} target"] + [
            f"{family} ({row['actual_bpp']:.3f} bpp)"
            for (_, family, _), row in zip(CANDIDATES, evaluations)
        ]
        figure, axes = plt.subplots(5, 5, figsize=(6.75, 7.2 if task == "semantic" else 6.75))
        for i, key in enumerate(arrays["keys"]):
            axes[i, 0].imshow(arrays["rgb"][i].transpose(1, 2, 0))
            for j, name in enumerate(fields, 1):
                if task == "semantic":
                    axes[i, j].imshow(np.ma.masked_equal(arrays[name][i], IGNORED_LABEL),
                                      cmap=colormap, vmin=0, vmax=16, interpolation="nearest")
                else:
                    axes[i, j].imshow(arrays[name][i, 0], cmap="gray", vmin=0, vmax=1)
            for j in range(5):
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])
                for spine in axes[i, j].spines.values():
                    spine.set_visible(False)
                if i == 0:
                    axes[i, j].set_title(labels[j], fontsize=7.5, pad=4)
            axes[i, 0].set_ylabel(str(key).split("/")[0], fontsize=7, labelpad=2)
        if task == "semantic":
            present = np.unique(np.concatenate([arrays[field].ravel() for field in fields]))
            handles = [Patch(facecolor=colors[index], label=class_names[index].replace("_", " "))
                       for index in present if index != IGNORED_LABEL]
            if IGNORED_LABEL in present:
                handles.append(Patch(facecolor="#808080", label="ignored target"))
            figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.52, 0.005),
                          ncol=5, frameon=False, fontsize=6.8, handlelength=0.8,
                          columnspacing=1.1, labelspacing=0.4)
        figure.subplots_adjust(left=0.043, right=0.998,
                               bottom=0.09 if task == "semantic" else 0.015,
                               top=0.968, wspace=0.025, hspace=0.025)
        save_figure(figure, output / f"paper_taskonomy_{task}_examples")


def image_boxes(page) -> list[pymupdf.Rect]:
    boxes = [pymupdf.Rect(item["bbox"]) for item in page.get_image_info()]
    if len(boxes) != 25:
        raise ValueError("Each panel must contain five rows of five images")
    return sorted(boxes, key=lambda box: (round(box.y0, 1), box.x0))


def legend_entries(page) -> list[pymupdf.Rect]:
    """Crop each semantic class swatch and its label out of the panel as one vector box."""
    entries = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                x0, y0, x1, y1 = span["bbox"]
                if y0 > 480:
                    entries.append(pymupdf.Rect(x0 - 11.5, y0 - 0.5, x1 + 0.5, y1 + 0.5))
    if len(entries) != 14:
        raise ValueError(f"Expected all fourteen semantic legend entries, found {len(entries)}")
    return entries


def compose(output: Path) -> Path:
    """Place the two panels in one figure sharing the RGB and target columns."""
    target = output / "paper_taskonomy_joint_examples.pdf"
    with (
        pymupdf.open(output / "paper_taskonomy_semantic_examples.pdf") as semantic,
        pymupdf.open(output / "paper_taskonomy_edge_examples.pdf") as edge,
        pymupdf.open() as document,
    ):
        semantic_boxes = image_boxes(semantic[0])
        edge_boxes = image_boxes(edge[0])
        width, left, gap = 486.0, 14.0, 1.4
        side = (width - left - 8 * gap) / 9
        top = 27.0
        legend_top = top + 5 * (side + gap) + 4
        page = document.new_page(width=width, height=legend_top + 25)
        page.insert_textbox(pymupdf.Rect(left + side + gap, 0, left + 5 * (side + gap), 14),
                            "Semantic predictions", fontsize=8.3, fontname="helv", align=1)
        page.insert_textbox(pymupdf.Rect(left + 5 * (side + gap), 0, width, 14),
                            "Edge predictions", fontsize=8.3, fontname="helv", align=1)
        for column, label in enumerate(("RGB", "Target", "D", "DS", "DE",
                                        "Target", "D", "DS", "DE")):
            x = left + column * (side + gap)
            label_width = pymupdf.get_text_length(label, fontname="helv", fontsize=8.3)
            page.insert_text((x + (side - label_width) / 2, 22), label,
                             fontsize=8.3, fontname="helv")
        for row, building in enumerate(BUILDINGS):
            y = top + row * (side + gap)
            page.insert_text((9, y + side - 1), building, fontsize=7.8, rotate=90)
            for column in range(9):
                source = semantic if column < 5 else edge
                source_column = column if column < 5 else column - 4
                boxes = semantic_boxes if column < 5 else edge_boxes
                x = left + column * (side + gap)
                page.show_pdf_page(pymupdf.Rect(x, y, x + side, y + side), source, 0,
                                   clip=boxes[5 * row + source_column])
        entries = legend_entries(semantic[0])
        scale, separation = 8.3 / 6.8, 10.0
        for row in range(2):
            group = entries[7 * row:7 * (row + 1)]
            row_width = sum(box.width * scale for box in group) + 6 * separation
            x = (width - row_width) / 2
            y = legend_top + 12 * row
            for box in group:
                placed = pymupdf.Rect(x, y, x + box.width * scale, y + box.height * scale)
                page.show_pdf_page(placed, semantic, 0, clip=box)
                x = placed.x1 + separation
        document.save(target, garbage=4, deflate=True)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=RESULTS / "taskonomy/examples")
    parser.add_argument("--heads", type=Path, default=RESULTS / "taskonomy/available_heads")
    parser.add_argument("--output", type=Path, default=FIGURES)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    arrays, evaluations = load(arguments.evidence, arguments.heads)
    panels(arrays, evaluations, arguments.output)
    print(compose(arguments.output))


if __name__ == "__main__":
    main()
