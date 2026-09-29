#!/usr/bin/env python3
"""Render the learned block-coding figures and the retained measurement table.

The main figure compares complete rates, conditional coding errors, and the mean
rate saving under corrupted confidence. The sensitivity figure compares block
lengths, the added-task tolerance sweep, and the three-state families.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from taskscope.figures import FULL_COLOR, REDUCED_COLOR, crop_to_ink, render_preview, write_pdf
from taskscope.paths import FIGURES, RESULTS

COLORS = {"full": FULL_COLOR, "reduced": REDUCED_COLOR}
SEEDS = [11, 12, 13]


def rows_for(rows: list[dict], condition: str, length: int = 16,
             access: str = "full", method: str = "learned") -> list[dict]:
    if access == "reduced":
        method = "reference"
    selected = sorted(
        [row for row in rows if row["condition"] == condition and row["length"] == length
         and row["access"] == access and row["method"] == method],
        key=lambda row: row["seed"],
    )
    if [row["seed"] for row in selected] != SEEDS:
        raise ValueError(f"Expected fits {SEEDS} for {condition}, {length}, {access}, {method}")
    return selected


def points(axis: Axes, x: float, values: Sequence[float], color: str,
           marker: str = "o", label: str | None = None, spread: float = 0.025) -> None:
    """Separate fitted observations horizontally without joining their values."""
    values = np.asarray(values)
    axis.scatter(x + np.linspace(-spread, spread, len(values)), values,
                 c=color, marker=marker, s=5.5, linewidths=0, label=label, zorder=4)


def optimum(axis: Axes, x: float, value: float, color: str, size: float = 17) -> None:
    axis.scatter([x], [value], marker="D", s=size, facecolors="white",
                 edgecolors=color, linewidths=0.8, zorder=5)


def style(axis: Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#dddddd", lw=0.5)
    axis.set_axisbelow(True)
    axis.tick_params(axis="x", length=2, pad=3)
    axis.tick_params(axis="y", length=2, pad=2)


def configure_style() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.3,
                         "axes.titlesize": 8.3, "axes.labelsize": 8.3,
                         "xtick.labelsize": 8.3, "ytick.labelsize": 8.3,
                         "legend.fontsize": 8.0, "pdf.fonttype": 42})


def save(figure: Figure, base: Path) -> None:
    path = write_pdf(figure, base, pad_inches=0.025)
    crop_to_ink(path, threshold=250, grayscale=True)
    render_preview(path, base.with_suffix(".png"), dpi=300)


def load(source: Path) -> tuple[list[dict], dict]:
    rows = json.loads((source / "summary.json").read_text())
    failures = [row for row in rows if not row.get("feasible_validation", False)
                or not row.get("assessed_feasible", False)]
    if failures:
        examples = [(r["condition"], r["seed"], r["length"], r["access"], r["method"])
                    for r in failures[:8]]
        raise ValueError("A feasible-rate figure requires explicit failure markings for these "
                         f"selections: {examples}")
    predictions = {r["name"]: r for r in json.loads((source / "predictions.json").read_text())}
    return rows, predictions


def sensitivity_figure(rows, predictions, base: Path) -> None:
    configure_style()
    height = 1.95
    figure, axes = plt.subplots(1, 3, figsize=(6.5, height))
    for access in ["full", "reduced"]:
        for index, length in enumerate([12, 16]):
            selected = rows_for(rows, "two_tau_0_single", length=length, access=access)
            ideal = predictions["two_tau_0_single"]["exact"][f"{access}_rate"]
            points(axes[0], index + (-0.15 if access == "full" else 0.15),
                   [r["complete_rate"] - ideal for r in selected], COLORS[access], spread=0.045)
    axes[0].set(title="(a) Finite-code excess", xlabel="Block length",
                ylabel="Excess bits per decision", xticks=[0, 1], xticklabels=["12", "16"],
                xlim=(-0.4, 1.4), ylim=(0.12, 0.195), yticks=[0.12, 0.14, 0.16, 0.18])
    for access in ["full", "reduced"]:
        names = ["pair", "tolerance_034", "tolerance_038", "tolerance_042", "loose"]
        shift = -0.01 if access == "full" else 0.01
        for tolerance, name in zip([0.30, 0.34, 0.38, 0.42, 0.46], names):
            selected = rows_for(rows, f"two_tau_0_{name}", access=access)
            points(axes[1], tolerance + shift, [r["complete_rate"] for r in selected],
                   COLORS[access], spread=0.0045)
            optimum(axes[1], tolerance + shift,
                    predictions[f"two_tau_0_{name}"]["exact"][f"{access}_rate"], COLORS[access])
    axes[1].set(title="(b) Added-task tolerance", xlabel="Second-task tolerated risk",
                ylabel="Bits per decision", xlim=(0.276, 0.484), ylim=(0.30, 0.75),
                xticks=[0.30, 0.38, 0.46], yticks=[0.3, 0.4, 0.5, 0.6, 0.7])
    for access in ["full", "reduced"]:
        for index, family in enumerate(["single", "partial", "cyclic"]):
            selected = rows_for(rows, f"three_{family}", access=access)
            shift = -0.17 if access == "full" else 0.17
            points(axes[2], index + shift, [r["complete_rate"] for r in selected],
                   COLORS[access], spread=0.08)
            optimum(axes[2], index + shift,
                    predictions[f"three_{family}"]["exact"][f"{access}_rate"], COLORS[access])
    axes[2].set(title="(c) Three confidence states", ylabel="Bits per decision",
                xticks=[0, 1, 2], xticklabels=["Single", "Partial\npair", "Cyclic\ntriple"],
                xlim=(-0.45, 2.45), ylim=(0.30, 0.75), yticks=[0.3, 0.4, 0.5, 0.6, 0.7])
    for axis in axes:
        style(axis)
        axis.tick_params(axis="x", length=2, pad=3)
        axis.tick_params(axis="y", length=2, pad=2)
    handles = [Line2D([], [], color=COLORS[access], marker="o", linestyle="none",
                      markersize=3, label=label)
               for access, label in [("full", "Fitted, with confidence"),
                                     ("reduced", "Fitted, without confidence")]]
    handles.append(Line2D([], [], color="#333333", marker="D", markerfacecolor="white",
                          linestyle="none", markersize=4, label="Asymptotic optimum"))
    figure.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.52, 1.015),
                  ncol=3, handletextpad=0.4, columnspacing=1.0, borderaxespad=0)
    figure.subplots_adjust(left=0.068, right=0.995, bottom=0.4325 / height,
                           top=1.5975 / height, wspace=0.55)
    save(figure, base)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=RESULTS / "block_coding")
    parser.add_argument("--output", type=Path, default=FIGURES)
    arguments = parser.parse_args()
    rows, predictions = load(arguments.source)
    arguments.output.mkdir(parents=True, exist_ok=True)
    sensitivity_figure(rows, predictions, arguments.output / "paper_block_coding_sensitivity")
    print(arguments.output / "paper_block_coding_sensitivity.pdf")


if __name__ == "__main__":
    main()
