#!/usr/bin/env python3
"""Render the three-panel exact-rate figure.

Panel (a) shows the reduced observation's risk floor for the overlapping
threshold family. Panels (b, c) show one task and the complementary pair under
shared Bayes predictions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from taskscope import exact
from taskscope.figures import FULL_COLOR, REDUCED_COLOR, crop_to_ink, manuscript_previews, write_pdf
from taskscope.paths import FIGURES, RESULTS


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.2,
        "axes.titlesize": 8.5, "axes.titleweight": "normal",
        "axes.labelsize": 8.2, "xtick.labelsize": 8.2, "ytick.labelsize": 8.2,
        "legend.fontsize": 8.2, "axes.linewidth": 0.65,
        "xtick.major.width": 0.65, "ytick.major.width": 0.65,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "pdf.fonttype": 42, "savefig.facecolor": "white",
    })


def draw(curves: dict[str, np.ndarray]) -> plt.Figure:
    configure_style()
    figure, panels = plt.subplots(1, 3, figsize=(6.5, 1.98))
    figure.subplots_adjust(left=0.065, right=0.993, bottom=0.235, top=0.80, wspace=0.16)
    full_line = {"color": FULL_COLOR, "linewidth": 1.45}
    reduced_line = {"color": REDUCED_COLOR, "linewidth": 1.45, "linestyle": (0, (4, 2.2))}
    figure.legend(
        handles=[Line2D([], [], label="Full observation $X$", **full_line),
                 Line2D([], [], label="Reduced observation $Y$", **reduced_line)],
        loc="upper center", bbox_to_anchor=(0.54, 1.015), ncol=2,
        frameon=False, handlelength=2.2, columnspacing=1.6,
    )
    for panel in panels:
        panel.spines[["top", "right"]].set_visible(False)
        panel.grid(axis="y", color="#e5e5e5", linewidth=0.5)
        panel.set_axisbelow(True)
        panel.tick_params(length=3, pad=2)
    panels[0].set(ylim=(0.88, 1.52), yticks=[1.0, 1.2, 1.4])
    for panel in panels[1:]:
        panel.set(ylim=(-0.025, 1.04), yticks=[0.0, 0.5, 1.0])
    panels[2].sharey(panels[1])

    threshold = curves["overlapping_thresholds"]
    panels[0].plot(threshold[:, 0], threshold[:, 1], **full_line)
    panels[0].plot(threshold[:, 0], threshold[:, 2], **reduced_line)
    panels[0].axvline(0.20, color=REDUCED_COLOR, linewidth=0.8, linestyle=(0, (1, 2)))
    panels[0].text(0.067, 1.435, "$Y$ infeasible", color="#222222", fontsize=8.2)
    panels[0].set(title="(a) Observation limit", xlabel="Additional-task error",
                  ylabel="Rate (bits/sample)", xlim=(0.05, 0.25),
                  xticks=[0.05, 0.10, 0.15, 0.20, 0.25])

    settings = [
        ("variable_confidence", "(b) One task", "Bayes prediction $Z$"),
        ("complementary_confidence_family", "(c) Complementary pair", "Bayes tuple $(Z,Z)$"),
    ]
    for panel, (name, title, context) in zip(panels[1:], settings):
        values = curves[name]
        panel.plot(values[:, 0], values[:, 1], **full_line)
        panel.plot(values[:, 0], values[:, 2], **reduced_line)
        panel.set(title=title, xlim=(0.25, 0.50), xticks=[0.25, 0.30, 0.40, 0.50])
        panel.text(0.274, 0.88, context, fontsize=8.2)
        selected = np.flatnonzero(np.isclose(values[:, 0], 0.30, rtol=0, atol=1e-10))
        if len(selected) != 1:
            raise ValueError(f"Expected one exact D=0.30 row for {name}, found {len(selected)}")
        full_rate, reduced_rate = values[selected[0], 1:]
        if name == "variable_confidence":
            panel.plot(0.30, full_rate, marker="o", markersize=3.4,
                       markerfacecolor="white", color=FULL_COLOR)
            panel.text(0.26, 0.17, f"{full_rate:.3f}", color="#222222", fontsize=8.2)
        panel.plot(0.30, reduced_rate, marker="o", markersize=3.4,
                   markerfacecolor="white", color=REDUCED_COLOR)
        panel.text(0.316, reduced_rate + 0.032, f"{reduced_rate:.3f}",
                   color="#222222", fontsize=8.2)

    panels[1].set_xlabel("Task error $D$")
    panels[2].set_xlabel("Both task errors $D$")
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curves", type=Path, default=RESULTS / "exact/rate_curves.csv")
    parser.add_argument("--output", type=Path, default=FIGURES,
                        help="directory the figure is written to")
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    path = write_pdf(draw(exact.load_curves(arguments.curves)),
                     arguments.output / "paper_task_scope_limits.pdf", pad_inches=0.02)
    crop_to_ink(path)
    manuscript_previews(path)
    print(path)


if __name__ == "__main__":
    main()
