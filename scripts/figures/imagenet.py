#!/usr/bin/env python3
"""Render the ImageNet ENTITY-30 task-scope figure from the frozen evidence.

Two panels place the coarse and fine accuracy drops of every fitted candidate group
against their allowances. Open markers are validation means, filled markers are the
held-out means of the groups the validation selection fixed, small dots are the two
codec fits behind each mean, and bars are conditional 95% paired-image intervals. The
row labels carry each group's fitting loss and its mean complete validation rate.

The analysis is recomputed from the sealed per-image predictions on every run and
written beside the figure, so the drawn numbers and the reported numbers are the same
object. It reads only the retained evidence: no dataset, no checkpoints, no GPU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from taskscope.figures import PLUM, SLATE, TEAL, render_preview, write_pdf
from taskscope.imagenet.analysis import (
    BOOTSTRAP_REPETITIONS,
    BOOTSTRAP_SEED,
    EVIDENCE,
    SELECTION,
    TEST_DIRECTORIES,
    analyze,
)
from taskscope.paths import FIGURES

COLORS = {"coarse": TEAL, "added_fine": PLUM}
MARKERS = {"coarse": "o", "added_fine": "s"}
LABELS = {"coarse": "Coarse", "added_fine": "+ fine"}
# Greys with no name in the shared palette: row text, allowance dashes, row rules.
TEXT_GREY = "#222222"
ALLOWANCE_GREY = "#777777"
ROW_RULE = "#E9E7E8"
PANEL_TITLES = ("(a) Coarse labels", "(b) Fine labels")
FIRST_TICK = (1, 3)


def draw(analysis: dict) -> plt.Figure:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.titlesize": 8.5, "axes.labelsize": 8,
                         "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    figure = plt.figure(figsize=(6.5, 1.95))
    panels = [figure.add_axes((0.23, 0.255, 0.35, 0.615)),
              figure.add_axes((0.645, 0.255, 0.35, 0.615))]
    groups = sorted(analysis["groups"], key=lambda group: group["validation"]["actual_bpp"])
    for column, panel in enumerate(panels):
        panel.axvline(analysis["allowance_pp"][column], color=ALLOWANCE_GREY,
                      linestyle=(0, (3, 2)), linewidth=0.8, zorder=0)
        for row, group in enumerate(groups):
            color, marker = COLORS[group["objective"]], MARKERS[group["objective"]]
            panel.axhline(row, color=ROW_RULE, linewidth=0.5, zorder=0)
            panel.scatter(group["validation"]["drop_pp"][column], row - 0.15, s=23,
                          marker=marker, facecolors="white", edgecolors=color,
                          linewidths=0.9, zorder=3)
            if "test" not in group:
                continue
            test = group["test"]
            lower, upper = test["drop_pointwise_95_interval_pp"][column]
            panel.hlines(row + 0.15, lower, upper, color=color, linewidth=0.9, zorder=2)
            panel.plot([lower, upper], [row + 0.15] * 2, marker="|", color=color,
                       markersize=4, linestyle="none", markeredgewidth=0.85)
            panel.scatter([fit["drop_pp"][column] for fit in test["fits"]],
                          row + 0.15 + np.linspace(-0.2, 0.2, len(test["fits"])),
                          s=9, marker=".", color=color, zorder=4)
            panel.scatter(test["drop_pp"][column], row + 0.15, s=23, marker=marker,
                          facecolors=color, edgecolors=color, linewidths=0.8, zorder=5)
        first_tick = FIRST_TICK[column]
        panel.set(ylim=(len(groups) - 0.5, -0.5), yticks=[],
                  xlim=(first_tick - 0.15, first_tick + 3.15),
                  xticks=np.arange(first_tick, first_tick + 4))
        position = panel.get_position()
        figure.text((position.x0 + position.x1) / 2, 0.97, PANEL_TITLES[column],
                    ha="center", va="top", fontsize=8.5)
        panel.spines[["top", "right", "left"]].set_visible(False)
        panel.tick_params(direction="out", length=2.5, width=0.6, pad=2)
    for row, group in enumerate(groups):
        y = figure.transFigure.inverted().transform(panels[0].transData.transform((0, row)))[1]
        figure.text(0.005, y, LABELS[group["objective"]], color=TEXT_GREY, va="center")
        figure.text(0.195, y, f"{group['validation']['actual_bpp']:.3f}", va="center", ha="right")
    figure.text(0.005, 0.97, "Fitting loss", va="top", fontsize=8.5)
    figure.text(0.195, 0.97, "bpp", va="top", ha="right", fontsize=8.5)
    handles = [Line2D([], [], color=SLATE, marker="o", markerfacecolor=face,
                      linestyle="none", markersize=4, label=label)
               for face, label in (("white", "Validation"), (SLATE, "Test"))]
    handles.append(Line2D([], [], color=ALLOWANCE_GREY, linestyle=(0, (3, 2)),
                          linewidth=0.8, label="Allowance"))
    figure.legend(handles=handles, loc="lower left", bbox_to_anchor=(-0.004, 0.015),
                  ncol=3, frameon=False, columnspacing=0.8, handletextpad=0.35,
                  handlelength=1.1, borderaxespad=0, fontsize=8)
    figure.text(0.7, 0.055, "Accuracy drop (percentage points)", ha="center", va="center")
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selection", type=Path, default=SELECTION,
                        help="frozen validation selection that fixed the assessed groups")
    parser.add_argument("--data-root", type=Path, default=EVIDENCE,
                        help="directory holding the candidate and assessment records")
    parser.add_argument("--tests", type=Path, nargs="+", default=None,
                        help="assessment directories, resolved against --data-root when relative")
    parser.add_argument("--output", type=Path, default=FIGURES,
                        help="directory that receives the figure")
    parser.add_argument("--bootstrap-repetitions", type=int, default=BOOTSTRAP_REPETITIONS)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    arguments = parser.parse_args()
    names = arguments.tests if arguments.tests else [Path(name) for name in TEST_DIRECTORIES]
    tests = [name if name.is_absolute() else arguments.data_root / name for name in names]
    analysis = analyze(arguments.selection, arguments.data_root, tests,
                       arguments.bootstrap_repetitions, arguments.bootstrap_seed)
    arguments.output.mkdir(parents=True, exist_ok=True)
    base = arguments.output / "paper_imagenet_task_scope"
    path = write_pdf(draw(analysis), base, pad_inches=0.015)
    render_preview(path, base.with_suffix(".png"), dpi=220)
    render_preview(path, base.with_name(base.name + "_manuscript_size.png"), dpi=100)
    base.with_suffix(".json").write_text(json.dumps(analysis, indent=2, allow_nan=False) + "\n")
    print(path)


if __name__ == "__main__":
    main()
