"""Figure style and PDF export helpers shared by the figure scripts.

Exported PDFs carry a fixed creation date so that repeated runs of the same
figure produce identical files.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pymupdf

FIXED_PDF_DATE = datetime.datetime(2000, 1, 1, tzinfo=datetime.timezone.utc)

SLATE = "#596267"
PLUM = "#8B4976"
TEAL = "#2C786C"
OCHRE = "#A88A42"
INK = "#252525"
GREY = "#737373"
LIGHT_GREY = "#D9D9D9"

# Names for the observation contrast that recurs across the figures.
FULL_COLOR = SLATE
REDUCED_COLOR = PLUM


def configure_style() -> None:
    """Apply the shared rcParams for multi-panel figures at manuscript width."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.0,
        "axes.titlesize": 8.7,
        "axes.titleweight": "semibold",
        "axes.labelsize": 8.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
        "axes.linewidth": 0.75,
        "lines.linewidth": 1.35,
        "lines.markersize": 4.0,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })


def style_axis(axis: plt.Axes, grid: str | None = "y") -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    if grid:
        axis.grid(axis=grid, color=LIGHT_GREY, linewidth=0.55, alpha=0.75, zorder=0)
    axis.set_axisbelow(True)


def save_figure(figure: plt.Figure, base: Path, close: bool = True) -> None:
    """Write uncropped vector and raster copies of a figure."""
    base.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(base.with_suffix(".pdf"), metadata={
        "Creator": "taskscope", "CreationDate": FIXED_PDF_DATE, "ModDate": FIXED_PDF_DATE,
    })
    figure.savefig(base.with_suffix(".png"), dpi=400)
    if close:
        plt.close(figure)


def write_pdf(figure: plt.Figure, path: Path, pad_inches: float, close: bool = True) -> Path:
    path = path.with_suffix(".pdf")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight", pad_inches=pad_inches, metadata={
        "Creator": "taskscope", "CreationDate": FIXED_PDF_DATE, "ModDate": FIXED_PDF_DATE,
    })
    if close:
        plt.close(figure)
    return path


def crop_to_ink(path: Path, *, dpi: float = 600, guard: float = 0.3,
                threshold: int = 255, grayscale: bool = False) -> list[float]:
    """Shrink the crop box to the rendered marks and return the resulting page rectangle.

    The artwork stays vector. ``guard`` is the retained margin in points, and
    ``threshold`` is the sample value below which a pixel counts as ink.
    """
    zoom = dpi / 72
    with pymupdf.open(path) as document:
        page = document[0]
        matrix = pymupdf.Matrix(zoom, zoom)
        if grayscale:
            raster = page.get_pixmap(matrix=matrix, colorspace=pymupdf.csGRAY, alpha=False)
            samples = np.frombuffer(raster.samples, dtype=np.uint8).reshape(
                raster.height, raster.width)
            rows, columns = np.nonzero(samples < threshold)
        else:
            raster = page.get_pixmap(matrix=matrix, alpha=False)
            samples = np.frombuffer(raster.samples, dtype=np.uint8).reshape(
                raster.height, raster.width, raster.n)
            rows, columns = np.nonzero(np.any(samples[:, :, :3] < threshold, axis=2))
        if not len(rows):
            raise ValueError(f"Figure contains no visible marks: {path}")
        crop = pymupdf.Rect(columns.min() / zoom - guard, rows.min() / zoom - guard,
                         (columns.max() + 1) / zoom + guard, (rows.max() + 1) / zoom + guard)
        page.set_cropbox(crop & page.rect)
        geometry = [page.rect.width, page.rect.height]
        cropped = path.with_suffix(".cropped.pdf")
        document.save(cropped, garbage=4, deflate=True)
    cropped.replace(path)
    return geometry


def render_preview(path: Path, output: Path, dpi: float, scale: float = 1.0) -> None:
    with pymupdf.open(path) as document:
        matrix = pymupdf.Matrix(dpi / 72 * scale, dpi / 72 * scale)
        document[0].get_pixmap(matrix=matrix, alpha=False).save(output)


def manuscript_previews(path: Path, width_points: float = 468) -> dict:
    """Write review rasters and report the height the figure occupies at 6.5 inches."""
    with pymupdf.open(path) as document:
        rect = document[0].rect
    scale = width_points / rect.width
    for dpi, suffix in ((220, ".png"), (100, "_manuscript_size.png")):
        render_preview(path, path.with_name(path.stem + suffix), dpi, scale)
    return {"native_points": [rect.x0, rect.y0, rect.x1, rect.y1],
            "height_inches_at_6_5in": rect.height * scale / 72}
