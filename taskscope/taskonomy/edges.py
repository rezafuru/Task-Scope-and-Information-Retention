"""Fixed Gaussian-Sobel edge target computed from the resized RGB observation.

The target is a deterministic function of the RGB input, not a Taskonomy-provided label.
RGB in [0, 1] is converted to grayscale with the 0.299/0.587/0.114 coefficients, smoothed by
a separable discrete Gaussian with zero padding and boundary mask normalisation, then
reduced to a Sobel gradient magnitude. The magnitude is divided by sqrt(2) so that a unit
step gives one, the outermost pixel ring is zeroed, and the result is rescaled by 1/0.08 and
clipped to [0, 1].
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

GAUSSIAN_SIGMA = 3.0
GAUSSIAN_TRUNCATE = 4.0
GAUSSIAN_RADIUS = 12
EDGE_SCALE = 0.08
# Keeps the magnitude derivative finite where both gradients vanish.
EDGE_MAGNITUDE_EPSILON = 1e-12
GRAYSCALE_COEFFICIENTS = (0.299, 0.587, 0.114)
SOBEL_HORIZONTAL = ((0.25, 0.5, 0.25), (0.0, 0.0, 0.0), (-0.25, -0.5, -0.25))
SOBEL_VERTICAL = ((0.25, 0.0, -0.25), (0.5, 0.0, -0.5), (0.25, 0.0, -0.25))


def _gaussian_kernel(reference: Tensor) -> Tensor:
    coordinates = torch.arange(
        -GAUSSIAN_RADIUS,
        GAUSSIAN_RADIUS + 1,
        dtype=reference.dtype,
        device=reference.device,
    )
    gaussian = torch.exp(-coordinates.square() / (2.0 * GAUSSIAN_SIGMA**2))
    return gaussian / gaussian.sum()


def _mask_normalized_gaussian(grayscale: Tensor) -> Tensor:
    """Separable Gaussian blur with zero padding, divided by the blur of an all-one mask."""
    gaussian = _gaussian_kernel(grayscale)
    horizontal = gaussian.view(1, 1, 1, -1)
    vertical = gaussian.view(1, 1, -1, 1)
    smoothed_x = F.conv2d(grayscale, horizontal, padding=(0, GAUSSIAN_RADIUS))
    smoothed = F.conv2d(smoothed_x, vertical, padding=(GAUSSIAN_RADIUS, 0))
    mask = torch.ones_like(grayscale)
    mask_x = F.conv2d(mask, horizontal, padding=(0, GAUSSIAN_RADIUS))
    mask_smoothed = F.conv2d(mask_x, vertical, padding=(GAUSSIAN_RADIUS, 0))
    return smoothed / mask_smoothed.clamp_min(torch.finfo(grayscale.dtype).eps)


def fixed_gaussian_sobel(rgb: Tensor) -> Tensor:
    """Return the fixed Gaussian-Sobel target for CHW or BCHW RGB."""
    squeeze = rgb.ndim == 3
    value = rgb.unsqueeze(0) if squeeze else rgb
    if value.ndim != 4 or value.shape[1] != 3:
        raise ValueError(f"expected CHW or BCHW RGB, got shape {tuple(rgb.shape)}")
    if not value.is_floating_point():
        raise ValueError("RGB input must be floating point")
    coefficients = value.new_tensor(GRAYSCALE_COEFFICIENTS).view(1, 3, 1, 1)
    grayscale = (value * coefficients).sum(dim=1, keepdim=True)
    smoothed = _mask_normalized_gaussian(grayscale)
    horizontal_kernel = value.new_tensor(SOBEL_HORIZONTAL).view(1, 1, 3, 3)
    vertical_kernel = value.new_tensor(SOBEL_VERTICAL).view(1, 1, 3, 3)
    gradient_horizontal = F.conv2d(smoothed, horizontal_kernel, padding=1)
    gradient_vertical = F.conv2d(smoothed, vertical_kernel, padding=1)
    magnitude = torch.sqrt(
        gradient_horizontal.square()
        + gradient_vertical.square()
        + value.new_tensor(EDGE_MAGNITUDE_EPSILON)
    ).div(math.sqrt(2.0))
    magnitude[..., 0, :] = 0
    magnitude[..., -1, :] = 0
    magnitude[..., :, 0] = 0
    magnitude[..., :, -1] = 0
    result = magnitude.div(EDGE_SCALE).clamp(0.0, 1.0)
    return result[0] if squeeze else result
