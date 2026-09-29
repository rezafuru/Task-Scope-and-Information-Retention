"""Analysis transform, task decoders, hyperprior stream, and the packet container.

The family codec sends one latent stream. An image is analysed to a stride-16 latent, the
latent is entropy coded against a scale-and-mean hyperprior, and every task decoder reads
the same decoded latent. Packets carry a fixed header with payload lengths and CRC32
checksums so a measured byte count is a complete message length.

CompressAI supplies the entropy models. It is imported lazily so that the rest of the
package stays importable without it, and the error names the package at first use.
"""

from __future__ import annotations

import math
import struct
import zlib
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .data import NUM_SEMANTIC_CLASSES

try:
    from compressai.entropy_models import EntropyBottleneck, GaussianConditional
    from compressai.models.utils import update_registered_buffers
except ImportError:  # Resolved at first use, see _require_compressai.
    EntropyBottleneck = None
    GaussianConditional = None
    update_registered_buffers = None

_COMPRESSAI_MISSING = (
    "The Taskonomy family codec needs the compressai package for its entropy models "
    "(pip install compressai)."
)

PACKET_MAGIC = b"REQTREE1"
PACKET_VERSION = 1
PACKET_HEADER = struct.Struct(">8sBBBBHHHHHHIIII")
# The family codec writes a single stream under a single tag. The identifiers fix the
# header byte values of the format the reported byte counts were measured with.
HIERARCHY_IDS = {"H_R": 1}
STREAM_IDS = {"C": 1}
ID_HIERARCHIES = {value: key for key, value in HIERARCHY_IDS.items()}
ID_STREAMS = {value: key for key, value in STREAM_IDS.items()}

SCALE_TABLE_MINIMUM = 0.11
SCALE_TABLE_MAXIMUM = 256.0
SCALE_TABLE_LEVELS = 64


def _require_compressai() -> None:
    if EntropyBottleneck is None:
        raise ImportError(_COMPRESSAI_MISSING)


def load_entropy_buffers(module: nn.Module, name: str, buffers: Sequence[str], state: dict) -> None:
    """Resize the registered CDF buffers of one entropy model to match a checkpoint."""
    _require_compressai()
    update_registered_buffers(module, name, list(buffers), state)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, value: Tensor) -> Tensor:
        return value + self.body(value)


class AnalysisTransform(nn.Module):
    """Four stride-two stages from a three-channel image to the latent."""

    def __init__(self, out_channels: int, base_channels: int = 96) -> None:
        super().__init__()
        widths = (base_channels, base_channels, base_channels * 2, out_channels)
        layers: list[nn.Module] = []
        in_channels = 3
        for index, width in enumerate(widths):
            layers.extend([nn.Conv2d(in_channels, width, 5, stride=2, padding=2), nn.GELU()])
            if index in (1, 2):
                layers.append(ResidualBlock(width))
            in_channels = width
        self.net = nn.Sequential(*layers)

    def forward(self, rgb: Tensor) -> Tensor:
        return self.net(rgb)


class TaskDecoder(nn.Module):
    """Four transposed-convolution stages back to image resolution, with a per-task output."""

    def __init__(
        self, in_channels: int, out_channels: int, task: str, base_channels: int = 96
    ) -> None:
        super().__init__()
        self.task = task
        widths = (base_channels * 2, base_channels, base_channels, base_channels)
        layers: list[nn.Module] = []
        current = in_channels
        for width in widths:
            layers.extend(
                [
                    nn.ConvTranspose2d(current, width, 4, stride=2, padding=1),
                    nn.GELU(),
                    ResidualBlock(width),
                ]
            )
            current = width
        layers.append(nn.Conv2d(current, out_channels, 3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, route: Tensor) -> Tensor:
        output = self.net(route)
        if self.task == "rgb":
            return output.sigmoid()
        if self.task == "depth":
            return F.softplus(output) + 1e-4
        return output


class HyperpriorStream(nn.Module):
    """Scale-and-mean hyperprior entropy model over one latent stream."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        _require_compressai()
        hyper_channels = max(16, channels // 2)
        self.channels = channels
        self.hyper_analysis = nn.Sequential(
            nn.Conv2d(channels, hyper_channels, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(hyper_channels, hyper_channels, 3, stride=2, padding=1),
        )
        self.hyper_synthesis = nn.Sequential(
            nn.ConvTranspose2d(hyper_channels, hyper_channels, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(hyper_channels, 2 * channels, 4, stride=2, padding=1),
        )
        self.entropy_bottleneck = EntropyBottleneck(hyper_channels)
        self.gaussian = GaussianConditional(None)

    def _distribution_parameters(self, z_hat: Tensor) -> tuple[Tensor, Tensor]:
        means, raw_scales = self.hyper_synthesis(z_hat).chunk(2, dim=1)
        scales = F.softplus(raw_scales) + SCALE_TABLE_MINIMUM
        return means, scales

    def forward(self, value: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        z = self.hyper_analysis(value.abs())
        z_hat, z_likelihood = self.entropy_bottleneck(z)
        means, scales = self._distribution_parameters(z_hat)
        value_hat, value_likelihood = self.gaussian(value, scales, means=means)
        return value_hat, {"hyper": z_likelihood, "main": value_likelihood}

    @torch.no_grad()
    def compress(self, value: Tensor) -> tuple[Tensor, bytes, bytes]:
        if value.shape[0] != 1:
            raise ValueError("actual entropy coding currently requires batch size one")
        z = self.hyper_analysis(value.abs())
        hyper_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(hyper_strings, z.shape[-2:])
        means, scales = self._distribution_parameters(z_hat)
        indexes = self.gaussian.build_indexes(scales)
        main_strings = self.gaussian.compress(value, indexes, means=means)
        value_hat = self.gaussian.decompress(main_strings, indexes, means=means)
        return value_hat, hyper_strings[0], main_strings[0]

    @torch.no_grad()
    def decompress(
        self, hyper_payload: bytes, main_payload: bytes, z_shape: tuple[int, int]
    ) -> Tensor:
        z_hat = self.entropy_bottleneck.decompress([hyper_payload], z_shape)
        means, scales = self._distribution_parameters(z_hat)
        indexes = self.gaussian.build_indexes(scales)
        return self.gaussian.decompress([main_payload], indexes, means=means)

    def update(self, force: bool = False, update_quantiles: bool = False) -> bool:
        device = next(self.parameters()).device
        scale_table = torch.exp(
            torch.linspace(
                math.log(SCALE_TABLE_MINIMUM),
                math.log(SCALE_TABLE_MAXIMUM),
                SCALE_TABLE_LEVELS,
                device=device,
            )
        )
        gaussian_updated = self.gaussian.update_scale_table(scale_table, force=force)
        bottleneck_updated = self.entropy_bottleneck.update(
            force=force, update_quantiles=update_quantiles
        )
        return gaussian_updated or bottleneck_updated

    def aux_loss(self) -> Tensor:
        return self.entropy_bottleneck.loss()


@dataclass(frozen=True)
class PacketRecord:
    stream: str
    packet: bytes
    header_bytes: int
    hyper_bytes: int
    main_bytes: int

    @property
    def total_bytes(self) -> int:
        return len(self.packet)


def build_packet(
    hierarchy: str,
    stream: str,
    value_shape: Sequence[int],
    z_shape: Sequence[int],
    hyper_payload: bytes,
    main_payload: bytes,
) -> PacketRecord:
    if len(value_shape) != 4 or len(z_shape) != 2:
        raise ValueError("packet shapes are inconsistent")
    _, channels, height, width = (int(item) for item in value_shape)
    z_height, z_width = (int(item) for item in z_shape)
    header = PACKET_HEADER.pack(
        PACKET_MAGIC,
        PACKET_VERSION,
        HIERARCHY_IDS[hierarchy],
        STREAM_IDS[stream],
        0,
        channels,
        height,
        width,
        z_height,
        z_width,
        0,
        len(hyper_payload),
        len(main_payload),
        zlib.crc32(hyper_payload) & 0xFFFFFFFF,
        zlib.crc32(main_payload) & 0xFFFFFFFF,
    )
    return PacketRecord(
        stream=stream,
        packet=header + hyper_payload + main_payload,
        header_bytes=len(header),
        hyper_bytes=len(hyper_payload),
        main_bytes=len(main_payload),
    )


@dataclass(frozen=True)
class ParsedPacket:
    hierarchy: str
    stream: str
    value_shape: tuple[int, int, int, int]
    z_shape: tuple[int, int]
    hyper_payload: bytes
    main_payload: bytes


def parse_packet(packet: bytes) -> ParsedPacket:
    if len(packet) < PACKET_HEADER.size:
        raise ValueError("truncated packet")
    (
        magic,
        version,
        hierarchy_id,
        stream_id,
        flags,
        channels,
        height,
        width,
        z_height,
        z_width,
        reserved,
        hyper_length,
        main_length,
        hyper_checksum,
        main_checksum,
    ) = PACKET_HEADER.unpack(packet[: PACKET_HEADER.size])
    if magic != PACKET_MAGIC or version != PACKET_VERSION or flags != 0 or reserved != 0:
        raise ValueError("unknown packet format")
    if hierarchy_id not in ID_HIERARCHIES or stream_id not in ID_STREAMS:
        raise ValueError("unknown hierarchy or stream identifier")
    payload = packet[PACKET_HEADER.size :]
    if len(payload) != hyper_length + main_length:
        raise ValueError("packet length mismatch")
    hyper_payload = payload[:hyper_length]
    main_payload = payload[hyper_length:]
    if zlib.crc32(hyper_payload) & 0xFFFFFFFF != hyper_checksum:
        raise ValueError("hyperlatent checksum mismatch")
    if zlib.crc32(main_payload) & 0xFFFFFFFF != main_checksum:
        raise ValueError("main-latent checksum mismatch")
    return ParsedPacket(
        hierarchy=ID_HIERARCHIES[hierarchy_id],
        stream=ID_STREAMS[stream_id],
        value_shape=(1, channels, height, width),
        z_shape=(z_height, z_width),
        hyper_payload=hyper_payload,
        main_payload=main_payload,
    )


def roundtrip_error(encoded: Tensor, decoded: Tensor, stream: str) -> float:
    """Largest deviation between the encoder-side and decoder-side latent, raising if it drifts."""
    maximum = float((encoded - decoded).abs().max())
    if not torch.allclose(encoded, decoded, rtol=1e-6, atol=1e-6):
        raise AssertionError(f"entropy round trip changed stream {stream} by {maximum:.9g}")
    return maximum


class FamilyCodec(nn.Module):
    """One analysis transform, one coded latent, and one decoder per supported task."""

    TASK_OUTPUTS = (("depth", 1), ("semantic", NUM_SEMANTIC_CLASSES), ("edge", 1), ("rgb", 3))

    def __init__(self, channels: int = 96, base_channels: int = 64, head_channels: int = 48) -> None:
        super().__init__()
        self.analysis = AnalysisTransform(channels, base_channels)
        self.stream = HyperpriorStream(channels)
        self.decoders = nn.ModuleDict(
            {
                task: TaskDecoder(channels, outputs, task, head_channels)
                for task, outputs in self.TASK_OUTPUTS
            }
        )

    def encode(self, rgb: Tensor, compressed: bool) -> tuple[Tensor, Tensor]:
        """Latent and estimated bits per pixel. An uncompressed reference skips the stream."""
        latent = self.analysis(rgb)
        if not compressed:
            return latent, latent.new_zeros(())
        latent, likelihoods = self.stream(latent.float())
        pixels = rgb.shape[0] * rgb.shape[-2] * rgb.shape[-1]
        rate = sum(-p.clamp_min(1e-9).log2().sum() / pixels for p in likelihoods.values())
        return latent, rate

    def predict(self, latent: Tensor, tasks: Sequence[str]) -> dict[str, Tensor]:
        result = {task: self.decoders[task](latent) for task in tasks}
        if "edge" in result:
            result["edge"] = result["edge"].sigmoid()
        return {task: value.float() for task, value in result.items()}

    @torch.no_grad()
    def code(self, rgb: Tensor) -> tuple[Tensor, PacketRecord]:
        """Entropy code one image, parse the packet back, and return the decoded latent."""
        raw = self.analysis(rgb)
        # Repeated entropy parameters must select the same arithmetic-coder indexes.
        with torch.backends.cudnn.flags(
            enabled=None,
            benchmark=False,
            benchmark_limit=None,
            deterministic=True,
            allow_tf32=None,
            fp32_precision=None,
        ):
            encoded, hyper, main = self.stream.compress(raw)
            record = build_packet(
                "H_R", "C", raw.shape, (raw.shape[-2] // 4, raw.shape[-1] // 4), hyper, main
            )
            parsed = parse_packet(record.packet)
            decoded = self.stream.decompress(
                parsed.hyper_payload, parsed.main_payload, parsed.z_shape
            )
        roundtrip_error(encoded, decoded, "C")
        return decoded, record
