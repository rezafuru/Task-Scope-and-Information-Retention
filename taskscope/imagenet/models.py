"""Frozen ResNet-50 observation, the layer-2 feature codec, and the adapted readout.

Both conditions share one interface. The encoder sees the 512x28x28 layer-2 tensor of an
ImageNet-pretrained ResNet-50 at 224x224 and the decoder returns a tensor of the same
shape to the frozen suffix, so the two fitting losses differ and nothing else does.

A message is one packet: a fixed 36-byte header carrying the magic string, the latent and
hyperlatent dimensions, both payload lengths and their CRC32 checksums, followed by the
hyperlatent and main entropy payloads. The four framing bytes a stored stream adds per
packet are counted separately by the caller, so a reported rate is a complete message
length.

CompressAI supplies the entropy models. It is imported lazily so that the rest of the
package stays importable without it, and the error names the package at first use.
"""

from __future__ import annotations

import hashlib
import math
import struct
import zlib
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.models import ResNet50_Weights, resnet50

from .data import COARSE_CLASSES, FINE_CLASSES, class_maps

try:
    from compressai.entropy_models import EntropyBottleneck, GaussianConditional
except ImportError:  # Resolved at first use, see _require_compressai.
    EntropyBottleneck = None
    GaussianConditional = None

_COMPRESSAI_MISSING = ("The ImageNet feature codec needs the compressai package for its entropy "
                       "models (pip install compressai).")

PACKET_MAGIC = b"IMGSCP01"
PACKET_HEADER = struct.Struct(">8sBBHHHHHIIII")
# Four bytes per packet frame its length when messages are written to one stream.
OUTER_FRAMING_BYTES = 4

FEATURE_SHAPE = (512, 28, 28)
LATENT_SHAPE = (64, 8, 8)
HYPER_SHAPE = (32, 2, 2)
ANALYSIS_WIDTHS = (512, 192, 64)
PADDED_FEATURE = 32

SCALE_TABLE_MINIMUM = 0.11
SCALE_TABLE_MAXIMUM = 256.0
SCALE_TABLE_LEVELS = 64

CODEC_CONFIGURATION = {"observation": "resnet50.layer2", "feature_shape": list(FEATURE_SHAPE),
                       "latent_shape": list(LATENT_SHAPE), "hyper_shape": list(HYPER_SHAPE),
                       "packet_magic": PACKET_MAGIC.decode(), "packet_version": 1,
                       "packet_type": "common_layer2_feature",
                       "packet_header_bytes": PACKET_HEADER.size}


def _require_compressai() -> None:
    if EntropyBottleneck is None:
        raise ImportError(_COMPRESSAI_MISSING)


def tensor_state_sha256(state: dict) -> str:
    digest = hashlib.sha256()
    for name, value in state.items():
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class ImageNetPacket:
    packet: bytes
    header_bytes: int
    hyper_bytes: int
    main_bytes: int

    @property
    def total_bytes(self) -> int:
        return len(self.packet)


def pack_message(hyper: bytes, main: bytes) -> ImageNetPacket:
    header = PACKET_HEADER.pack(PACKET_MAGIC, 1, 1, *LATENT_SHAPE, *HYPER_SHAPE[1:],
                                len(hyper), len(main), zlib.crc32(hyper) & 0xFFFFFFFF,
                                zlib.crc32(main) & 0xFFFFFFFF)
    return ImageNetPacket(header + hyper + main, len(header), len(hyper), len(main))


def unpack_message(packet: bytes) -> tuple[bytes, bytes]:
    if len(packet) < PACKET_HEADER.size:
        raise ValueError("Truncated ImageNet feature packet")
    fields = PACKET_HEADER.unpack(packet[:PACKET_HEADER.size])
    if fields[:8] != (PACKET_MAGIC, 1, 1, *LATENT_SHAPE, *HYPER_SHAPE[1:]):
        raise ValueError("Unknown ImageNet packet identifier, type or dimensions")
    hyper_size, main_size, hyper_crc, main_crc = fields[8:]
    if min(hyper_size, main_size) <= 0 or len(packet) != PACKET_HEADER.size + hyper_size + main_size:
        raise ValueError("ImageNet packet length mismatch")
    hyper = packet[PACKET_HEADER.size:PACKET_HEADER.size + hyper_size]
    main = packet[PACKET_HEADER.size + hyper_size:]
    if (zlib.crc32(hyper) & 0xFFFFFFFF) != hyper_crc or (zlib.crc32(main) & 0xFFFFFFFF) != main_crc:
        raise ValueError("ImageNet entropy payload checksum mismatch")
    return hyper, main


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
        return means, F.softplus(raw_scales) + SCALE_TABLE_MINIMUM

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
    def decompress(self, hyper_payload: bytes, main_payload: bytes,
                   z_shape: tuple[int, int]) -> Tensor:
        z_hat = self.entropy_bottleneck.decompress([hyper_payload], z_shape)
        means, scales = self._distribution_parameters(z_hat)
        indexes = self.gaussian.build_indexes(scales)
        return self.gaussian.decompress([main_payload], indexes, means=means)

    def update(self, force: bool = False, update_quantiles: bool = False) -> bool:
        device = next(self.parameters()).device
        scale_table = torch.exp(torch.linspace(math.log(SCALE_TABLE_MINIMUM),
                                               math.log(SCALE_TABLE_MAXIMUM),
                                               SCALE_TABLE_LEVELS, device=device))
        gaussian_updated = self.gaussian.update_scale_table(scale_table, force=force)
        bottleneck_updated = self.entropy_bottleneck.update(force=force,
                                                            update_quantiles=update_quantiles)
        return gaussian_updated or bottleneck_updated

    def aux_loss(self) -> Tensor:
        return self.entropy_bottleneck.loss()


class FrozenTeacher(nn.Module):
    """Common pretrained layer2 observation and differentiable frozen suffix."""

    def __init__(self, mapping: dict) -> None:
        super().__init__()
        self.network = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.network.eval().requires_grad_(False)
        _, original, groups = class_maps(mapping)
        self.register_buffer("fine_indices", torch.tensor(original))
        self.register_buffer("groups", torch.tensor(groups))
        self.identity = {"architecture": "torchvision.resnet50", "weights": "IMAGENET1K_V2",
                         "state_sha256": tensor_state_sha256(self.network.state_dict()),
                         "transform": repr(ResNet50_Weights.IMAGENET1K_V2.transforms()),
                         "fine_indices": original, "fine_to_coarse": groups}

    def observe(self, image: Tensor) -> Tensor:
        net = self.network
        value = net.maxpool(net.relu(net.bn1(net.conv1(image))))
        return net.layer2(net.layer1(value))

    def predict(self, feature: Tensor) -> Tensor:
        net = self.network
        value = net.layer4(net.layer3(feature))
        return net.fc(net.avgpool(value).flatten(1))[:, self.fine_indices]


class FeatureCodec(nn.Module):
    """A fixed 512x28x28 feature interface with a 64x8x8 hyperprior latent."""

    def __init__(self) -> None:
        super().__init__()
        wide, middle, narrow = ANALYSIS_WIDTHS
        self.register_buffer("mean", torch.zeros(1, wide, 1, 1))
        self.register_buffer("scale", torch.ones(1, wide, 1, 1))
        self.analysis = nn.Sequential(nn.Conv2d(wide, middle, 5, 2, 2), nn.GELU(),
                                      nn.Conv2d(middle, narrow, 5, 2, 2))
        self.stream = HyperpriorStream(narrow)
        self.synthesis = nn.Sequential(nn.ConvTranspose2d(narrow, middle, 4, 2, 1), nn.GELU(),
                                       nn.ConvTranspose2d(middle, wide, 4, 2, 1))

    def encode_feature(self, feature: Tensor) -> Tensor:
        if feature.shape[1:] != FEATURE_SHAPE:
            raise ValueError("Require the native224 ResNet50 layer2 observation")
        padding = PADDED_FEATURE - FEATURE_SHAPE[1]
        return self.analysis(F.pad((feature - self.mean) / self.scale, (0, padding, 0, padding)))

    def decode_feature(self, latent: Tensor) -> Tensor:
        height, width = FEATURE_SHAPE[1:]
        return self.synthesis(latent)[:, :, :height, :width] * self.scale + self.mean

    def forward(self, feature: Tensor) -> tuple[Tensor, Tensor]:
        latent = self.encode_feature(feature)
        decoded, likelihoods = self.stream(latent.float())
        rate = sum(-value.clamp_min(1e-9).log2().sum() for value in likelihoods.values())
        return self.decode_feature(decoded), rate / (len(feature) * 224 * 224)

    @torch.no_grad()
    def code(self, feature: Tensor) -> tuple[Tensor, ImageNetPacket]:
        latent = self.encode_feature(feature)
        compressed, hyper, main = self.stream.compress(latent.float())
        packet = pack_message(hyper, main)
        decoded = self.decode_packet(packet.packet)
        if not torch.equal(self.decode_feature(compressed), decoded):
            raise RuntimeError("Entropy packet round trip differs from encoded feature")
        return decoded, packet

    @torch.no_grad()
    def decode_packet(self, packet: bytes) -> Tensor:
        hyper, main = unpack_message(packet)
        latent = self.stream.decompress(hyper, main, HYPER_SHAPE[1:])
        return self.decode_feature(latent)


class SuffixReadout(nn.Module):
    """Coarse and fine heads on the frozen suffix's 2048-dimensional penultimate feature."""

    def __init__(self, features: int = 2048, hidden: int = 512) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Linear(features, hidden), nn.ReLU())
        self.fine = nn.Linear(hidden, FINE_CLASSES)
        self.coarse = nn.Linear(hidden, COARSE_CLASSES)

    def forward(self, value: Tensor) -> tuple[Tensor, Tensor]:
        hidden = self.body(value)
        return self.fine(hidden), self.coarse(hidden)
