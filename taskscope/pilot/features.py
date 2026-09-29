"""The frozen ImageNet ResNet-50, its residual-block exits, and the reader families.

One network supplies every observation in this chain. ``ResNet50_Weights.IMAGENET1K_V2`` is
downloaded into the torch hub cache on first use and is never adapted.

Exits are ``layer3`` (1024x14x14, the stage 3 output) and ``block1``, ``block2``, ``block3``
(2048x7x7, the outputs of the three stage 4 residual blocks). ``block3`` is the stage 4 output.
At every exit the unchanged remainder of the network reproduces the original logits bit for bit,
which :func:`verify_exit` asserts before any feature is written.

Readers, one per exit, all fitted with the same budget:
  rgb      complete ResNet-50 from the ImageNet weights with a fresh output layer
  layer3   the pretrained stage 4, global average pooling, fresh output layer
  block*   the pretrained final stage 4 block, global average pooling, fresh output layer

torchvision is an optional dependency and is imported only when a stage needs the network.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import torch
from torch import nn

from taskscope.pilot.common import file_hash, read_json

WEIGHTS_NAME = "ResNet50_Weights.IMAGENET1K_V2"
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
EXITS = ("layer3", "block1", "block2", "block3")
VIEWS = ("rgb", "layer3", "block1", "block2", "block3")
STAGE_SHAPES = {"layer1": (256, 56, 56), "layer2": (512, 28, 28),
                "layer3": (1024, 14, 14), "layer4": (2048, 7, 7)}
EXIT_SHAPES = {"layer3": STAGE_SHAPES["layer3"], "block1": STAGE_SHAPES["layer4"],
               "block2": STAGE_SHAPES["layer4"], "block3": STAGE_SHAPES["layer4"]}


def torchvision_module():
    try:
        import torchvision
    except ImportError as error:
        raise ImportError(
            "The pilot stages need torchvision for the ImageNet ResNet-50. Install it with "
            "pip install 'taskscope[train]', or pip install torchvision."
        ) from error
    return torchvision


def weights_enum():
    torchvision_module()
    from torchvision.models import ResNet50_Weights

    return ResNet50_Weights.IMAGENET1K_V2


def resnet50(pretrained: bool) -> nn.Module:
    """ImageNet ResNet-50. ``pretrained`` downloads and loads IMAGENET1K_V2."""
    torchvision_module()
    from torchvision.models import resnet50 as build

    return build(weights=weights_enum() if pretrained else None)


def runtime_provenance() -> dict[str, str]:
    torchvision = torchvision_module()
    from torchvision.models.resnet import ResNet

    return {"torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
            "torchvision_model_source_sha256": file_hash(Path(inspect.getfile(ResNet)))}


def checkpoint_provenance() -> dict[str, str]:
    """Identify the downloaded weight file in the torch hub cache."""
    url = weights_enum().url
    path = Path(torch.hub.get_dir()) / "checkpoints" / Path(urlparse(url).path).name
    return {"weights": WEIGHTS_NAME, "url": url, "path": str(path), "sha256": file_hash(path)}


def source_provenance(symbol: Any) -> dict[str, str]:
    path = Path(inspect.getfile(symbol)).resolve()
    return {"path": str(path), "sha256": file_hash(path)}


class FrozenTeacher(nn.Module):
    """The unchanged ImageNet network, evaluated on prepared 224x224 RGB in [0, 1]."""

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        transform = weights_enum().transforms()
        self.network = resnet50(pretrained=True).to(device).eval().requires_grad_(False)
        self.register_buffer("mean", torch.tensor(transform.mean, device=device).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(transform.std, device=device).view(1, 3, 1, 1))
        self.eval()

    def forward(self, rgb: torch.Tensor, depth: str = "both",
                logits: bool = False) -> dict[str, torch.Tensor]:
        if depth not in (*STAGE_SHAPES, "both", "all"):
            raise ValueError("Unknown teacher depth")
        network = self.network
        value = (rgb - self.mean) / self.std
        value = network.maxpool(network.relu(network.bn1(network.conv1(value))))
        early = network.layer1(value)
        result = {"layer1": early}
        final_stage = 4 if depth in {"both", "all"} or logits else int(depth[-1])
        value = early
        for stage in range(2, final_stage + 1):
            name = f"layer{stage}"
            value = getattr(network, name)(value)
            if depth == "all" or name == depth or stage == 4:
                result[name] = value
        if logits:
            result["logits"] = network.fc(network.avgpool(value).flatten(1))
        return result

    def provenance(self) -> dict[str, Any]:
        return {"weights": WEIGHTS_NAME, "checkpoint": checkpoint_provenance(),
                "source": source_provenance(type(self.network)),
                "mean": self.mean.flatten().tolist(), "std": self.std.flatten().tolist(),
                "input": "Prepared 224x224 RGB input, normalization only, no further resize or crop",
                "feature_shapes": {name: list(shape) for name, shape in STAGE_SHAPES.items()}
                                  | {"logits": [1000]}}


@torch.no_grad()
def native_exit(teacher: FrozenTeacher, rgb: torch.Tensor, view: str) -> torch.Tensor:
    """Complete feature map at one exit."""
    if view.startswith("layer"):
        return teacher(rgb, depth=view)[view]
    value = teacher(rgb, depth="layer3")["layer3"]
    for block in list(teacher.network.layer4.children())[:int(view[-1])]:
        value = block(value)
    return value


@torch.no_grad()
def verify_exit(teacher: FrozenTeacher, rgb: torch.Tensor, view: str) -> None:
    """The unchanged suffix on a complete exit must reproduce the original logits exactly."""
    value = native_exit(teacher, rgb, view)
    if view.startswith("layer"):
        for stage in range(int(view[-1]) + 1, 5):
            value = getattr(teacher.network, f"layer{stage}")(value)
    else:
        for block in list(teacher.network.layer4.children())[int(view[-1]):]:
            value = block(value)
    after = teacher.network.fc(teacher.network.avgpool(value).flatten(1))
    original = teacher(rgb, depth="all", logits=True)["logits"]
    if not torch.equal(after, original):
        raise ValueError(f"Unchanged suffix does not recover the exact teacher logits at {view}")


def normalized(batch: torch.Tensor, device: torch.device) -> torch.Tensor:
    batch = batch.to(device=device, dtype=torch.float32, non_blocking=True).div_(255)
    mean = batch.new_tensor(MEAN).view(1, 3, 1, 1)
    std = batch.new_tensor(STD).view(1, 3, 1, 1)
    return (batch - mean) / std


def training_images(batch: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Training-only horizontal flip, probability 0.5, from an independent CPU generator."""
    flip = torch.rand(len(batch), generator=generator) < 0.5
    augmented = batch.clone()
    augmented[flip] = augmented[flip].flip(-1)
    return augmented


def new_reader(view: str, classes: int, seed: int, pretrained: bool) -> nn.Module:
    """Reader for one exit. The seed is set before construction so the fresh output layer is fixed."""
    if view not in VIEWS:
        raise ValueError(f"Unknown reader view: {view}")
    torch.manual_seed(seed)
    network = resnet50(pretrained=pretrained)
    if view == "rgb":
        network.fc = nn.Linear(network.fc.in_features, classes)
        return network
    if view == "layer3":
        return nn.Sequential(network.layer4, network.avgpool, nn.Flatten(), nn.Linear(2048, classes))
    return nn.Sequential(network.layer4[-1], network.avgpool, nn.Flatten(), nn.Linear(2048, classes))


def checked_feature_subset(cache: Path, records: list[dict], view: str) -> tuple[np.ndarray, dict]:
    """Select rows of a cached exit array by observation, verifying the join and the array hash."""
    cache = Path(cache)
    path = cache / "extraction.json"
    run = read_json(path)
    members = run["membership"]
    indexed = {row["observation_id"]: (index, row) for index, row in enumerate(members)}
    if run["status"] != "complete" or len(indexed) != len(members):
        raise ValueError("Incomplete or repeated cached feature membership")
    positions = []
    for row in records:
        index, stored = indexed[row["observation_id"]]
        if any(stored[key] != row[key] for key in
               ("group_id", "split", "input_sha256", "input_path", "pixel_sha256")):
            raise ValueError("Cached feature and source join differs")
        positions.append(index)
    source = cache / f"{view}.npy"
    if file_hash(source) != run["artifacts"][view]["sha256"]:
        raise ValueError(f"Cached feature array changed: {source}")
    values = np.load(source, mmap_mode="r", allow_pickle=False)
    if values.dtype != np.float32 or values.shape != (len(members), *EXIT_SHAPES[view]):
        raise ValueError("Cached feature dimensions changed")
    subset = values[positions].copy()
    if not np.isfinite(subset).all():
        raise ValueError("Nonfinite cached features")
    return subset, {"teacher": run["teacher"], "extraction_path": str(path.resolve()),
                    "extraction_sha256": file_hash(path),
                    "array_sha256": run["artifacts"][view]["sha256"]}
