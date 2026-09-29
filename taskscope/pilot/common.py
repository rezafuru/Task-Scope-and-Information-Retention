"""Hashing, JSON, and device helpers shared by the pilot stages.

Every stage records the SHA256 of the files it reads and writes, so a later stage can
refuse inputs that changed after the stage that produced them ran.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from taskscope.paths import REPO_ROOT

DEFAULT_RUN = REPO_ROOT / "runs/pilot"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bytes_hash(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def object_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def write_json(path: Path, value: Any, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=indent, allow_nan=False) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def log(**fields: Any) -> None:
    print(json.dumps(fields, allow_nan=False), flush=True)


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return device.type


def deterministic(strict: bool = True) -> None:
    """Fixed numerics for fitting and inference. Disable TF32 so exits reproduce exactly."""
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(strict)
    torch.backends.cudnn.deterministic = strict
    torch.backends.cudnn.benchmark = not strict
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
