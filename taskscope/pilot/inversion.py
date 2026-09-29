"""RGB inverses of the frozen exits, fitted on ImageNet and applied to the CUB photographs.

One inverse per exit maps the complete feature map back to a 224 by 224 image. The architecture
is a stack of 5 by 5 convolutions with 128 hidden channels, transposed with stride 2 while the
resolution has to double and plain otherwise, ReLU between layers and no output activation. A
14x14 exit needs four layers, a 7x7 exit five.

Fitting samples ImageNet training photographs uniformly with replacement from the complete
1,281,167, reads each by byte offset from the official archive, resizes the short side to 232 and
centre-crops 224. The objective is mean squared error on the unconstrained RGB output with
gradients clipped to norm 1, Adam at learning rate 1e-4, seed 17. The retained inverses ran
100000 updates of 64 photographs, validating every 2500 updates on a fixed 5000-photograph subset
of the official ImageNet validation set and keeping the state with the lowest validation error,
step zero included.

Export clamps the output to [0, 1] and rounds to 8-bit RGB. Labels never enter this path.
"""

from __future__ import annotations

import io
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from taskscope.pilot import features as feat
from taskscope.pilot.common import bytes_hash, file_hash, log, object_hash, read_json, write_json
from taskscope.pilot.data import check_png

SEED = 17
CLIP_NORM = 1.0
LEARNING_RATE = 1e-4
HIDDEN_CHANNELS = 128
KERNEL_SIZE = 5
IMAGENET_TRAIN_IMAGES = 1281167
IMAGENET_TRAIN_BYTES = 147897477120
IMAGENET_TRAIN_MD5 = "1d675b47d978889d74fa0da5fadfb00e"
IMAGENET_VALIDATION_IMAGES = 50000
FIT_KEYS = ("view", "runtime", "teacher", "initial_inverse", "training_data", "validation_data",
            "configuration", "continued_from")


def depth_of(view: str) -> str:
    """The stage whose channel count and resolution the inverse reads."""
    return view if view.startswith("layer") else "layer4"


def geometry(depth: str) -> tuple[int, int, int]:
    channels, size, _ = feat.STAGE_SHAPES[depth]
    ratio = 224 // size
    doubling_layers = ratio.bit_length() - 1
    if size * 2 ** doubling_layers != 224:
        raise ValueError("Input size must reach 224 through exact doubling")
    return channels, doubling_layers, max(4, doubling_layers)


class SimpleInverse(nn.Module):
    def __init__(self, depth: str) -> None:
        super().__init__()
        channels, doubling_layers, layer_count = geometry(depth)
        layers: list[nn.Module] = []
        for index in range(layer_count):
            following = 3 if index == layer_count - 1 else HIDDEN_CHANNELS
            if index < doubling_layers:
                layer = nn.ConvTranspose2d(channels, following, KERNEL_SIZE, stride=2, padding=2,
                                           output_padding=1)
            else:
                layer = nn.Conv2d(channels, following, KERNEL_SIZE, stride=1, padding=2)
            layers.append(layer)
            if index < layer_count - 1:
                layers.append(nn.ReLU(inplace=False))
            channels = following
        self.layers = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values)


def initial(view: str, device: torch.device, teacher_info: dict) -> tuple[nn.Module, dict]:
    depth = depth_of(view)
    torch.manual_seed(SEED)
    model = SimpleInverse(depth)
    state = {name: {"shape": list(value.shape), "sha256": bytes_hash(value.numpy().tobytes())}
             for name, value in model.state_dict().items()}
    channels, doubling_layers, layer_count = geometry(depth)
    provenance = {
        "target_exit": view,
        "architecture": {"latent_channels": channels, "hidden_channels": HIDDEN_CHANNELS,
                         "layers": layer_count, "kernel_size": KERNEL_SIZE,
                         "doubling_layers": doubling_layers,
                         "transposed_convolution": "Stride 2, padding 2, output_padding 1",
                         "hidden_activation": "ReLU", "output_activation": "None",
                         "parameters": sum(parameter.numel() for parameter in model.parameters())},
        "initialization": f"PyTorch convolution defaults, seed {SEED}. No normalization buffers.",
        "initial_state_sha256": object_hash(state),
        "normalization": "None. Raw complete teacher features.",
        "recipe": {"loss": "Mean squared error on unconstrained RGB", "gradient_clip_norm": CLIP_NORM},
        "teacher": teacher_info}
    return model.to(device).eval(), provenance


def image_crop(image: Image.Image) -> torch.Tensor:
    feat.torchvision_module()
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as TF

    image = TF.resize(image.convert("RGB"), 232, InterpolationMode.BILINEAR)
    return TF.pil_to_tensor(TF.center_crop(image, [224, 224]))


class ArchiveImages(Dataset):
    """Read a JPEG by absolute byte offset from the verified official ImageNet training archive."""

    def __init__(self, index: Path) -> None:
        index = Path(index)
        info = read_json(index.with_suffix(".json"))
        if (info["archive_bytes"] != IMAGENET_TRAIN_BYTES or info["archive_md5"] != IMAGENET_TRAIN_MD5
                or info["images"] != IMAGENET_TRAIN_IMAGES or info["classes"] != 1000
                or file_hash(index) != info["index_sha256"]):
            raise ValueError("Require the verified complete official ImageNet training index")
        self.path = Path(info["archive"])
        stat = self.path.stat()
        self.stat_identity = (stat.st_size, stat.st_mtime_ns)
        if stat.st_size != IMAGENET_TRAIN_BYTES:
            raise ValueError("Training archive size changed after indexing")
        with np.load(index, allow_pickle=False) as saved:
            self.offsets, self.lengths = saved["offsets"].copy(), saved["lengths"].copy()
        if (self.offsets.dtype != np.int64 or self.lengths.dtype != np.int64
                or self.offsets.shape != (IMAGENET_TRAIN_IMAGES,)
                or self.lengths.shape != self.offsets.shape
                or (self.offsets < 0).any() or (self.lengths <= 0).any()
                or (self.offsets + self.lengths > IMAGENET_TRAIN_BYTES).any()):
            raise ValueError("Invalid training image offsets")
        self.fd = None
        self.identity = {**info, "index_path": str(index.resolve()),
                         "index_metadata_sha256": file_hash(index.with_suffix(".json")),
                         "archive_mtime_ns": stat.st_mtime_ns}

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int) -> torch.Tensor:
        if self.fd is None:
            self.fd = os.open(self.path, os.O_RDONLY)
            stat = os.fstat(self.fd)
            if (stat.st_size, stat.st_mtime_ns) != self.stat_identity:
                raise ValueError("Training archive changed before worker access")
        size, offset = int(self.lengths[index]), int(self.offsets[index])
        content = os.pread(self.fd, size, offset)
        if len(content) != size:
            raise OSError(f"Short ImageNet image read at index {index}")
        with Image.open(io.BytesIO(content)) as image:
            return image_crop(image)


class ValidationImages(Dataset):
    """A fixed subset of the official ImageNet validation set, drawn once at seed 17."""

    def __init__(self, root: Path, count: int) -> None:
        paths = sorted(Path(root).glob("ILSVRC2012_val_*.JPEG"))
        expected = [f"ILSVRC2012_val_{index:08d}.JPEG"
                    for index in range(1, IMAGENET_VALIDATION_IMAGES + 1)]
        if [path.name for path in paths] != expected or not 1 <= count <= IMAGENET_VALIDATION_IMAGES:
            raise ValueError("Require the complete 50,000-image official validation directory")
        order = torch.randperm(len(paths), generator=torch.Generator().manual_seed(SEED))[:count]
        self.paths = [paths[index] for index in sorted(order.tolist())]
        self.identity = {"root": str(Path(root).resolve()),
                         "official_population": IMAGENET_VALIDATION_IMAGES, "selection_seed": SEED,
                         "selected_count": count,
                         "members": [{"name": path.name, "sha256": file_hash(path)} for path in self.paths]}

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.paths[index]) as image:
            return image_crop(image)


def mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (prediction.float() - target.float()).square().mean()


def _backend(strict: bool) -> None:
    torch.use_deterministic_algorithms(strict)
    torch.backends.cudnn.deterministic = strict
    torch.backends.cudnn.benchmark = not strict


def step(model: nn.Module, teacher: feat.FrozenTeacher, rgb: torch.Tensor,
         optimizer: torch.optim.Optimizer, view: str, amp: bool) -> float:
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=rgb.device.type, dtype=torch.bfloat16, enabled=amp and rgb.is_cuda):
        prediction = model(feat.native_exit(teacher, rgb, view))
    loss = mse(prediction, rgb)
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite inverse MSE")
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM, error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach())


@torch.no_grad()
def validate(model: nn.Module, teacher: feat.FrozenTeacher, loader: DataLoader, view: str,
             device: torch.device, layout: torch.memory_format) -> float:
    _backend(True)
    model.eval()
    total, count = 0.0, 0
    for images in loader:
        rgb = images.to(device=device, dtype=torch.float32, non_blocking=True, memory_format=layout) / 255
        value = mse(model(feat.native_exit(teacher, rgb, view)), rgb)
        if not torch.isfinite(value):
            raise FloatingPointError("Nonfinite float32 validation error")
        total += float(value) * len(rgb)
        count += len(rgb)
    return total / count


def fit(output: Path, *, view: str, train_index: Path, validation_root: Path, device: torch.device,
        iterations: int = 100000, batch_size: int = 64, validation_every: int = 2500,
        validation_count: int = 5000, workers: int = 8, memory_format: str = "channels_last",
        fast_training: bool = True) -> dict:
    """Fit one inverse on ImageNet and keep the state with the lowest validation RGB error."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    layout = torch.channels_last if memory_format == "channels_last" else torch.contiguous_format
    training = ArchiveImages(train_index)
    validation = ValidationImages(validation_root, validation_count)
    generator = torch.Generator().manual_seed(SEED)
    order = torch.randint(len(training), (iterations * batch_size,), generator=generator)
    loader_options = {"batch_size": batch_size, "num_workers": workers,
                      "pin_memory": device.type == "cuda", "persistent_workers": workers > 0}
    validation_loader = DataLoader(validation, shuffle=False, **loader_options)
    teacher = feat.FrozenTeacher(device)
    teacher_info = teacher.provenance()
    model, provenance = initial(view, device, teacher_info)
    model.to(memory_format=layout)
    teacher.to(memory_format=layout)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    info = {"status": "running", "view": view, "runtime": feat.runtime_provenance(),
            "teacher": teacher_info, "initial_inverse": provenance,
            "training_data": training.identity, "validation_data": validation.identity,
            "continued_from": None,
            "configuration": {
                "iterations": iterations, "batch_size": batch_size, "seed": SEED, "optimizer": "Adam",
                "learning_rate": LEARNING_RATE, "loss": "MSE on unconstrained RGB",
                "gradient_clip_norm": CLIP_NORM,
                "sampling": "Uniform independent images with replacement from all "
                            f"{IMAGENET_TRAIN_IMAGES} training photographs",
                "sample_order_sha256": bytes_hash(order.numpy().tobytes()),
                "crop": "PIL RGB, bilinear short-side resize 232, center crop 224. No augmentation.",
                "teacher_training_precision": "bfloat16 autocast with frozen float32 parameters",
                "inverse_training_precision": "bfloat16 autocast, float32 parameters and MSE loss",
                "validation_precision": "float32, TF32 disabled", "validation_every": validation_every,
                "memory_format": memory_format,
                "export_pixels": "Clamp to [0,1], round to 8-bit RGB. Training and validation MSE use "
                                 "raw outputs.",
                "training_deterministic_algorithms": not fast_training,
                "training_cudnn_benchmark": fast_training,
                "selection": "Minimum validation RGB MSE error, including step zero. Earliest step on ties.",
                "workers": workers}}
    started = time.monotonic()
    best = validate(model, teacher, validation_loader, view, device, layout)
    selected = 0
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    history = [{"iteration": 0, "validation_mse": best, "selected_iteration": 0,
                "seconds": time.monotonic() - started}]
    write_json(output / "run.json", info)
    checkpoint = output / "inverse.pt"
    last_checkpoint = output / "last.pt"
    signature = object_hash({key: info[key] for key in FIT_KEYS})
    torch.save({"state_dict": best_state, "selected_iteration": selected, "view": view,
                "fit_signature": signature}, checkpoint)
    train_loader = DataLoader(training, sampler=order.tolist(), **loader_options)
    total, count, iteration = 0.0, 0, 0
    for iteration, images in enumerate(train_loader, 1):
        model.train()
        rgb = images.to(device=device, dtype=torch.float32, non_blocking=True, memory_format=layout) / 255
        if iteration == 1:
            _backend(True)
            feat.verify_exit(teacher, rgb, view)
        _backend(not fast_training)
        total += step(model, teacher, rgb, optimizer, view, True)
        count += 1
        if iteration % validation_every == 0 or iteration == iterations:
            value = validate(model, teacher, validation_loader, view, device, layout)
            if value < best:
                best, selected = value, iteration
                best_state = {key: tensor.detach().cpu().clone()
                              for key, tensor in model.state_dict().items()}
                torch.save({"state_dict": best_state, "selected_iteration": selected, "view": view,
                            "fit_signature": signature}, checkpoint)
            row = {"iteration": iteration, "training_mse": total / count, "validation_mse": value,
                   "selected_iteration": selected, "seconds": time.monotonic() - started}
            history.append(row)
            torch.save({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "iteration": iteration, "view": view, "fit_signature": signature},
                       last_checkpoint)
            write_json(output / "progress.json", {"history": history})
            log(**row)
            total, count = 0.0, 0
    if iteration != iterations:
        raise ValueError("Incomplete iteration budget")
    stat = training.path.stat()
    if (stat.st_size, stat.st_mtime_ns) != training.stat_identity:
        raise ValueError("Training archive changed during fitting")
    info.update(status="complete", fit_signature=signature, selected_iteration=selected,
                validation_mse=best, history=history, seconds=time.monotonic() - started,
                checkpoint={"path": str(checkpoint.resolve()), "sha256": file_hash(checkpoint)},
                last_checkpoint={"path": str(last_checkpoint.resolve()),
                                 "sha256": file_hash(last_checkpoint)})
    write_json(output / "run.json", info)
    return info


def load_inverse(run_directory: Path, device: torch.device,
                 layout: torch.memory_format = torch.contiguous_format) -> tuple[nn.Module, dict]:
    """Load the selected inverse of one exit, verifying the run record against the checkpoint."""
    run_directory = Path(run_directory)
    path = run_directory / "run.json"
    run = read_json(path)
    checkpoint_path = run_directory / Path(run["checkpoint"]["path"]).name
    if run["status"] != "complete" or file_hash(checkpoint_path) != run["checkpoint"]["sha256"]:
        raise ValueError(f"Inverse checkpoint identity differs for {run_directory}")
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if saved["view"] != run["view"] or saved["selected_iteration"] != run["selected_iteration"]:
        raise ValueError("Inverse checkpoint content differs from its run record")
    model = SimpleInverse(depth_of(run["view"]))
    model.load_state_dict(saved["state_dict"])
    model.to(device=device, memory_format=layout).eval()
    return model, {"run_path": str(path.resolve()), "run_sha256": file_hash(path),
                   "checkpoint": run["checkpoint"], "selected_iteration": run["selected_iteration"],
                   "view": run["view"], "teacher": run["teacher"]}


def top5(logits: torch.Tensor) -> list[dict]:
    probability, indices = logits.float().softmax(0).topk(5)
    names = feat.weights_enum().meta["categories"]
    return [{"index": int(index), "name": names[int(index)], "probability": float(value)}
            for index, value in zip(indices.cpu(), probability.cpu())]


@torch.inference_mode()
def apply_to_sources(output: Path, *, run_directory: Path, sources: Path, device: torch.device,
                     batch_size: int = 32) -> dict:
    """Invert an explicit list of prepared PNGs and export the reconstructions."""
    output = Path(output)
    run = read_json(Path(run_directory) / "run.json")
    rows = read_json(sources)
    if not isinstance(rows, list) or not rows or len({str(row["id"]) for row in rows}) != len(rows):
        raise ValueError("Require a nonempty source list with unique IDs")
    arrays, inputs = [], []
    for row in rows:
        path = Path(row["path"])
        path = path if path.is_absolute() else Path(sources).parent / path
        if file_hash(path) != row["sha256"]:
            raise ValueError(f"Input hash changed for {row['id']}")
        with Image.open(path) as image:
            if image.format != "PNG" or image.mode != "RGB" or image.size != (224, 224):
                raise ValueError("Source must already be an exact 224x224 RGB PNG")
            array = np.asarray(image).copy()
        arrays.append(torch.from_numpy(array).permute(2, 0, 1))
        inputs.append({"source": row, "resolved_path": str(path.resolve()),
                       "pixel_sha256": bytes_hash(array.tobytes())})
    _backend(True)
    layout = (torch.channels_last if run["configuration"]["memory_format"] == "channels_last"
              else torch.contiguous_format)
    model, identity = load_inverse(run_directory, device, layout)
    teacher = feat.FrozenTeacher(device).to(memory_format=layout)
    if teacher.provenance() != run["teacher"]:
        raise ValueError("Current teacher differs from the inverse fitting teacher")
    output.mkdir(parents=True, exist_ok=True)
    images = torch.stack(arrays)
    exported = []
    for begin in range(0, len(images), batch_size):
        rgb = images[begin:begin + batch_size].to(device=device, dtype=torch.float32,
                                                  memory_format=layout) / 255
        feat.verify_exit(teacher, rgb, run["view"])
        prediction = model(feat.native_exit(teacher, rgb, run["view"]))
        original = teacher(rgb, depth="all", logits=True)["logits"]
        if not torch.isfinite(prediction).all() or not torch.isfinite(original).all():
            raise FloatingPointError("Nonfinite original prediction or inverse RGB")
        pixels = (prediction.clamp(0, 1) * 255).round().to(torch.uint8)
        after = teacher(pixels.float() / 255, depth="all", logits=True)["logits"]
        for offset, pixel in enumerate(pixels.cpu()):
            index = begin + offset
            path = output / f"{index:04d}.png"
            array = pixel.permute(1, 2, 0).numpy()
            Image.fromarray(array).save(path)
            exported.append({**inputs[index], "output_path": path.name,
                             "output_sha256": file_hash(path),
                             "output_pixel_sha256": bytes_hash(array.tobytes()),
                             "original_top5": top5(original[offset]),
                             "reconstructed_top5": top5(after[offset])})
    preview = {"status": "complete", "view": run["view"],
               "source_list_sha256": file_hash(sources), "fit_run_sha256": identity["run_sha256"],
               "selected_checkpoint": run["checkpoint"],
               "selected_iteration": run["selected_iteration"], "teacher": teacher.provenance(),
               "runtime": feat.runtime_provenance(),
               "precision": "Deterministic FP32 complete features and inverse. The network reads the "
                            "exact exported 8-bit RGB.",
               "memory_format": run["configuration"]["memory_format"], "sources": exported}
    write_json(output / "preview.json", preview)
    log(stage="apply", view=run["view"], images=len(exported))
    return preview


def convert_export(output: Path, *, preview: Path, records: list[dict], sources: list[dict],
                   population: str, cub_root: Path, inventory: Path) -> dict:
    """Bind exported reconstructions back to their source records and write an inventory."""
    preview_path = Path(preview)
    exported = read_json(preview_path)
    if exported["status"] != "complete":
        raise ValueError("Incomplete RGB export")
    entries = exported["sources"]
    indexed = {row["source"]["id"]: row for row in entries}
    if len(indexed) != len(entries) or set(indexed) != {row["id"] for row in sources}:
        raise ValueError("Export membership differs from the requested population")
    if len({row["output_path"] for row in entries}) != len(entries):
        raise ValueError("Multiple sources refer to the same exported PNG")
    inverse = {key: exported[key] for key in
               ("view", "fit_run_sha256", "selected_checkpoint", "selected_iteration", "teacher",
                "runtime")}
    converted = []
    for original, source in zip(records, sources):
        entry = indexed[original["observation_id"]]
        if (entry["source"] != source or entry["resolved_path"] != source["path"]
                or entry["pixel_sha256"] != original["pixel_sha256"]):
            raise ValueError("Reconstruction lost its original source identity")
        path = preview_path.parent / entry["output_path"]
        check_png(path, entry["output_sha256"], entry["output_pixel_sha256"])
        row = dict(original)
        row.update(original_input_path=original["input_path"],
                   original_input_sha256=original["input_sha256"],
                   original_pixel_sha256=original["pixel_sha256"],
                   input_path=entry["output_path"], input_sha256=entry["output_sha256"],
                   pixel_sha256=entry["output_pixel_sha256"], input_bytes=path.stat().st_size)
        converted.append(row)
    value = {"population": population, "records": converted, "inverse": inverse,
             "counts": {split: sum(row["split"] == split for row in converted)
                        for split in sorted({row["split"] for row in converted})},
             "data_root": str(preview_path.parent.resolve()),
             "original_data_root": str(Path(cub_root).resolve()),
             "original_inventory": {"path": str(Path(inventory).resolve()),
                                    "sha256": file_hash(inventory)},
             "preview": {"path": str(preview_path.resolve()), "sha256": file_hash(preview_path)},
             "label_interface": "Fine labels remain cub_class_id minus one. No ImageNet label is "
                                "assigned."}
    write_json(Path(output) / "inventory.json", value)
    return value
