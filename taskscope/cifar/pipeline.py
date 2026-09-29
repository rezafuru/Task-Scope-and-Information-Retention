"""CIFAR-100 teacher, fine readouts, and feature codecs for the local preservation study.

A ResNet-18 teacher is trained on the twenty coarse labels only. Its early
features (after layer2) and its coarse logits are cached, and fine-label
readouts are fitted on both. A small codec then compresses one of those two
observations under one of three losses (early-feature squared error, logit
squared error, or a coarse/fine risk family), and the decoded early tensor is
scored through the teacher's late stack and through refitted readouts.

Stages, in order:

  teacher                fit the coarse-label teacher on the fixed split
  extract                cache early features and logits for train and validation
  readouts               fit linear and MLP fine readouts on the cached observations
  codec                  fit one feature codec at one rate weight
  decode                 serialize and decode validation messages, cache the result
  extract-test           cache test observations
  decode-test            serialize and decode test messages
  score-validation       score readouts on a cached observation
  score-test             the same on the test split
  score-logit-validation score the reference logit readouts on a decoded cache
  score-logit-test       the same on the test split
  analyze-test           pool the assessed runs into groups and bootstrap intervals

The teacher, extract, and extract-test stages need the CIFAR-100 archive and all
fitting stages need a GPU to finish in reasonable time. Rates reported by
``decode`` are complete message lengths (header, hyperprior, and main payload),
not entropy estimates.
"""

from __future__ import annotations

import importlib
import json
import math
import pickle
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

MEAN = (0.5071, 0.4867, 0.4408)
STD = (0.2675, 0.2565, 0.2761)
SPLIT_SEED = 20260905
BOOTSTRAP_SEED = 20260906

DATA_STAGES = ("teacher", "extract", "extract-test")
STAGES = (
    "teacher", "extract", "readouts", "codec", "decode", "extract-test", "decode-test",
    "score-test", "score-validation", "score-logit-validation", "score-logit-test", "analyze-test",
)
DEFAULT_EPOCHS = {"teacher": 200, "readouts": 100, "codec": 60}
DEFAULT_LR = {"teacher": 0.1, "readouts": 0.001, "codec": 0.0002}

# The hyperprior entropy model and the packet container are shared with the Taskonomy codec.
CODING_MODULE = "taskscope.taskonomy.models"


@dataclass
class Settings:
    """One stage invocation. The recorded fields become the stored run configuration."""

    stage: str
    output: str
    data: str | None = None
    device: str = "auto"
    epochs: int | None = None
    lr: float | None = None
    batch_size: int = 256
    seed: int = 17
    observation: str = "early"
    objective: str = "early"
    beta: float = 0.1
    checkpoint: str | None = None
    decoded_readouts: bool = False
    initialization: str | None = None
    run_tag: str = ""
    reference_output: str | None = None
    readout_output: str | None = None
    pool_decoded_cache: bool = False

    def __post_init__(self) -> None:
        if self.stage not in STAGES:
            raise ValueError(f"Unknown stage: {self.stage}")
        if self.epochs is None:
            self.epochs = DEFAULT_EPOCHS.get(self.stage, 1)
        if self.lr is None:
            self.lr = DEFAULT_LR.get(self.stage, 0.0)

    @property
    def output_path(self) -> Path:
        return Path(self.output)

    @property
    def readout_path(self) -> Path:
        return Path(self.readout_output) if self.readout_output else self.output_path

    @property
    def reference_path(self) -> Path:
        return Path(self.reference_output) if self.reference_output else self.output_path


def entropy_coding():
    """Return the module holding ``HyperpriorStream``, ``build_packet``, ``parse_packet``.

    Imported on demand so that the scoring and aggregation stages run without CompressAI.
    """
    return importlib.import_module(CODING_MODULE)


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def autocast(device: torch.device):
    """Mixed precision on the devices that support bfloat16 autocast, otherwise a no-op."""
    return torch.autocast(device.type, dtype=torch.bfloat16,
                          enabled=device.type in ("cuda", "cpu"))


def _prepare(device: torch.device) -> None:
    torch.set_num_threads(8)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def record(path: Path, value) -> None:
    line = json.dumps(value, allow_nan=False)
    with path.open("a") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_teacher() -> nn.Module:
    """ResNet-18 with the stem adapted to 32-pixel inputs and a twenty-way head."""
    from torchvision.models import resnet18

    model = resnet18(weights=None, num_classes=20)
    model.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def early_features(model, image):
    value = model.maxpool(model.relu(model.bn1(model.conv1(image))))
    return model.layer2(model.layer1(value))


def late_features(model, early):
    value = model.layer4(model.layer3(early))
    return model.fc(model.avgpool(value).flatten(1))


def load_data(root: Path, output: Path, device: torch.device) -> dict:
    """Download CIFAR-100 into ``root`` and return the fixed coarse-stratified splits.

    ``root`` is the directory that holds ``cifar-100-python/``. The 5000-image
    validation split is drawn once, stored next to the run, and rechecked on every
    later call so that a run cannot silently change its split.
    """
    from torchvision.datasets import CIFAR100

    CIFAR100(root=str(root), train=True, download=True)
    directory = root / "cifar-100-python"
    with (directory / "train").open("rb") as handle:
        training = pickle.load(handle, encoding="latin1")
    with (directory / "test").open("rb") as handle:
        testing = pickle.load(handle, encoding="latin1")
    labels = np.asarray(training["coarse_labels"])
    rng = np.random.default_rng(SPLIT_SEED)
    validation = np.concatenate(
        [rng.permutation(np.flatnonzero(labels == q))[:250] for q in range(20)])
    train = np.setdiff1d(np.arange(50000), validation)
    output.mkdir(parents=True, exist_ok=True)
    split_path = output / "split_indices.npz"
    if split_path.exists():
        existing = np.load(split_path)
        if (not np.array_equal(existing["train"], train)
                or not np.array_equal(existing["validation"], validation)):
            raise ValueError("Stored CIFAR split differs from the fixed coarse-stratified split")
    else:
        np.savez(split_path, train=train, validation=validation)
    result = {}
    for name, source, indices in (("train", training, train),
                                  ("validation", training, validation),
                                  ("test", testing, np.arange(10000))):
        images = np.asarray(source["data"], dtype=np.uint8).reshape(-1, 3, 32, 32)[indices].copy()
        result[name] = {
            "images": torch.from_numpy(images).to(device),
            "coarse": torch.tensor(np.asarray(source["coarse_labels"])[indices], device=device),
            "fine": torch.tensor(np.asarray(source["fine_labels"])[indices], device=device),
        }
    return result


def prepare_images(images, augment: bool = False):
    """Scale to unit range, optionally pad-crop and flip, then normalize per channel."""
    value = images.float().div_(255)
    if augment:
        padded = F.pad(value, (4, 4, 4, 4), mode="constant", value=0)
        count = len(value)
        offsets = torch.randint(9, (count, 2), device=value.device)
        rows = torch.arange(32, device=value.device)[None, :, None] + offsets[:, 0, None, None]
        cols = torch.arange(32, device=value.device)[None, None, :] + offsets[:, 1, None, None]
        value = padded.permute(0, 2, 3, 1)[
            torch.arange(count, device=value.device)[:, None, None], rows, cols].permute(0, 3, 1, 2)
        flip = torch.rand(count, 1, 1, 1, device=value.device) < 0.5
        value = torch.where(flip, value.flip(-1), value)
    mean = value.new_tensor(MEAN)[None, :, None, None]
    std = value.new_tensor(STD)[None, :, None, None]
    return ((value - mean) / std).contiguous(memory_format=torch.channels_last)


@torch.no_grad()
def evaluate_teacher(model, data, device: torch.device, batch_size: int = 512):
    model.eval()
    total_loss = 0.0
    correct = 0
    predictions = []
    for begin in range(0, len(data["images"]), batch_size):
        image = prepare_images(data["images"][begin:begin + batch_size])
        target = data["coarse"][begin:begin + batch_size]
        with autocast(device):
            output = model(image)
        total_loss += F.cross_entropy(output.float(), target, reduction="sum").item()
        correct += (output.argmax(1) == target).sum().item()
        predictions.append(output.argmax(1).cpu())
    count = len(data["images"])
    return {"accuracy": correct / count, "cross_entropy": total_loss / count}, torch.cat(predictions)


def train_teacher(settings: Settings, data: dict, output: Path, device: torch.device) -> None:
    """Fit the coarse-label teacher, keeping the best validation checkpoint."""
    seed_all(settings.seed)
    model = make_teacher().to(device).to(memory_format=torch.channels_last)
    optimizer = torch.optim.SGD(model.parameters(), lr=settings.lr, momentum=0.9, weight_decay=5e-4)
    count = len(data["train"]["images"])
    best = (-1.0, -float("inf"))
    start = time.perf_counter()
    metadata = asdict(settings)
    metadata.update(train_images=count, validation_images=len(data["validation"]["images"]),
                    teacher_supervision="coarse only",
                    parameters=sum(p.numel() for p in model.parameters()),
                    torch_version=torch.__version__)
    write_json(output / "teacher_config.json", metadata)
    for epoch in range(settings.epochs):
        epoch_start = time.perf_counter()
        if epoch < 5:
            lr = settings.lr * (epoch + 1) / 5
        else:
            lr = settings.lr * 0.5 * (
                1 + math.cos(math.pi * (epoch - 5) / max(1, settings.epochs - 5)))
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        permutation = torch.randperm(count, device=device)
        train_loss = 0.0
        train_correct = 0
        for begin in range(0, count, settings.batch_size):
            indices = permutation[begin:begin + settings.batch_size]
            image = prepare_images(data["train"]["images"][indices], augment=True)
            target = data["train"]["coarse"][indices]
            optimizer.zero_grad(set_to_none=True)
            with autocast(device):
                logits = model(image)
                loss = F.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(indices)
            train_correct += (logits.argmax(1) == target).sum().item()
        metrics, _ = evaluate_teacher(model, data["validation"], device)
        current = (metrics["accuracy"], -metrics["cross_entropy"])
        if current > best:
            best = current
            torch.save({"model": model.state_dict(), "epoch": epoch + 1,
                        "validation": metrics, "config": metadata}, output / "teacher_best.pt")
        if device.type == "cuda":
            torch.cuda.synchronize()
        record(output / "teacher_training.jsonl", {
            "epoch": epoch + 1, "lr": lr, "train_accuracy_augmented": train_correct / count,
            "train_cross_entropy": train_loss / count, "validation": metrics,
            "epoch_seconds": time.perf_counter() - epoch_start,
            "elapsed_seconds": time.perf_counter() - start,
            "image_exposures": (epoch + 1) * count,
        })
    write_json(output / "teacher_finished.json", {
        "elapsed_seconds": time.perf_counter() - start,
        "image_exposures": settings.epochs * count,
        "best_validation_accuracy": best[0], "test_used": False,
    })


def load_teacher(output: Path, device: torch.device):
    checkpoint = torch.load(output / "teacher_best.pt", map_location="cpu", weights_only=False)
    model = make_teacher().to(device).to(memory_format=torch.channels_last)
    model.load_state_dict(checkpoint["model"])
    model.eval().requires_grad_(False)
    return model, checkpoint


@torch.no_grad()
def extract_raw(settings: Settings, data: dict, output: Path, device: torch.device) -> None:
    """Cache the teacher's early features and coarse logits for the requested splits."""
    teacher, checkpoint = load_teacher(output, device)
    start = time.perf_counter()
    summaries = {"teacher_epoch": checkpoint["epoch"], "teacher_validation": checkpoint["validation"]}
    splits = ("test",) if settings.stage == "extract-test" else ("train", "validation")
    for split in splits:
        early_values = []
        late_values = []
        for begin in range(0, len(data[split]["images"]), settings.batch_size):
            image = prepare_images(data[split]["images"][begin:begin + settings.batch_size])
            with autocast(device):
                early = early_features(teacher, image)
                late = late_features(teacher, early)
            early_values.append(early.half().cpu())
            late_values.append(late.float().cpu())
        values = {"early": torch.cat(early_values), "late": torch.cat(late_values),
                  "coarse": data[split]["coarse"].cpu(), "fine": data[split]["fine"].cpu()}
        torch.save(values, output / f"raw_{split}.pt")
        summaries[split] = {
            "count": len(values["coarse"]),
            "coarse_accuracy": (values["late"].argmax(1) == values["coarse"]).float().mean().item(),
            "early_dimensions": list(values["early"].shape[1:]),
            "late_dimensions": list(values["late"].shape[1:]),
        }
    summaries["elapsed_seconds"] = time.perf_counter() - start
    summaries["test_used"] = settings.stage == "extract-test"
    summary_name = "raw_test_summary.json" if summaries["test_used"] else "raw_summary.json"
    write_json(output / summary_name, summaries)
    print(json.dumps(summaries), flush=True)


class FineReadout(nn.Module):
    """Fine-label head over a pooled early tensor or over the twenty coarse logits."""

    def __init__(self, feature: str, capacity: str):
        super().__init__()
        self.feature = feature
        dimensions = 128 * 4 * 4 if feature == "early" else 20
        if capacity == "linear":
            self.net = nn.Linear(dimensions, 100)
        else:
            self.net = nn.Sequential(
                nn.Linear(dimensions, 512), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(512, 512), nn.ReLU(), nn.Dropout(0.2), nn.Linear(512, 100))

    def forward(self, feature):
        if feature.ndim == 4:
            feature = F.adaptive_avg_pool2d(feature, 4).flatten(1)
        return self.net(feature)


def pool_for_readout(value):
    if value.ndim == 4:
        return F.adaptive_avg_pool2d(value.float(), 4).flatten(1)
    return value.float()


def fit_raw_readouts(settings: Settings, output: Path, device: torch.device) -> None:
    """Fit linear and MLP fine readouts on the cached observations, selecting on validation."""
    start = time.perf_counter()
    readout_output = settings.readout_path
    readout_output.mkdir(parents=True, exist_ok=True)
    caches = {name: torch.load(output / f"raw_{name}.pt", map_location="cpu", weights_only=False)
              for name in ("train", "validation")}
    results = {}
    features = ("early",) if settings.decoded_readouts else ("early", "late")
    for feature in features:
        values = {name: pool_for_readout(caches[name][feature]).to(device) for name in caches}
        labels = {name: caches[name]["fine"].to(device) for name in caches}
        mean = values["train"].mean(0, keepdim=True)
        scale = values["train"].std(0, keepdim=True).clamp_min(1e-3)
        values = {name: (value - mean) / scale for name, value in values.items()}
        for capacity in ("linear", "mlp"):
            seed_all(settings.seed)
            model = FineReadout(feature, capacity).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=settings.lr, weight_decay=1e-4)
            best = (-1.0, -float("inf"))
            name = f"{feature}_{capacity}"
            for epoch in range(settings.epochs):
                model.train()
                train_loss = 0.0
                train_correct = 0
                lr = settings.lr * 0.5 * (1 + math.cos(math.pi * epoch / settings.epochs))
                for group in optimizer.param_groups:
                    group["lr"] = lr
                permutation = torch.randperm(len(labels["train"]), device=device)
                for begin in range(0, len(permutation), settings.batch_size):
                    index = permutation[begin:begin + settings.batch_size]
                    optimizer.zero_grad(set_to_none=True)
                    with autocast(device):
                        prediction = model(values["train"][index])
                        loss = F.cross_entropy(prediction, labels["train"][index])
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item() * len(index)
                    train_correct += (prediction.argmax(1) == labels["train"][index]).sum().item()
                model.eval()
                with torch.no_grad(), autocast(device):
                    prediction = model(values["validation"])
                    accuracy = (prediction.argmax(1) == labels["validation"]).float().mean().item()
                    ce = F.cross_entropy(prediction.float(), labels["validation"]).item()
                current = (accuracy, -ce)
                if current > best:
                    best = current
                    torch.save({"model": model.state_dict(), "feature": feature,
                                "capacity": capacity, "mean": mean.cpu(), "scale": scale.cpu(),
                                "epoch": epoch + 1, "validation_accuracy": accuracy,
                                "validation_cross_entropy": ce, "config": asdict(settings)},
                               readout_output / f"readout_{name}.pt")
                record(readout_output / f"readout_{name}.jsonl", {
                    "epoch": epoch + 1,
                    "train_accuracy": train_correct / len(labels["train"]),
                    "train_cross_entropy": train_loss / len(labels["train"]),
                    "validation_accuracy": accuracy, "validation_cross_entropy": ce,
                    "elapsed_seconds": time.perf_counter() - start,
                })
            results[name] = {
                "validation_accuracy": best[0], "validation_cross_entropy": -best[1],
                "parameters": sum(p.numel() for p in model.parameters()),
                "image_exposures": settings.epochs * len(labels["train"]),
            }
    results["elapsed_seconds"] = time.perf_counter() - start
    results["test_used"] = False
    write_json(readout_output / "raw_readouts.json", results)


class FeatureCodec(nn.Module):
    """Analysis transform, conditional entropy bottleneck, and synthesis to the early tensor.

    Both observations decode to the early tensor shape, so the same synthesis transform
    and the same downstream scoring apply to either encoder input.
    """

    def __init__(self, observation: str):
        super().__init__()
        coding = entropy_coding()
        self.observation = observation
        if observation == "early":
            self.analysis = nn.Sequential(
                nn.Conv2d(128, 128, 5, stride=2, padding=2), nn.GELU(),
                nn.Conv2d(128, 64, 5, stride=2, padding=2))
        else:
            self.analysis = nn.Sequential(
                nn.Linear(20, 512), nn.GELU(), nn.Linear(512, 64 * 4 * 4),
                nn.Unflatten(1, (64, 4, 4)))
        self.stream = coding.HyperpriorStream(64)
        self.synthesis = nn.Sequential(
            nn.ConvTranspose2d(64, 128, 4, stride=2, padding=1), nn.GELU(),
            nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1))

    def forward(self, observation):
        latent = self.analysis(observation)
        decoded, likelihoods = self.stream(latent.float())
        rate = sum(-value.clamp_min(1e-9).log2().sum()
                   for value in likelihoods.values()) / (len(observation) * 32 * 32)
        return self.synthesis(decoded), rate

    @torch.no_grad()
    def code(self, observation):
        """Serialize one batch to a packet and decode it back, checking the round trip."""
        coding = entropy_coding()
        latent = self.analysis(observation)
        compressed, hyper, main = self.stream.compress(latent.float())
        packet = coding.build_packet("H_R", "C", latent.shape, (1, 1), hyper, main)
        parsed = coding.parse_packet(packet.packet)
        decoded = self.stream.decompress(parsed.hyper_payload, parsed.main_payload,
                                         parsed.z_shape)
        if not torch.equal(compressed, decoded):
            raise RuntimeError("Decoded CIFAR latent differs from the encoded reconstruction")
        return self.synthesis(decoded), packet


def load_fine_reference(output: Path, device: torch.device):
    checkpoint = torch.load(output / "readout_early_mlp.pt", map_location="cpu", weights_only=False)
    model = FineReadout("early", "mlp").to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval().requires_grad_(False)
    return model, checkpoint["mean"].to(device), checkpoint["scale"].to(device)


def fine_prediction(head, value, mean, scale):
    pooled = F.adaptive_avg_pool2d(value, 4).flatten(1)
    return head((pooled - mean) / scale)


def coding_distortion(decoded, original, teacher, fine_head, mean, scale, objective, normalizers):
    """Distortion of one decoded batch under the requested loss, scaled by its normalizer."""
    if objective == "early":
        return F.mse_loss(decoded, original["early"].float()) / normalizers["early_variance"]
    late = late_features(teacher, decoded)
    if objective == "late":
        return F.mse_loss(late, original["late"].float()) / normalizers["late_variance"]
    coarse_loss = F.cross_entropy(late, original["coarse"])
    fine_loss = F.cross_entropy(fine_prediction(fine_head, decoded, mean, scale), original["fine"])
    return 0.5 * (coarse_loss / normalizers["coarse_cross_entropy"]
                  + fine_loss / normalizers["fine_cross_entropy"])


@torch.no_grad()
def evaluate_codec(model, cache, teacher, fine_head, mean, scale, settings: Settings,
                   normalizers, device: torch.device):
    model.eval()
    count = len(cache["coarse"])
    totals = {"distortion": 0.0, "estimated_bpp": 0.0, "coarse_accuracy": 0.0,
              "coarse_teacher_agreement": 0.0, "fine_accuracy_fixed_readout": 0.0}
    for begin in range(0, count, settings.batch_size):
        batch = {key: value[begin:begin + settings.batch_size].to(device)
                 for key, value in cache.items()}
        size = len(batch["coarse"])
        decoded, rate = model(batch[settings.observation].float())
        distortion = coding_distortion(decoded, batch, teacher, fine_head, mean, scale,
                                       settings.objective, normalizers)
        coarse = late_features(teacher, decoded)
        fine = fine_prediction(fine_head, decoded, mean, scale)
        totals["distortion"] += distortion.item() * size
        totals["estimated_bpp"] += rate.item() * size
        totals["coarse_accuracy"] += (coarse.argmax(1) == batch["coarse"]).sum().item()
        totals["coarse_teacher_agreement"] += (coarse.argmax(1) == batch["late"].argmax(1)).sum().item()
        totals["fine_accuracy_fixed_readout"] += (fine.argmax(1) == batch["fine"]).sum().item()
    return {key: value / count for key, value in totals.items()}


def codec_normalizers(output: Path, training, teacher_checkpoint) -> dict:
    """Constant-predictor risks on the training split, written once per teacher directory."""
    normalizer_path = output / "codec_normalizers.json"
    if normalizer_path.exists():
        return json.loads(normalizer_path.read_text())
    with torch.no_grad():
        early_variance = training["early"].float().var(dim=0, correction=0).mean().item()
        late_variance = training["late"].var(dim=0, correction=0).mean().item()
        entropies = {}
        for target, classes in (("coarse", 20), ("fine", 100)):
            probability = torch.bincount(training[target], minlength=classes).float()
            probability = probability / len(training[target])
            entropies[target] = -(probability * probability.clamp_min(1e-12).log()).sum().item()
        normalizers = {
            "early_variance": max(early_variance, 1e-6), "late_variance": max(late_variance, 1e-6),
            "coarse_cross_entropy": entropies["coarse"], "fine_cross_entropy": entropies["fine"],
            "source": "training split constant-predictor risks",
            "teacher_epoch": teacher_checkpoint["epoch"],
        }
    write_json(normalizer_path, normalizers)
    return normalizers


def train_codec(settings: Settings, output: Path, device: torch.device) -> None:
    """Fit one codec at one rate weight, selecting on the validation coding objective."""
    seed_all(settings.seed)
    teacher, teacher_checkpoint = load_teacher(output, device)
    fine_head, mean, scale = load_fine_reference(output, device)
    caches = {name: torch.load(output / f"raw_{name}.pt", map_location="cpu", weights_only=False)
              for name in ("train", "validation")}
    training = {key: value.to(device) for key, value in caches["train"].items()}
    normalizers = codec_normalizers(output, training, teacher_checkpoint)
    model = FeatureCodec(settings.observation).to(device)
    initialization = None
    if settings.initialization:
        initialization = torch.load(settings.initialization, map_location="cpu", weights_only=False)
        if initialization["config"]["observation"] != settings.observation:
            raise ValueError("Full codec initialization requires the same encoder observation")
        model.load_state_dict(initialization["model"])
    named = dict(model.named_parameters())
    main_parameters = [value for name, value in named.items() if not name.endswith(".quantiles")]
    auxiliary_parameters = [value for name, value in named.items() if name.endswith(".quantiles")]
    optimizer = torch.optim.Adam(main_parameters, lr=settings.lr)
    auxiliary = torch.optim.Adam(auxiliary_parameters, lr=1e-3)
    run = output / (f"codec_{settings.observation}_{settings.objective}"
                    f"_b{settings.beta:g}_s{settings.seed}")
    if settings.run_tag:
        run = run.with_name(run.name + "_" + settings.run_tag)
    run.mkdir(parents=True, exist_ok=True)
    config = asdict(settings)
    config.update(normalizers=normalizers,
                  parameters=sum(p.numel() for p in model.parameters()),
                  train_images=len(training["coarse"]), test_used=False)
    if initialization is not None:
        # A warm start inherits the ancestor's exposure count so that the fitting budget
        # reported for the run covers every image the weights have already seen.
        ancestor_config = initialization["config"]
        selected_exposures = initialization["epoch"] * ancestor_config.get(
            "train_images", len(training["coarse"]))
        ancestry = ancestor_config.get("initialization_weight_ancestry_exposures", 0) + selected_exposures
        finished_path = Path(settings.initialization).parent / "finished.json"
        completed = (json.loads(finished_path.read_text())["image_exposures"]
                     if finished_path.exists() else None)
        ancestor_completed = ancestor_config.get("initialization_completed_training_exposures", 0)
        total_completed = (completed + ancestor_completed
                           if completed is not None and ancestor_completed is not None else None)
        config.update(initialization_selected_epoch=initialization["epoch"],
                      initialization_weight_ancestry_exposures=ancestry,
                      initialization_completed_training_exposures=total_completed)
    write_json(run / "config.json", config)
    count = len(training["coarse"])
    best = float("inf")
    start = time.perf_counter()
    for epoch in range(settings.epochs):
        model.train()
        train_distortion = torch.zeros((), device=device)
        train_rate = torch.zeros((), device=device)
        lr = settings.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * epoch / settings.epochs)))
        for group in optimizer.param_groups:
            group["lr"] = lr
        permutation = torch.randperm(count, device=device)
        for begin in range(0, count, settings.batch_size):
            index = permutation[begin:begin + settings.batch_size]
            batch = {key: value[index] for key, value in training.items()}
            optimizer.zero_grad(set_to_none=True)
            decoded, rate = model(batch[settings.observation].float())
            distortion = coding_distortion(decoded, batch, teacher, fine_head, mean, scale,
                                           settings.objective, normalizers)
            loss = distortion + settings.beta * rate
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite coding objective at epoch {epoch + 1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(main_parameters, 5.0)
            optimizer.step()
            auxiliary.zero_grad(set_to_none=True)
            auxiliary_loss = model.stream.aux_loss()
            auxiliary_loss.backward()
            auxiliary.step()
            train_distortion += distortion.detach() * len(index)
            train_rate += rate.detach() * len(index)
        validation = evaluate_codec(model, caches["validation"], teacher, fine_head, mean, scale,
                                    settings, normalizers, device)
        objective = validation["distortion"] + settings.beta * validation["estimated_bpp"]
        if objective < best:
            best = objective
            torch.save({"model": model.state_dict(), "epoch": epoch + 1,
                        "validation": validation, "config": config}, run / "best.pt")
        record(run / "training.jsonl", {
            "epoch": epoch + 1, "train_distortion": train_distortion.item() / count,
            "train_estimated_bpp": train_rate.item() / count, "validation": validation,
            "objective": objective, "elapsed_seconds": time.perf_counter() - start,
            "image_exposures": (epoch + 1) * count,
        })
    torch.save({"model": model.state_dict(), "epoch": settings.epochs, "validation": validation,
                "config": config, "selection": "fixed completed fitting budget"}, run / "last.pt")
    write_json(run / "finished.json", {
        "elapsed_seconds": time.perf_counter() - start,
        "image_exposures": settings.epochs * count,
        "best_validation_objective": best, "test_used": False,
    })


@torch.no_grad()
def decode_codec(settings: Settings, output: Path, device: torch.device) -> None:
    """Serialize and decode each image, caching the decoded tensors and the packet lengths.

    Training uses deterministic quantization, so validation and test images are coded
    one at a time through the real entropy messages to obtain complete byte counts.
    """
    run = Path(settings.checkpoint).parent
    checkpoint = torch.load(settings.checkpoint, map_location="cpu", weights_only=False)
    observation = checkpoint["config"]["observation"]
    model = FeatureCodec(observation).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    model.stream.update(force=True, update_quantiles=True)
    teacher, _ = load_teacher(output, device)
    fine_head, mean, scale = load_fine_reference(output, device)
    start = time.perf_counter()
    test_used = settings.stage == "decode-test"
    summary = {
        "codec_epoch": checkpoint["epoch"], "test_used": test_used,
        "training_decode": "deterministic quantization",
        "evaluation_decode": "serialized main and hyperprior entropy messages",
        "pooled_early_cache": settings.pool_decoded_cache,
    }
    splits = ("test",) if test_used else ("train", "validation")
    for split in splits:
        decoded_cache_path = run / f"raw_{split}.pt"
        if decoded_cache_path.is_symlink():
            raise ValueError(f"Refusing to replace a linked historical cache: {decoded_cache_path}")
        cache = torch.load(output / f"raw_{split}.pt", map_location="cpu", weights_only=False)
        features = []
        late_values = []
        lengths = []
        fixed_predictions = []
        early_squared_error = 0.0
        late_squared_error = 0.0
        step = settings.batch_size if split == "train" else 1
        for begin in range(0, len(cache["coarse"]), step):
            value = cache[observation][begin:begin + step].to(device).float()
            if split == "train":
                decoded, _ = model(value)
            else:
                decoded, packet = model.code(value)
                lengths.append((packet.total_bytes, packet.header_bytes,
                                packet.hyper_bytes, packet.main_bytes))
            cached_early = pool_for_readout(decoded) if settings.pool_decoded_cache else decoded
            features.append(cached_early.cpu())
            reconstructed_late = late_features(teacher, decoded)
            late_values.append(reconstructed_late.cpu())
            early_squared_error += F.mse_loss(
                decoded, cache["early"][begin:begin + step].to(device).float(),
                reduction="sum").item()
            late_squared_error += F.mse_loss(
                reconstructed_late, cache["late"][begin:begin + step].to(device).float(),
                reduction="sum").item()
            fixed_predictions.append(
                fine_prediction(fine_head, decoded, mean, scale).argmax(1).cpu())
            if split != "train" and (begin + 1) % 1000 == 0:
                record(run / "decode_progress.jsonl", {
                    "split": split, "decoded_images": begin + 1,
                    "elapsed_seconds": time.perf_counter() - start})
        values = {"early": torch.cat(features), "late": torch.cat(late_values),
                  "coarse": cache["coarse"], "fine": cache["fine"]}
        torch.save(values, decoded_cache_path)
        metrics = {
            "coarse_accuracy": (values["late"].argmax(1) == values["coarse"]).float().mean().item(),
            "coarse_teacher_agreement":
                (values["late"].argmax(1) == cache["late"].argmax(1)).float().mean().item(),
            "fine_accuracy_fixed_readout":
                (torch.cat(fixed_predictions) == values["fine"]).float().mean().item(),
            "count": len(values["fine"]),
        }
        metrics.update(early_mse=early_squared_error / cache["early"].numel(),
                       late_mse=late_squared_error / cache["late"].numel())
        if lengths:
            byte_counts = np.asarray(lengths, dtype=np.int32)
            np.savez(run / f"{split}_coded_results.npz", bytes=byte_counts,
                     coarse_prediction=values["late"].argmax(1).numpy(),
                     coarse_teacher_prediction=cache["late"].argmax(1).numpy(),
                     fine_prediction_fixed=torch.cat(fixed_predictions).numpy(),
                     coarse_target=values["coarse"].numpy(), fine_target=values["fine"].numpy())
            metrics.update(mean_packet_bytes=float(byte_counts[:, 0].mean()),
                           actual_bpp=float(byte_counts[:, 0].mean() * 8 / (32 * 32)),
                           mean_header_bytes=float(byte_counts[:, 1].mean()),
                           mean_hyper_bytes=float(byte_counts[:, 2].mean()),
                           mean_main_bytes=float(byte_counts[:, 3].mean()))
        summary[split] = metrics
        record(run / "decode_progress.jsonl", {
            "split": split, "metrics": metrics, "elapsed_seconds": time.perf_counter() - start})
    summary["elapsed_seconds"] = time.perf_counter() - start
    summary_name = "coded_test_summary.json" if test_used else "coded_summary.json"
    write_json(run / summary_name, summary)


@torch.no_grad()
def score_readouts(settings: Settings, output: Path, device: torch.device) -> None:
    """Score the fitted fine readouts on a cached observation and store their logits."""
    split = "validation" if settings.stage == "score-validation" else "test"
    readout_output = settings.readout_path
    cache = torch.load(output / f"raw_{split}.pt", map_location="cpu", weights_only=False)
    results = {f"{split}_images": len(cache["fine"]), "test_used": split == "test", "split": split}
    arrays = {"coarse_target": cache["coarse"].numpy(), "fine_target": cache["fine"].numpy(),
              "coarse_prediction": cache["late"].argmax(1).numpy()}
    features = ("early",) if settings.decoded_readouts else ("early", "late")
    results["coarse_accuracy"] = (cache["late"].argmax(1) == cache["coarse"]).float().mean().item()
    for feature in features:
        for capacity in ("linear", "mlp"):
            name = f"{feature}_{capacity}"
            checkpoint = torch.load(readout_output / f"readout_{name}.pt", map_location="cpu",
                                    weights_only=False)
            model = FineReadout(feature, capacity).to(device).eval()
            model.load_state_dict(checkpoint["model"])
            mean = checkpoint["mean"].to(device)
            scale = checkpoint["scale"].to(device)
            predictions = []
            for begin in range(0, len(cache["fine"]), settings.batch_size):
                value = pool_for_readout(cache[feature][begin:begin + settings.batch_size]).to(device)
                with autocast(device):
                    predictions.append(model((value - mean) / scale).float().cpu())
            logits = torch.cat(predictions)
            results[name] = {
                "accuracy": (logits.argmax(1) == cache["fine"]).float().mean().item(),
                "cross_entropy": F.cross_entropy(logits, cache["fine"]).item(),
                "selected_epoch": checkpoint["epoch"], "selection_split": "validation",
            }
            arrays[f"{name}_logits"] = logits.numpy()
    np.savez(readout_output / f"{split}_readout_predictions.npz", **arrays)
    write_json(readout_output / f"{split}_readouts.json", results)


@torch.no_grad()
def score_preserved_logit_readouts(settings: Settings, output: Path, device: torch.device) -> None:
    """Score the reference logit readouts, unchanged, on a decoded logit cache."""
    split = "test" if settings.stage == "score-logit-test" else "validation"
    reference = settings.reference_path
    cache = torch.load(output / f"raw_{split}.pt", map_location="cpu", weights_only=False, mmap=True)
    results = {"split": split, "test_used": split == "test", "count": len(cache["fine"]),
               "reference": str(reference), "additional_training": False}
    arrays = {"fine_target": cache["fine"].numpy()}
    for capacity in ("linear", "mlp"):
        checkpoint = torch.load(reference / f"readout_late_{capacity}.pt", map_location="cpu",
                                weights_only=False)
        model = FineReadout("late", capacity).to(device).eval()
        model.load_state_dict(checkpoint["model"])
        mean, scale = checkpoint["mean"].to(device), checkpoint["scale"].to(device)
        predictions = []
        for begin in range(0, len(cache["fine"]), settings.batch_size):
            value = cache["late"][begin:begin + settings.batch_size].to(device).float()
            with autocast(device):
                predictions.append(model((value - mean) / scale).float().cpu())
        logits = torch.cat(predictions)
        results[capacity] = {
            "accuracy": (logits.argmax(1) == cache["fine"]).float().mean().item(),
            "cross_entropy": F.cross_entropy(logits, cache["fine"]).item(),
            "selected_epoch": checkpoint["epoch"], "selection_split": "validation",
        }
        arrays[f"{capacity}_prediction"] = logits.argmax(1).numpy()
        arrays[f"{capacity}_cross_entropy"] = F.cross_entropy(
            logits, cache["fine"], reduction="none").numpy()
    np.savez(output / f"{split}_preserved_logit_score_predictions.npz", **arrays)
    write_json(output / f"{split}_preserved_logit_scores.json", results)


def stratified_bootstrap_means(values, labels, samples: int = 5000, seed: int = BOOTSTRAP_SEED):
    """Resample paired images within fine classes and return the per-column means."""
    values = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    means = np.zeros((samples, values.shape[1]), dtype=np.float64)
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        counts = generator.multinomial(len(indices), np.full(len(indices), 1 / len(indices)),
                                       size=samples)
        means += counts @ values[indices] / len(labels)
    return means


TEST_GROUPS = {
    "preservation_early": [f"codec_early_early_b0.03_s{seed}_controlled" for seed in (17, 23, 29)],
    "preservation_late_rate_match":
        [f"codec_early_late_b0.001_s{seed}_controlled" for seed in (17, 23, 29)],
    "preservation_late_efficient":
        [f"codec_early_late_b0.01_s{seed}_controlled" for seed in (17, 23, 29)],
    "observation_early": ["codec_early_late_b0.001_s17_pressure"]
        + [f"codec_early_late_b0.001_s{seed}_observation" for seed in (23, 29)],
    "observation_late": [f"codec_late_late_b0.001_s{seed}_observation" for seed in (17, 23, 29)],
}

FEASIBILITY_NOTE = (
    "Each accuracy-drop upper endpoint is a one-sided 97.5% percentile bound. Requiring both "
    "provides an approximate Bonferroni 95% joint check for one prespecified candidate's coarse "
    "and fine mean risks, conditional on the three fitted seeds. This is not simultaneous "
    "coverage across candidate groups or a post-selection guarantee for the cheapest candidate. "
    "Mean feasibility does not imply feasibility of every seed. Seeds remain replications, not "
    "candidates selected by test score."
)


def analyze_test_results(output: Path) -> None:
    """Pool the assessed runs by seed, bootstrap paired-image intervals, and list feasible groups."""
    seeds = (17, 23, 29)
    groups = TEST_GROUPS
    raw = np.load(output / "test_readout_predictions.npz")
    coarse_target, fine_target = raw["coarse_target"], raw["fine_target"]
    references = {"coarse_accuracy": raw["coarse_prediction"] == coarse_target,
                  "refit_mlp_accuracy": raw["early_mlp_logits"].argmax(1) == fine_target}
    rows, per_run = [], {}
    evaluated_runs = ({run for runs in groups.values() for run in runs}
                      | {path.parent.name for path in output.glob("codec_*/test_coded_results.npz")})
    for run in sorted(evaluated_runs):
        directory = output / run
        config = json.loads((directory / "config.json").read_text())
        coded_summary = json.loads((directory / "coded_test_summary.json").read_text())
        packets = np.load(directory / "test_coded_results.npz")
        readouts = np.load(directory / "test_readout_predictions.npz")
        preserved = np.load(directory / "test_preserved_logit_score_predictions.npz")
        for predictions in (packets, readouts):
            if (not np.array_equal(predictions["coarse_target"], coarse_target)
                    or not np.array_equal(predictions["fine_target"], fine_target)):
                raise ValueError(f"Misaligned test targets in {run}")
        if not np.array_equal(preserved["fine_target"], fine_target):
            raise ValueError(f"Misaligned preserved-logit targets in {run}")
        metrics = {
            "actual_bpp": packets["bytes"][:, 0] * 8 / (32 * 32),
            "coarse_accuracy": packets["coarse_prediction"] == coarse_target,
            "teacher_agreement": packets["coarse_prediction"] == packets["coarse_teacher_prediction"],
            "fixed_fine_accuracy": packets["fine_prediction_fixed"] == fine_target,
            "refit_linear_accuracy": readouts["early_linear_logits"].argmax(1) == fine_target,
            "refit_mlp_accuracy": readouts["early_mlp_logits"].argmax(1) == fine_target,
            "preserved_logit_linear_accuracy": preserved["linear_prediction"] == fine_target,
            "preserved_logit_mlp_accuracy": preserved["mlp_prediction"] == fine_target,
        }
        per_run[run] = metrics
        row = {key: config[key] for key in ("observation", "objective", "beta", "seed")}
        row.update(run=run, **{key: float(value.mean()) for key, value in metrics.items()})
        row.update(selected_codec_epoch=coded_summary["codec_epoch"], phase_epochs=config["epochs"],
                   initialization=config.get("initialization"),
                   weight_ancestry_exposures=config.get("initialization_weight_ancestry_exposures", 0)
                   + coded_summary["codec_epoch"] * config.get("train_images", 45000))
        rows.append(row)
    summary = {
        "split": "test", "test_used": True, "test_images": len(fine_target),
        "raw_coarse_accuracy": float(references["coarse_accuracy"].mean()),
        "raw_early_fine_mlp": float(references["refit_mlp_accuracy"].mean()),
        "raw_early_fine_linear": float((raw["early_linear_logits"].argmax(1) == fine_target).mean()),
        "raw_late_fine_linear": float((raw["late_linear_logits"].argmax(1) == fine_target).mean()),
        "raw_late_fine_mlp": float((raw["late_mlp_logits"].argmax(1) == fine_target).mean()),
        "declared_coarse_accuracy_loss": 0.03, "declared_fine_accuracy_losses": [0.03, 0.05, 0.10],
        "equivalence_margin": 0.01, "rows": rows,
    }
    write_json(output / "test_summary.json", summary)

    seed_values = {group: {metric: np.stack([per_run[run][metric] for run in runs])
                           for metric in per_run[runs[0]]}
                   for group, runs in groups.items()}
    columns = [("reference", metric) for metric in references]
    values = list(references.values())
    for group, metrics in seed_values.items():
        for metric, value in metrics.items():
            columns.append((group, metric))
            values.append(value.mean(0))
    index = {column: position for position, column in enumerate(columns)}
    matrix = np.column_stack(values)
    bootstrap = stratified_bootstrap_means(matrix, fine_target)

    def interval(value):
        return np.quantile(value, [0.025, 0.975]).tolist()

    report = {
        "split": "test", "test_images": len(fine_target), "codec_seeds": list(seeds),
        "bootstrap_samples": 5000, "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_unit": "paired test images resampled within fine classes",
        "interval_scope": "conditional on the three fitted codec seeds and fixed teacher",
        "seed_variation_scope": "codec initialization and fitting with teacher, data splits, "
                                "and readout seed fixed",
        "equivalence_margin": 0.01, "groups": {}, "comparisons": {}, "feasible_choices": [],
    }
    for group, metrics in seed_values.items():
        estimates = {}
        for metric, value in metrics.items():
            seed_means = value.mean(1)
            estimates[metric] = {"mean": float(seed_means.mean()),
                                 "seed_values": seed_means.tolist(),
                                 "seed_sd": float(seed_means.std(ddof=1)),
                                 "paired_image_ci95": interval(bootstrap[:, index[group, metric]])}
        margins = {}
        for metric, tolerance in (("coarse_accuracy", 0.03), ("refit_mlp_accuracy", None)):
            drop = references[metric].mean() - metrics[metric].mean()
            draw = bootstrap[:, index["reference", metric]] - bootstrap[:, index[group, metric]]
            tolerances = (tolerance,) if tolerance is not None else (0.03, 0.05, 0.10)
            margins[metric] = {
                "accuracy_drop": float(drop), "paired_image_ci95": interval(draw),
                "tolerances": [{"allowance": allowed, "point_margin": float(allowed - drop),
                                "margin_ci95": interval(allowed - draw),
                                "point_feasible": bool(drop <= allowed),
                                "upper_interval_feasible": bool(np.quantile(draw, 0.975) <= allowed)}
                               for allowed in tolerances],
            }
        report["groups"][group] = {"runs": groups[group], "metrics": estimates,
                                   "reference_risk_margins": margins}
    comparisons = {
        "preservation_near_rate": ("preservation_late_rate_match", "preservation_early"),
        "preservation_cheaper": ("preservation_late_efficient", "preservation_early"),
        "observation": ("observation_early", "observation_late"),
        "achieved_family_choice": ("preservation_late_efficient", "observation_late"),
    }
    for name, (first, second) in comparisons.items():
        comparison = {"difference": f"{first} minus {second}", "metrics": {}}
        for metric in seed_values[first]:
            difference = seed_values[first][metric].mean(1) - seed_values[second][metric].mean(1)
            ci = interval(bootstrap[:, index[first, metric]] - bootstrap[:, index[second, metric]])
            estimate = {"mean_difference": float(difference.mean()),
                        "seed_differences": difference.tolist(),
                        "seed_sd": float(difference.std(ddof=1)), "paired_image_ci95": ci}
            if metric in ("coarse_accuracy", "refit_mlp_accuracy"):
                estimate["within_equivalence_margin"] = bool(ci[0] >= -0.01 and ci[1] <= 0.01)
            comparison["metrics"][metric] = estimate
        ratio = bootstrap[:, index[first, "actual_bpp"]] / bootstrap[:, index[second, "actual_bpp"]]
        comparison["rate_ratio"] = {
            "mean_ratio": float(seed_values[first]["actual_bpp"].mean()
                                / seed_values[second]["actual_bpp"].mean()),
            "paired_image_ci95": interval(ratio),
        }
        report["comparisons"][name] = comparison
    for fine_tolerance in (None, 0.03, 0.05, 0.10):
        point, conservative = [], []
        for group, estimate in report["groups"].items():
            margins = estimate["reference_risk_margins"]
            constraints = [margins["coarse_accuracy"]["tolerances"][0]]
            if fine_tolerance is not None:
                constraints.append(next(item for item in margins["refit_mlp_accuracy"]["tolerances"]
                                        if item["allowance"] == fine_tolerance))
            candidate = {"group": group, "mean_actual_bpp": estimate["metrics"]["actual_bpp"]["mean"]}
            if all(item["point_feasible"] for item in constraints):
                point.append(candidate)
            if all(item["upper_interval_feasible"] for item in constraints):
                conservative.append(candidate)
        report["feasible_choices"].append({
            "coarse_tolerance": 0.03, "fine_tolerance": fine_tolerance,
            "point_candidates": sorted(point, key=lambda item: item["mean_actual_bpp"]),
            "upper_interval_candidates": sorted(conservative,
                                                key=lambda item: item["mean_actual_bpp"]),
        })
    report["feasibility_interval_note"] = FEASIBILITY_NOTE
    write_json(output / "focal_uncertainty.json", report)


def run(settings: Settings) -> None:
    """Dispatch one stage after resolving the device and checking the stage's inputs."""
    if settings.stage in DATA_STAGES and not settings.data:
        raise ValueError(f"Stage {settings.stage} needs --data, the directory holding cifar-100-python")
    if settings.stage in ("decode", "decode-test") and settings.checkpoint is None:
        raise ValueError("Decoding needs --checkpoint")
    device = resolve_device(settings.device)
    _prepare(device)
    output = settings.output_path
    if settings.stage == "readouts":
        fit_raw_readouts(settings, output, device)
    elif settings.stage == "codec":
        train_codec(settings, output, device)
    elif settings.stage in ("decode", "decode-test"):
        decode_codec(settings, output, device)
    elif settings.stage in ("score-test", "score-validation"):
        score_readouts(settings, output, device)
    elif settings.stage in ("score-logit-validation", "score-logit-test"):
        score_preserved_logit_readouts(settings, output, device)
    elif settings.stage == "analyze-test":
        analyze_test_results(output)
    else:
        data = load_data(Path(settings.data), output, device)
        if settings.stage == "teacher":
            train_teacher(settings, data, output, device)
        else:
            extract_raw(settings, data, output, device)
