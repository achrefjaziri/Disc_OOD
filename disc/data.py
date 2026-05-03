"""Data loading for ImageNet-only DISC experiments.

Images are resized to the diffusion model resolution, converted to RGB, and
normalized to the range expected by OpenAI-style ImageNet diffusion models:
``[-1, 1]`` by default.

Each dataset item is a dictionary with image tensor and sample metadata:
``dataset_name``, ``ood_family``, ``original_index``, ``label``, ``path``, and
``source_name``. The source field is most useful for ImageNet-C roots that
contain many corruption/severity folders.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision import datasets, transforms


IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png", ".bmp", ".webp"}
SEVERITY_NAMES = {"1", "2", "3", "4", "5"}


@dataclass(frozen=True)
class DatasetInfo:
    """Human-readable metadata attached to a configured dataset."""

    dataset_name: str
    ood_family: str
    role: str


class MetadataImageFolder(Dataset):
    """Wrap `torchvision.datasets.ImageFolder` and add DISC metadata."""

    def __init__(
        self,
        root: str | Path,
        *,
        dataset_name: str,
        ood_family: str,
        role: str,
        transform: Any,
        source_name: str | None = None,
    ):
        self.root = Path(root)
        self.dataset_name = dataset_name
        self.ood_family = ood_family
        self.role = role
        self.source_name = source_name or dataset_name
        self.imagefolder = datasets.ImageFolder(self.root, transform=transform)
        if len(self.imagefolder) == 0:
            raise FileNotFoundError(f"No ImageFolder samples found under {self.root}")

    def __len__(self) -> int:
        return len(self.imagefolder)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image, label = self.imagefolder[index]
        path, _ = self.imagefolder.samples[index]
        return {
            "image": image,
            "dataset_name": self.dataset_name,
            "ood_family": self.ood_family,
            "role": self.role,
            "source_name": self.source_name,
            "original_index": index,
            "index": index,
            "label": int(label),
            "target": int(label),
            "path": path,
        }


class MetadataFlatImageDataset(Dataset):
    """Load a flat directory of images and attach DISC metadata.

    This is needed for ImageNet validation layouts like
    ``ILSVRC/Data/CLS-LOC/val/*.JPEG`` where class labels are unavailable or not
    arranged as `ImageFolder` subdirectories. Labels are set to ``-1`` because
    the DISC evaluations do not use ImageNet class labels.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        dataset_name: str,
        ood_family: str,
        role: str,
        transform: Any,
        source_name: str | None = None,
    ):
        self.root = Path(root)
        self.dataset_name = dataset_name
        self.ood_family = ood_family
        self.role = role
        self.source_name = source_name or dataset_name
        self.transform = transform
        self.samples = sorted(
            p for p in self.root.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.samples:
            raise FileNotFoundError(f"No flat image samples found under {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            image = self.transform(image)
        return {
            "image": image,
            "dataset_name": self.dataset_name,
            "ood_family": self.ood_family,
            "role": self.role,
            "source_name": self.source_name,
            "original_index": index,
            "index": index,
            "label": -1,
            "target": -1,
            "path": str(path),
        }


class MetadataTorchvisionDataset(Dataset):
    """Wrap a torchvision dataset and add DISC metadata."""

    def __init__(
        self,
        dataset: Dataset,
        *,
        dataset_name: str,
        ood_family: str,
        role: str,
        transform: Any,
    ):
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.ood_family = ood_family
        self.role = role
        self.source_name = dataset_name
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image, label = self.dataset[index]
        if hasattr(image, "convert"):
            image = image.convert("RGB")
        image = self.transform(image)
        return {
            "image": image,
            "dataset_name": self.dataset_name,
            "ood_family": self.ood_family,
            "role": self.role,
            "source_name": self.source_name,
            "original_index": index,
            "index": index,
            "label": int(label),
            "target": int(label),
            "path": "",
        }


class MetadataConcatDataset(ConcatDataset):
    """Concatenate metadata datasets while exposing one dataset-level identity."""

    def __init__(self, datasets_: Iterable[Dataset], info: DatasetInfo):
        super().__init__(list(datasets_))
        self.dataset_name = info.dataset_name
        self.ood_family = info.ood_family
        self.role = info.role
        self.source_name = info.dataset_name


class MetadataSubset(Subset):
    """Subset wrapper that preserves dataset-level metadata attributes."""

    def __init__(self, dataset: Dataset, indices: list[int]):
        super().__init__(dataset, indices)
        self.dataset_name = getattr(dataset, "dataset_name", "unknown")
        self.ood_family = getattr(dataset, "ood_family", "unknown")
        self.role = getattr(dataset, "role", "unknown")
        self.source_name = getattr(dataset, "source_name", self.dataset_name)


def build_transform(data_cfg: dict[str, Any]) -> transforms.Compose:
    """Build the preprocessing transform expected by the diffusion model."""
    image_size = int(data_cfg.get("image_size", 64))
    normalization = data_cfg.get("normalization", "minus_one_one")
    ops: list[Any] = [
        transforms.Lambda(lambda image: image.convert("RGB") if hasattr(image, "convert") else image),
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ]
    if normalization in ("minus_one_one", "diffusion", None):
        ops.append(transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))
    elif normalization == "imagenet_standard":
        ops.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    elif normalization == "none":
        pass
    else:
        raise ValueError(
            f"Unsupported normalization '{normalization}'. Use 'minus_one_one' for "
            "model-expected diffusion inputs or 'imagenet_standard' for standard ImageNet inputs."
        )
    return transforms.Compose(ops)


def _has_unresolved_env(value: str) -> bool:
    return "$" in value or "{" in value or "}" in value


def _configured_root(data_cfg: dict[str, Any], key: str) -> Path | None:
    value = data_cfg.get(key)
    if value in (None, ""):
        return None
    value = str(value)
    if _has_unresolved_env(value):
        return None
    root = Path(value)
    if not root.exists():
        raise FileNotFoundError(f"Configured data root for '{key}' does not exist: {root}")
    return root


def _looks_like_imagefolder_leaf(root: Path) -> bool:
    """Return true when `root` looks like class-folder/images."""
    if not root.is_dir():
        return False
    for child in root.iterdir():
        if not child.is_dir():
            continue
        if any(p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS for p in child.iterdir()):
            return True
    return False


def _looks_like_flat_image_dir(root: Path) -> bool:
    """Return true when `root` directly contains image files."""
    return root.is_dir() and any(
        p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        for p in root.iterdir()
    )


def discover_imagenet_c_folders(root: str | Path) -> list[tuple[Path, str]]:
    """Discover one or many ImageNet-C corruption/severity ImageFolder roots.

    Supported layouts:
    - ``root/class_name/*.JPEG`` for one already-selected corruption/severity.
    - ``root/corruption/severity/class_name/*.JPEG`` for standard ImageNet-C.
    - ``root/corruption_severity/class_name/*.JPEG`` for pre-flattened groups.
    """
    root = Path(root)
    if _looks_like_imagefolder_leaf(root):
        return [(root, "imagenet_c")]

    official_layout: list[tuple[Path, str]] = []
    for corruption_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for severity_dir in sorted(p for p in corruption_dir.iterdir() if p.is_dir()):
            if severity_dir.name in SEVERITY_NAMES and _looks_like_imagefolder_leaf(severity_dir):
                source = f"imagenet_c/{corruption_dir.name}/{severity_dir.name}"
                official_layout.append((severity_dir, source))
    if official_layout:
        return official_layout

    flat_layout = [
        (child, f"imagenet_c/{child.name}")
        for child in sorted(p for p in root.iterdir() if p.is_dir())
        if _looks_like_imagefolder_leaf(child)
    ]
    if flat_layout:
        return flat_layout

    raise FileNotFoundError(
        "Could not find ImageNet-C ImageFolder roots. Expected either "
        "root/class/*.JPEG or root/corruption/severity/class/*.JPEG."
    )


def _limit_dataset(dataset: Dataset, max_samples: int | None) -> Dataset:
    if max_samples is None:
        return dataset
    count = min(int(max_samples), len(dataset))
    return MetadataSubset(dataset, list(range(count)))


def _dataset_labels(dataset: Dataset) -> list[int] | None:
    """Return labels without loading images when the wrapper exposes them."""
    if isinstance(dataset, MetadataImageFolder):
        return [int(label) for _, label in dataset.imagefolder.samples]
    if isinstance(dataset, MetadataTorchvisionDataset):
        raw = dataset.dataset
        labels = getattr(raw, "targets", None)
        if labels is None:
            labels = getattr(raw, "labels", None)
        if labels is None:
            return None
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().tolist()
        return [int(label) for label in labels]
    if isinstance(dataset, MetadataConcatDataset):
        labels: list[int] = []
        for child in dataset.datasets:
            child_labels = _dataset_labels(child)
            if child_labels is None:
                return None
            labels.extend(child_labels)
        return labels
    return None


def _limit_dataset_per_class(dataset: Dataset, max_samples_per_class: int | None) -> Dataset:
    """Keep at most N samples for each available label."""
    if max_samples_per_class is None:
        return dataset
    labels = _dataset_labels(dataset)
    if labels is None:
        return dataset
    limit = int(max_samples_per_class)
    if limit <= 0:
        raise ValueError("data.max_samples_per_class must be positive when set.")
    by_label: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        if label < 0:
            continue
        bucket = by_label.setdefault(label, [])
        if len(bucket) < limit:
            bucket.append(idx)
    if not by_label:
        return dataset
    indices = sorted(idx for bucket in by_label.values() for idx in bucket)
    return MetadataSubset(dataset, indices)


def _build_imagenet_c(root: Path, transform: Any) -> Dataset:
    folders = discover_imagenet_c_folders(root)
    info = DatasetInfo(dataset_name="imagenet_c", ood_family="ImageNet-C", role="ood")
    datasets_ = [
        MetadataImageFolder(
            folder,
            dataset_name=info.dataset_name,
            ood_family=info.ood_family,
            role=info.role,
            transform=transform,
            source_name=source,
        )
        for folder, source in folders
    ]
    if len(datasets_) == 1:
        return datasets_[0]
    return MetadataConcatDataset(datasets_, info)


def build_datasets(cfg: dict[str, Any]) -> dict[str, Dataset]:
    """Build all configured datasets without constructing dataloaders."""
    data_cfg = dict(cfg["data"] if "data" in cfg else cfg)
    transform = build_transform(data_cfg)
    max_samples = data_cfg.get("max_samples_per_dataset")
    max_samples_per_class = data_cfg.get("max_samples_per_class")

    built: dict[str, Dataset] = {}
    folder_specs = [
        ("imagenet_val", "imagenet_val_root", "ImageNet", "id"),
        ("imagenet_a", "imagenet_a_root", "ImageNet-A", "ood"),
        ("imagenet_o", "imagenet_o_root", "ImageNet-O", "ood"),
    ]
    for dataset_name, root_key, family, role in folder_specs:
        root = _configured_root(data_cfg, root_key)
        if root is None:
            continue
        if dataset_name == "imagenet_val" and _looks_like_flat_image_dir(root):
            dataset = MetadataFlatImageDataset(
                root,
                dataset_name=dataset_name,
                ood_family=family,
                role=role,
                transform=transform,
            )
        else:
            dataset = MetadataImageFolder(
                root,
                dataset_name=dataset_name,
                ood_family=family,
                role=role,
                transform=transform,
            )
        dataset = _limit_dataset_per_class(dataset, max_samples_per_class)
        built[dataset_name] = _limit_dataset(dataset, max_samples)

    imagenet_c_root = _configured_root(data_cfg, "imagenet_c_root")
    if imagenet_c_root is not None:
        dataset = _build_imagenet_c(imagenet_c_root, transform)
        dataset = _limit_dataset_per_class(dataset, max_samples_per_class)
        built["imagenet_c"] = _limit_dataset(dataset, max_samples)

    cifar10_root = _configured_root(data_cfg, "cifar10_root")
    if cifar10_root is not None:
        raw = datasets.CIFAR10(
            root=cifar10_root,
            train=data_cfg.get("cifar10_split", "test") == "train",
            download=bool(data_cfg.get("cifar10_download", False)),
        )
        dataset = MetadataTorchvisionDataset(
            raw,
            dataset_name="cifar10",
            ood_family="CIFAR-10",
            role="ood",
            transform=transform,
        )
        dataset = _limit_dataset_per_class(dataset, max_samples_per_class)
        built["cifar10"] = _limit_dataset(dataset, max_samples)

    mnist_root = _configured_root(data_cfg, "mnist_root")
    if mnist_root is not None:
        raw = datasets.MNIST(
            root=mnist_root,
            train=data_cfg.get("mnist_split", "test") == "train",
            download=bool(data_cfg.get("mnist_download", False)),
        )
        dataset = MetadataTorchvisionDataset(
            raw,
            dataset_name="mnist",
            ood_family="MNIST",
            role="ood",
            transform=transform,
        )
        dataset = _limit_dataset_per_class(dataset, max_samples_per_class)
        built["mnist"] = _limit_dataset(dataset, max_samples)

    return built


def build_dataloaders(cfg: dict[str, Any]) -> dict[str, DataLoader]:
    """Build deterministic dataloaders for all configured DISC datasets."""
    data_cfg = dict(cfg["data"] if "data" in cfg else cfg)
    seed = int(data_cfg.get("seed", cfg.get("seed", 42) if "data" in cfg else 42))
    generator = torch.Generator()
    generator.manual_seed(seed)

    loaders: dict[str, DataLoader] = {}
    for name, dataset in build_datasets(cfg).items():
        loaders[name] = DataLoader(
            dataset,
            batch_size=int(data_cfg.get("batch_size", 32)),
            shuffle=False,
            num_workers=int(data_cfg.get("num_workers", 0)),
            pin_memory=torch.cuda.is_available(),
            generator=generator,
        )
    return loaders


def smoke_check_dataloaders(loaders: dict[str, DataLoader]) -> dict[str, tuple[int, int, int, int]]:
    """Verify each loader yields batched image tensors shaped ``[B, C, H, W]``."""
    shapes: dict[str, tuple[int, int, int, int]] = {}
    for name, loader in loaders.items():
        batch = next(iter(loader))
        images = batch["image"]
        if not isinstance(images, torch.Tensor) or images.ndim != 4:
            raise AssertionError(f"{name} did not return image tensor with shape [B, C, H, W].")
        shapes[name] = tuple(int(dim) for dim in images.shape)
    return shapes
