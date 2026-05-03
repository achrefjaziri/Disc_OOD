#!/usr/bin/env python
"""Compute and save DISC feature files from the YAML configuration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from disc.config import load_config
from disc.data import build_dataloaders
from disc.diffusion import DiffusionReconstructor
from disc.features import FeatureExtractor
from disc.io import (
    feature_file_path,
    load_feature_file,
    merge_feature_shards,
    save_feature_shard,
    shard_dir,
    shard_file_path,
)
from disc.utils import ensure_dir, get_device, set_seed


DATASET_ALIASES = {
    "id": "imagenet_val",
    "imagenet": "imagenet_val",
    "imagenet_id": "imagenet_val",
    "imagenet_val": "imagenet_val",
    "imagenet-a": "imagenet_a",
    "imagenet_a": "imagenet_a",
    "imagenet-o": "imagenet_o",
    "imagenet_o": "imagenet_o",
    "imagenet-c": "imagenet_c",
    "imagenet_c": "imagenet_c",
    "cifar10": "cifar10",
    "cifar-10": "cifar10",
    "mnist": "mnist",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/imagenet_disc.yaml")
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Run a tiny in-memory feature extraction check without loading data or a checkpoint.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        help="Optional dataset names to compute, e.g. imagenet_id imagenet_a or cifar10,mnist.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Override data.max_samples_per_dataset for quick runs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute datasets even when final feature archives already exist.",
    )
    parser.add_argument(
        "--shard_size",
        type=int,
        default=1024,
        help="Approximate number of samples per partial shard archive.",
    )
    return parser.parse_args()


def _requested_dataset_names(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    tokens: list[str] = []
    for value in values:
        tokens.extend(part.strip() for part in value.split(",") if part.strip())
    if not tokens or any(token.lower() == "all" for token in tokens):
        return None
    normalized: set[str] = set()
    for token in tokens:
        key = token.lower()
        if key not in DATASET_ALIASES:
            known = ", ".join(sorted(DATASET_ALIASES))
            raise ValueError(f"Unknown dataset '{token}'. Known names: {known}")
        normalized.add(DATASET_ALIASES[key])
    return normalized


def _batch_size(batch: dict[str, Any]) -> int:
    return int(batch["image"].shape[0])


def _slice_batch(batch: dict[str, Any], start: int) -> dict[str, Any]:
    if start <= 0:
        return batch
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value[start:]
        elif isinstance(value, (list, tuple)):
            out[key] = list(value)[start:]
        else:
            out[key] = value
    return out


def _metadata_array(batch: dict[str, Any], key: str, batch_size: int, default: int = -1) -> np.ndarray:
    value = batch.get(key)
    if value is None:
        return np.full(batch_size, default, dtype=np.int64)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().reshape(-1)
    return np.asarray(value).reshape(-1)


def _metadata_strings(batch: dict[str, Any], key: str, batch_size: int) -> list[str]:
    value = batch.get(key)
    if value is None:
        return [""] * batch_size
    if isinstance(value, torch.Tensor):
        return [str(x) for x in value.detach().cpu().numpy().reshape(-1).tolist()]
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value]
    return [str(value)] * batch_size


def _clear_shards(root: Path) -> None:
    if not root.exists():
        return
    for path in sorted(root.glob("shard_*.npz")):
        path.unlink()


def _existing_shard_state(root: Path) -> tuple[list[Path], int, list[str] | None]:
    shard_paths = sorted(root.glob("shard_*.npz"))
    processed = 0
    feature_names: list[str] | None = None
    for path in shard_paths:
        item = load_feature_file(path)
        processed += int(item["features"].shape[0])
        if feature_names is None:
            feature_names = list(item["feature_names"])
        elif feature_names != list(item["feature_names"]):
            raise ValueError(f"Existing shard feature names differ under {root}. Use --overwrite.")
    return shard_paths, processed, feature_names


def _save_pending_shard(
    *,
    feature_dir: Path,
    dataset_name: str,
    shard_index: int,
    pending_features: list[np.ndarray],
    pending_ids: list[np.ndarray],
    pending_labels: list[np.ndarray],
    pending_paths: list[str],
    pending_sources: list[str],
    feature_names: list[str],
    timesteps: list[int],
    cfg: dict[str, Any],
    family: str,
    role: str,
) -> Path:
    features = np.vstack(pending_features).astype(np.float32)
    sample_ids = np.concatenate(pending_ids)
    labels = np.concatenate(pending_labels)
    path = shard_file_path(feature_dir, dataset_name, shard_index)
    save_feature_shard(
        path,
        features=features,
        feature_names=feature_names,
        dataset_name=dataset_name,
        family=family,
        role=role,
        sample_ids=sample_ids,
        labels=labels,
        paths=pending_paths,
        source_names=pending_sources,
        timesteps=timesteps,
        config=cfg,
        metadata={
            "config_path": cfg.get("_config_path", ""),
            "shard_index": shard_index,
            "n_samples": int(features.shape[0]),
        },
    )
    print(f"  wrote shard {shard_index:06d}: {features.shape} -> {path}")
    return path


def run_smoke_test(cfg: dict[str, Any], device: torch.device) -> None:
    """Exercise the reconstructor and feature code with a dummy backend."""
    image_size = int(cfg.get("data", {}).get("image_size", cfg.get("diffusion", {}).get("image_size", 64)))
    smoke_cfg = dict(cfg)
    smoke_cfg["diffusion"] = dict(cfg.get("diffusion", {}))
    smoke_cfg["diffusion"].update(
        {
            "backend": "dummy",
            "device": str(device),
            "reconstruction_mode": "single_step",
            "diffusion_steps": 16,
        }
    )
    smoke_features = dict(cfg.get("features", {}))
    smoke_features.update(
        {
            "mode": "single_step",
            "timesteps": [4, 1],
            "pixel_metrics": ["mse", "ssim"],
            "texture": {"enabled": False},
            "local_complexity": {"enabled": False},
        }
    )
    smoke_cfg["features"] = smoke_features
    reconstructor = DiffusionReconstructor(smoke_cfg, device)
    extractor = FeatureExtractor(smoke_cfg, device)
    images = torch.randn(2, 3, image_size, image_size, device=device)
    recon = reconstructor.reconstruct(images, smoke_features["timesteps"])
    batch = extractor.compute_batch(images, reconstructor)
    print(f"Smoke reconstruction timesteps: {list(recon.keys())}")
    print(f"Smoke feature matrix shape: {batch.X.shape}")
    print(f"Smoke feature names: {batch.feature_names}")
    print("Smoke test passed.")


def compute_dataset_features(
    *,
    dataset_name: str,
    loader: torch.utils.data.DataLoader,
    diffusion: DiffusionReconstructor,
    extractor: FeatureExtractor,
    cfg: dict[str, Any],
    feature_dir: Path,
    shard_size: int,
    overwrite: bool,
) -> None:
    """Compute, shard, and merge features for one configured dataset."""
    out_path = feature_file_path(feature_dir, dataset_name)
    if out_path.exists() and not overwrite:
        print(f"Skipping {dataset_name}: final feature file already exists at {out_path}")
        return

    dataset_obj = loader.dataset
    family = str(getattr(dataset_obj, "ood_family", dataset_name))
    role = str(getattr(dataset_obj, "role", "unknown"))
    timesteps = list(extractor.timesteps)

    root = ensure_dir(shard_dir(feature_dir, dataset_name))
    if overwrite:
        _clear_shards(root)
    existing_shards, processed, feature_names = _existing_shard_state(root)
    next_shard_index = len(existing_shards)
    if existing_shards and not overwrite:
        print(f"Resuming {dataset_name}: found {len(existing_shards)} shards covering {processed} samples.")

    pending_features: list[np.ndarray] = []
    pending_ids: list[np.ndarray] = []
    pending_labels: list[np.ndarray] = []
    pending_paths: list[str] = []
    pending_sources: list[str] = []
    pending_count = 0
    seen = 0

    progress = tqdm(loader, desc=f"features:{dataset_name}", unit="batch")
    for batch in progress:
        original_batch_size = _batch_size(batch)
        if seen + original_batch_size <= processed:
            seen += original_batch_size
            continue
        if seen < processed:
            batch = _slice_batch(batch, processed - seen)
        seen += original_batch_size

        images = batch["image"].to(diffusion.device, non_blocking=True)
        feat_batch = extractor.compute_batch(images, diffusion)
        if feature_names is None:
            feature_names = feat_batch.feature_names
        elif feature_names != feat_batch.feature_names:
            raise RuntimeError("Feature names changed between batches or shards.")

        current_bs = feat_batch.X.shape[0]
        pending_features.append(feat_batch.X)
        pending_ids.append(_metadata_array(batch, "original_index", current_bs))
        pending_labels.append(_metadata_array(batch, "label", current_bs))
        pending_paths.extend(_metadata_strings(batch, "path", current_bs))
        pending_sources.extend(_metadata_strings(batch, "source_name", current_bs))
        pending_count += current_bs
        progress.set_postfix(samples=processed + pending_count)

        if pending_count >= shard_size:
            _save_pending_shard(
                feature_dir=feature_dir,
                dataset_name=dataset_name,
                shard_index=next_shard_index,
                pending_features=pending_features,
                pending_ids=pending_ids,
                pending_labels=pending_labels,
                pending_paths=pending_paths,
                pending_sources=pending_sources,
                feature_names=feature_names,
                timesteps=timesteps,
                cfg=cfg,
                family=family,
                role=role,
            )
            next_shard_index += 1
            processed += pending_count
            pending_features = []
            pending_ids = []
            pending_labels = []
            pending_paths = []
            pending_sources = []
            pending_count = 0

    if pending_count:
        _save_pending_shard(
            feature_dir=feature_dir,
            dataset_name=dataset_name,
            shard_index=next_shard_index,
            pending_features=pending_features,
            pending_ids=pending_ids,
            pending_labels=pending_labels,
            pending_paths=pending_paths,
            pending_sources=pending_sources,
            feature_names=feature_names or [],
            timesteps=timesteps,
            cfg=cfg,
            family=family,
            role=role,
        )

    shard_paths = sorted(root.glob("shard_*.npz"))
    if not shard_paths:
        raise RuntimeError(f"No feature shards were produced for {dataset_name}.")
    merged = merge_feature_shards(
        shard_paths,
        out_path,
        config=cfg,
        metadata={
            "config_path": cfg.get("_config_path", ""),
            "n_shards": len(shard_paths),
            "source": "scripts/compute_features.py",
        },
    )
    print(f"Saved {dataset_name}: {merged['features'].shape} -> {out_path}")


def main() -> None:
    args = parse_args()
    if args.shard_size <= 0:
        raise ValueError("--shard_size must be a positive integer.")

    cfg = load_config(args.config)
    if args.max_samples is not None:
        cfg.setdefault("data", {})["max_samples_per_dataset"] = int(args.max_samples)
    set_seed(int(cfg.get("seed", 42)))
    device = get_device(cfg.get("device", "auto"))
    if args.smoke_test:
        run_smoke_test(cfg, device)
        return

    feature_dir = ensure_dir(cfg["paths"]["feature_dir"])
    requested = _requested_dataset_names(args.datasets)
    loaders = build_dataloaders(cfg)
    if requested is not None:
        missing = sorted(name for name in requested if name not in loaders)
        if missing:
            available = ", ".join(sorted(loaders)) or "none"
            raise FileNotFoundError(f"Requested datasets are not configured: {missing}. Available: {available}")
        loaders = {name: loader for name, loader in loaders.items() if name in requested}
    if not loaders:
        raise FileNotFoundError("No datasets were configured. Check data roots in the YAML config.")

    work = [
        name
        for name in loaders
        if args.overwrite or not feature_file_path(feature_dir, name).exists()
    ]
    if not work:
        print("All requested feature files already exist. Use --overwrite to recompute.")
        return

    diffusion = DiffusionReconstructor(cfg, device)
    extractor = FeatureExtractor(cfg, device)

    for dataset_name, loader in loaders.items():
        compute_dataset_features(
            dataset_name=dataset_name,
            loader=loader,
            diffusion=diffusion,
            extractor=extractor,
            cfg=cfg,
            feature_dir=feature_dir,
            shard_size=int(args.shard_size),
            overwrite=bool(args.overwrite),
        )


if __name__ == "__main__":
    with torch.no_grad():
        main()
