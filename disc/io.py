"""Input/output helpers for DISC feature archives.

Feature extraction writes compressed NumPy archives. The canonical keys are
``features``, ``labels``, ``sample_ids``, ``feature_names``, ``timesteps``,
``family``, ``config_hash``, and ``config_json``. Aliases ``X`` and ``targets``
are also stored for downstream analysis helpers.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .utils import ensure_dir


CANONICAL_DATASET_STEMS = {
    "imagenet_val": "imagenet_id",
    "imagenet_id": "imagenet_id",
    "imagenet_a": "imagenet_a",
    "imagenet_o": "imagenet_o",
    "imagenet_c": "imagenet_c",
    "cifar10": "cifar10",
    "mnist": "mnist",
}


def _json_default(value: Any) -> Any:
    """Convert common scientific/Python objects into JSON-compatible values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return str(value)


def dumps_json(payload: Any) -> str:
    """Return a stable JSON representation for configs and metadata."""
    return json.dumps(payload, sort_keys=True, default=_json_default)


def config_hash(config: dict[str, Any] | None) -> str:
    """Short stable hash of a config dictionary."""
    return hashlib.sha256(dumps_json(config or {}).encode("utf-8")).hexdigest()[:16]


def canonical_dataset_stem(dataset_name: str) -> str:
    """Map loader names to reviewer-facing feature filename stems."""
    return CANONICAL_DATASET_STEMS.get(dataset_name, dataset_name)


def feature_file_path(feature_dir: str | Path, dataset_name: str) -> Path:
    """Return the canonical final archive path for a dataset."""
    stem = canonical_dataset_stem(dataset_name)
    return Path(feature_dir) / f"{stem}_disc_features.npz"


def shard_dir(feature_dir: str | Path, dataset_name: str) -> Path:
    """Return the private shard directory for a dataset."""
    return Path(feature_dir) / "_shards" / canonical_dataset_stem(dataset_name)


def shard_file_path(feature_dir: str | Path, dataset_name: str, shard_index: int) -> Path:
    """Return the path for one partial feature shard."""
    return shard_dir(feature_dir, dataset_name) / f"shard_{shard_index:06d}.npz"


def _as_object_array(values: list[str] | tuple[str, ...] | np.ndarray | None) -> np.ndarray:
    if values is None:
        return np.asarray([], dtype=object)
    return np.asarray(list(values), dtype=object)


def _as_1d_array(values: np.ndarray | list[Any] | tuple[Any, ...] | None, *, dtype: Any | None = None) -> np.ndarray:
    if values is None:
        return np.asarray([], dtype=dtype or np.int64)
    arr = np.asarray(values, dtype=dtype)
    return arr.reshape(-1)


def save_feature_file(
    path: str | Path,
    *,
    features: np.ndarray | None = None,
    X: np.ndarray | None = None,
    feature_names: list[str],
    dataset_name: str,
    family: str,
    sample_ids: np.ndarray | list[Any],
    timesteps: list[int] | np.ndarray | None = None,
    config: dict[str, Any] | None = None,
    labels: np.ndarray | list[Any] | None = None,
    targets: np.ndarray | list[Any] | None = None,
    role: str = "unknown",
    paths: list[str] | np.ndarray | None = None,
    source_names: list[str] | np.ndarray | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save one dataset's DISC feature matrix and metadata."""
    path = Path(path)
    ensure_dir(path.parent)
    matrix = features if features is not None else X
    if matrix is None:
        raise ValueError("save_feature_file requires `features` or `X`.")
    matrix = np.asarray(matrix, dtype=np.float32)
    label_array = _as_1d_array(labels if labels is not None else targets, dtype=np.int64)
    sample_id_array = _as_1d_array(sample_ids)
    timestep_array = _as_1d_array(timesteps, dtype=np.int64)
    config_json = dumps_json(config or {})
    metadata_json = dumps_json(metadata or {})
    np.savez_compressed(
        path,
        features=matrix,
        X=matrix,
        labels=label_array,
        targets=label_array,
        feature_names=np.asarray(feature_names, dtype=object),
        dataset_name=np.asarray(dataset_name, dtype=object),
        family=np.asarray(family, dtype=object),
        role=np.asarray(role, dtype=object),
        sample_ids=sample_id_array,
        paths=_as_object_array(paths),
        source_names=_as_object_array(source_names),
        timesteps=timestep_array,
        config_hash=np.asarray(config_hash(config), dtype=object),
        config_json=np.asarray(config_json, dtype=object),
        metadata=np.asarray(metadata_json, dtype=object),
    )


def save_feature_shard(
    path: str | Path,
    *,
    features: np.ndarray,
    feature_names: list[str],
    dataset_name: str,
    family: str,
    sample_ids: np.ndarray | list[Any],
    timesteps: list[int] | np.ndarray,
    config: dict[str, Any],
    labels: np.ndarray | list[Any] | None = None,
    role: str = "unknown",
    paths: list[str] | np.ndarray | None = None,
    source_names: list[str] | np.ndarray | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save one partial feature shard with the same schema as final archives."""
    save_feature_file(
        path,
        features=features,
        feature_names=feature_names,
        dataset_name=dataset_name,
        family=family,
        sample_ids=sample_ids,
        labels=labels,
        role=role,
        paths=paths,
        source_names=source_names,
        timesteps=timesteps,
        config=config,
        metadata=metadata,
    )


def _scalar_string(data: np.lib.npyio.NpzFile, key: str, default: str = "") -> str:
    if key not in data:
        return default
    value = data[key]
    if value.shape == ():
        return str(value.item())
    return str(value.reshape(-1)[0])


def _object_list(data: np.lib.npyio.NpzFile, key: str) -> list[str]:
    if key not in data:
        return []
    return [str(x) for x in data[key].tolist()]


def _json_field(data: np.lib.npyio.NpzFile, key: str) -> Any:
    if key not in data:
        return {}
    raw = _scalar_string(data, key, "{}")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def load_feature_file(path: str | Path) -> dict[str, Any]:
    """Load a feature archive saved by `save_feature_file`."""
    data = np.load(path, allow_pickle=True)
    features = data["features"] if "features" in data else data["X"]
    labels = (
        data["labels"]
        if "labels" in data
        else data["targets"]
        if "targets" in data
        else np.asarray([], dtype=np.int64)
    )
    return {
        "features": features,
        "X": features,
        "labels": labels,
        "targets": labels,
        "feature_names": _object_list(data, "feature_names"),
        "dataset_name": _scalar_string(data, "dataset_name"),
        "family": _scalar_string(data, "family"),
        "role": _scalar_string(data, "role", "unknown"),
        "sample_ids": data["sample_ids"] if "sample_ids" in data else np.arange(features.shape[0]),
        "paths": _object_list(data, "paths"),
        "source_names": _object_list(data, "source_names"),
        "timesteps": data["timesteps"] if "timesteps" in data else np.asarray([], dtype=np.int64),
        "config_hash": _scalar_string(data, "config_hash"),
        "config_json": _json_field(data, "config_json"),
        "metadata": _json_field(data, "metadata"),
    }


def merge_feature_shards(
    shard_paths: list[str | Path],
    output_path: str | Path,
    *,
    config: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge shard archives into a final dataset archive.

    Feature names and timesteps must match across shards. The returned mapping is
    the final archive loaded through `load_feature_file`.
    """
    if not shard_paths:
        raise ValueError("No feature shards were provided for merging.")

    loaded = [load_feature_file(path) for path in sorted(Path(p) for p in shard_paths)]
    first = loaded[0]
    feature_names = list(first["feature_names"])
    timesteps = np.asarray(first["timesteps"], dtype=np.int64)
    dataset_name = str(first["dataset_name"])
    family = str(first["family"])
    role = str(first["role"])

    for item in loaded[1:]:
        if list(item["feature_names"]) != feature_names:
            raise ValueError("Feature names differ between shards; refusing to merge.")
        if not np.array_equal(np.asarray(item["timesteps"], dtype=np.int64), timesteps):
            raise ValueError("Timesteps differ between shards; refusing to merge.")
        if str(item["dataset_name"]) != dataset_name:
            raise ValueError("Dataset names differ between shards; refusing to merge.")

    features = np.vstack([item["features"] for item in loaded]).astype(np.float32)
    labels = np.concatenate([np.asarray(item["labels"]).reshape(-1) for item in loaded])
    sample_ids = np.concatenate([np.asarray(item["sample_ids"]).reshape(-1) for item in loaded])
    paths: list[str] = []
    source_names: list[str] = []
    for item in loaded:
        paths.extend(item.get("paths", []))
        source_names.extend(item.get("source_names", []))

    save_feature_file(
        output_path,
        features=features,
        feature_names=feature_names,
        dataset_name=dataset_name,
        family=family,
        role=role,
        sample_ids=sample_ids,
        labels=labels,
        paths=paths,
        source_names=source_names,
        timesteps=timesteps,
        config=config,
        metadata=metadata or first.get("metadata", {}),
    )
    return load_feature_file(output_path)


def load_feature_dir(feature_dir: str | Path) -> dict[str, dict[str, Any]]:
    """Load every `.npz` feature file in a directory keyed by dataset name."""
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(Path(feature_dir).glob("*_disc_features.npz")):
        item = load_feature_file(path)
        out[item["dataset_name"]] = item
    for path in sorted(Path(feature_dir).glob("*.npz")):
        if path.name.endswith("_disc_features.npz"):
            continue
        item = load_feature_file(path)
        out[item["dataset_name"]] = item
    return out
