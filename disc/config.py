"""Configuration loading helpers for the minimal DISC pipeline.

The project is intentionally YAML-driven. This module keeps configuration
parsing small: it loads YAML, expands environment variables in string values,
and exposes a few helpers used by scripts.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import yaml


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config and expand environment variables in all strings."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg = _expand_env(cfg)
    cfg["_config_path"] = str(path)
    cfg["_repo_root"] = str(path.resolve().parents[1])
    return cfg


def get_section(cfg: Mapping[str, Any], key: str) -> dict[str, Any]:
    """Return a required mapping section from the loaded config."""
    value = cfg.get(key)
    if not isinstance(value, dict):
        raise KeyError(f"Config section '{key}' is missing or not a mapping.")
    return dict(value)


def resolve_path(path_value: str | Path, base_dir: str | Path | None = None) -> Path:
    """Resolve a possibly relative path against `base_dir` without hard-coding roots."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return Path(base_dir or ".").resolve() / path


def enabled_datasets(cfg: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Return enabled dataset entries keyed by configured dataset name."""
    data_cfg = get_section(cfg, "data")
    datasets = data_cfg.get("datasets")
    if isinstance(datasets, dict):
        return {
            name: dict(ds_cfg)
            for name, ds_cfg in datasets.items()
            if dict(ds_cfg).get("enabled", True)
        }
    roots = {
        "imagenet_val": data_cfg.get("imagenet_val_root"),
        "imagenet_a": data_cfg.get("imagenet_a_root"),
        "imagenet_o": data_cfg.get("imagenet_o_root"),
        "imagenet_c": data_cfg.get("imagenet_c_root"),
        "cifar10": data_cfg.get("cifar10_root"),
        "mnist": data_cfg.get("mnist_root"),
    }
    return {
        name: {"root": root}
        for name, root in roots.items()
        if root and "$" not in str(root)
    }
