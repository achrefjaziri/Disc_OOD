"""General utilities shared by feature extraction and evaluation."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def get_device(device_name: str | None = "auto") -> torch.device:
    """Resolve `auto`, `cpu`, or a PyTorch device string."""
    if device_name in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a `Path`."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: str | Path, payload: Any) -> None:
    """Write JSON with stable formatting."""
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def to_numpy_1d(values: torch.Tensor) -> np.ndarray:
    """Detach a tensor and return a flat float NumPy array."""
    return values.detach().cpu().float().numpy().reshape(-1)

