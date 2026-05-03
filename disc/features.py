"""DISC feature extraction over diffusion trajectories.

The main function, `compute_disc_features_for_batch`, turns a batch of images
into a dense feature matrix with stable feature names. It adds noise at a
selected high timestep, runs reverse diffusion, and records per-sample metrics
at configured timesteps.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .metrics import MetricComputer


@dataclass
class FeatureBatch:
    """Feature matrix and matching column names for one input batch."""

    X: np.ndarray
    feature_names: list[str]


def _feature_tensor_to_columns(name: str, value: torch.Tensor) -> tuple[list[str], np.ndarray]:
    """Convert a metric tensor shaped [B] or [B, d...] into named columns."""
    value = value.detach().cpu().float()
    if value.ndim == 1:
        return [name], value.numpy().reshape(-1, 1)
    flat = value.reshape(value.shape[0], -1)
    names = [f"{name}_{idx:03d}" for idx in range(flat.shape[1])]
    return names, flat.numpy()


def _flatten_feature_dict(values: dict[str, torch.Tensor]) -> FeatureBatch:
    names: list[str] = []
    columns: list[np.ndarray] = []
    for name, value in values.items():
        metric_names, metric_values = _feature_tensor_to_columns(name, value)
        names.extend(metric_names)
        columns.append(metric_values)
    if not columns:
        raise ValueError("No metrics were computed; check the feature configuration.")
    return FeatureBatch(X=np.hstack(columns).astype(np.float32), feature_names=names)


def _metric_reference_and_reconstruction(
    comparison: str,
    images: torch.Tensor,
    reconstruction: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Choose the metric inputs for the configured comparison convention."""
    if comparison == "orig_vs_recon":
        return reconstruction.x_noisy, reconstruction.x_recon, images
    if comparison == "noisy_vs_recon":
        return reconstruction.x_noisy, reconstruction.x_recon, None
    if comparison == "orig_vs_x0_hat":
        if reconstruction.x0_hat is None:
            raise ValueError("orig_vs_x0_hat requested, but the backend did not return x0_hat.")
        return reconstruction.x_noisy, reconstruction.x0_hat, images
    raise ValueError(f"Unsupported comparison mode: {comparison}")


class FeatureExtractor:
    """Extract one DISC feature vector per image for configured timesteps."""

    def __init__(self, config: dict[str, Any], device: torch.device | str):
        self.config = config
        self.feature_cfg = dict(config.get("features", config))
        self.device = torch.device(device)
        self.timesteps = sorted([int(t) for t in self.feature_cfg["timesteps"]], reverse=True)
        self.comparison = self.feature_cfg.get("comparison", "orig_vs_recon")
        self.metric_computer: MetricComputer | None = None

    def _get_metric_computer(self, reconstructor: Any) -> MetricComputer:
        """Build metrics lazily so optional LC can receive the reconstructor."""
        if self.metric_computer is None:
            metric_cfg = deepcopy(self.feature_cfg)
            lc_cfg = dict(metric_cfg.get("local_complexity", {}))
            if lc_cfg.get("enabled", False):
                lc_cfg.setdefault("reconstructor", reconstructor)
                metric_cfg["local_complexity"] = lc_cfg
            self.metric_computer = MetricComputer(metric_cfg, self.device)
        return self.metric_computer

    def _set_metric_timestep(self, timestep: int) -> None:
        if self.metric_computer is None:
            return
        local_complexity = getattr(self.metric_computer, "local_complexity", None)
        if local_complexity is not None:
            local_complexity.timestep = timestep

    @torch.no_grad()
    def compute_batch(self, images: torch.Tensor, reconstructor: Any) -> FeatureBatch:
        """Compute a feature matrix for one dataloader batch."""
        images = images.to(self.device)
        metric_computer = self._get_metric_computer(reconstructor)
        reconstructions = reconstructor.reconstruct(images, self.timesteps)
        values: dict[str, torch.Tensor] = {}
        for timestep in self.timesteps:
            self._set_metric_timestep(timestep)
            reconstruction = reconstructions[timestep]
            x_noisy, x_recon, x_clean = _metric_reference_and_reconstruction(
                self.comparison,
                images,
                reconstruction,
            )
            metrics = metric_computer.compute(x_noisy, x_recon, x_clean=x_clean)
            for metric_name, metric_value in metrics.items():
                values[f"t{timestep}/{metric_name}"] = metric_value
        return _flatten_feature_dict(values)


@torch.no_grad()
def compute_disc_features_for_batch(
    images: torch.Tensor,
    diffusion_backend: Any,
    feature_cfg: dict[str, Any],
    *,
    lpips_scorer: Any | None = None,
) -> FeatureBatch:
    """Compute DISC features for a batch of images.

    Args:
        images: Tensor shaped `(B, C, H, W)` on the diffusion device.
        diffusion_backend: Object implementing `reconstruct`, or
            `q_sample`/`p_sample` methods.
        feature_cfg: The `features` section of the YAML config.
        lpips_scorer: Optional LPIPS scorer when `lpips` is enabled.

    Returns:
        A `FeatureBatch` containing one row per input image.
    """
    config = {"features": feature_cfg}
    comparison = feature_cfg.get("comparison", "orig_vs_recon")

    if hasattr(diffusion_backend, "reconstruct"):
        extractor = FeatureExtractor(config, images.device)
        if lpips_scorer is not None:
            metric_computer = extractor._get_metric_computer(diffusion_backend)
            metric_computer.lpips_scorer = lpips_scorer
        return extractor.compute_batch(images, diffusion_backend)

    timesteps = sorted([int(t) for t in feature_cfg["timesteps"]], reverse=True)
    mode = feature_cfg.get("mode", "full_reverse")
    metric_computer = MetricComputer(feature_cfg, images.device)
    if lpips_scorer is not None:
        metric_computer.lpips_scorer = lpips_scorer
    values: dict[str, torch.Tensor] = {}
    batch_size = images.shape[0]

    if mode == "single_step":
        for timestep in timesteps:
            t = torch.full((batch_size,), timestep, dtype=torch.long, device=images.device)
            noisy = diffusion_backend.q_sample(images, t, noise=torch.randn_like(images))
            recon = diffusion_backend.p_sample(noisy, t)
            x_clean = images if comparison == "orig_vs_recon" else None
            metrics = metric_computer.compute(noisy, recon, x_clean=x_clean)
            values.update({f"t{timestep}/{name}": value for name, value in metrics.items()})
    elif mode == "full_reverse":
        record_set = set(timesteps)
        start_t = max(record_set)
        end_t = min(record_set)
        t = torch.full((batch_size,), start_t, dtype=torch.long, device=images.device)
        sample = diffusion_backend.q_sample(images, t, noise=torch.randn_like(images))
        for timestep in range(start_t, end_t - 1, -1):
            t = torch.full((batch_size,), timestep, dtype=torch.long, device=images.device)
            pre_step = sample
            sample = diffusion_backend.p_sample(sample, t)
            if timestep in record_set:
                x_clean = images if comparison == "orig_vs_recon" else None
                metrics = metric_computer.compute(pre_step, sample, x_clean=x_clean)
                values.update({f"t{timestep}/{name}": value for name, value in metrics.items()})
    else:
        raise ValueError(f"Unsupported feature extraction mode: {mode}")

    return _flatten_feature_dict(values)


def flatten_model(model: torch.nn.Module) -> tuple[list[str], list[torch.nn.Module]]:
    """Flatten a module tree into leaf module names and modules."""
    names: list[str] = []
    modules: list[torch.nn.Module] = []

    def visit(prefix: str, module: torch.nn.Module) -> None:
        children = list(module.named_children())
        if not children:
            names.append(prefix)
            modules.append(module)
            return
        for child_name, child in children:
            visit(f"{prefix}::{child_name}" if prefix else child_name, child)

    visit("", model)
    return names, modules


def make_orthogonal_hulls(
    x: torch.Tensor,
    *,
    hull_points: int,
    radius: float,
    seed: int | None,
) -> torch.Tensor:
    """Create symmetric orthogonal perturbation hulls around each sample."""
    if hull_points % 2 != 0:
        raise ValueError("hull_points must be even when the centroid is not included.")
    generator = torch.Generator(device=x.device)
    if seed is not None:
        generator.manual_seed(seed)
    half = hull_points // 2
    flat_dim = int(np.prod(x.shape[1:]))
    random_matrix = torch.randn(flat_dim, half, device=x.device, dtype=x.dtype, generator=generator)
    q, _ = torch.linalg.qr(random_matrix, mode="reduced")
    directions = q.T
    directions = directions / directions.norm(dim=1, keepdim=True).clamp_min(1e-12)
    directions = radius * directions
    directions = directions.unsqueeze(0).expand(x.shape[0], -1, -1)
    flat = x.flatten(1).unsqueeze(1)
    hull = torch.cat([flat + directions, flat - directions], dim=1)
    return hull.reshape(x.shape[0], hull_points, *x.shape[1:])


class LocalComplexityScorer:
    """Optional local-complexity scorer based on activation sign changes."""

    def __init__(self, model: torch.nn.Module, diffusion_backend: Any):
        self.model = model
        self.diffusion_backend = diffusion_backend
        self.activation_buffer: dict[str, torch.Tensor] = {}
        self.layer_names: list[str] = []
        self._register_silu_pre_hooks()

    def _register_silu_pre_hooks(self) -> None:
        names, modules = flatten_model(self.model)
        for idx, module in enumerate(modules[:-1]):
            if modules[idx + 1].__class__.__name__ != "SiLU":
                continue
            name = names[idx]
            self.layer_names.append(name)

            def hook(_: torch.nn.Module, __: Any, output: torch.Tensor, layer_name: str = name) -> None:
                self.activation_buffer[layer_name] = output.detach()

            module.register_forward_hook(hook)

    @staticmethod
    def _intersections(activation: torch.Tensor, batch_size: int) -> torch.Tensor:
        signs = torch.sign(activation).reshape(batch_size, activation.shape[0] // batch_size, -1)
        changed = (signs[:, 1:] != signs[:, :1]).any(dim=1)
        return changed.sum(dim=1).float()

    @torch.no_grad()
    def score(
        self,
        noisy_samples: torch.Tensor,
        timestep: int,
        *,
        hull_points: int = 32,
        radius: float = 0.1,
        batch_size: int = 1,
        seed: int | None = 42,
    ) -> torch.Tensor:
        """Return one mean local-complexity value per sample."""
        hulls = make_orthogonal_hulls(
            noisy_samples, hull_points=hull_points, radius=radius, seed=seed
        )
        scores = torch.zeros(noisy_samples.shape[0], device=noisy_samples.device)
        for start in range(0, hulls.shape[0], batch_size):
            batch = hulls[start : start + batch_size]
            current_bs = batch.shape[0]
            concat = batch.reshape(current_bs * hull_points, *noisy_samples.shape[1:])
            t = torch.full((concat.shape[0],), timestep, dtype=torch.long, device=concat.device)
            self.diffusion_backend.p_sample(concat, t)
            layer_scores = [
                self._intersections(self.activation_buffer[name], current_bs)
                for name in self.layer_names
            ]
            scores[start : start + current_bs] = torch.stack(layer_scores, dim=1).mean(dim=1)
        return scores
