"""Per-image comparison metrics used in DISC feature vectors."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Any

import torch
import torch.nn.functional as F


def to_01(x: torch.Tensor) -> torch.Tensor:
    """Map tensors that are typically in [-1, 1] into [0, 1] and clamp."""
    return (x * 0.5 + 0.5).clamp(0.0, 1.0)


def rgb_to_gray_01(x: torch.Tensor) -> torch.Tensor:
    """Convert batched RGB tensors to grayscale in [0, 1]."""
    x01 = to_01(x)
    if x01.shape[1] == 1:
        return x01
    weights = torch.tensor([0.299, 0.587, 0.114], device=x.device, dtype=x.dtype)
    return (x01 * weights.view(1, 3, 1, 1)).sum(dim=1, keepdim=True)


@torch.no_grad()
def mse_score_imagewise(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Return one MSE value per image."""
    return ((x - y) ** 2).mean(dim=(1, 2, 3))


@torch.no_grad()
def ssim_score_imagewise(x: torch.Tensor, y: torch.Tensor, window_size: int = 7) -> torch.Tensor:
    """Return one SSIM value per image.

    Kornia is used when available. A compact global SSIM fallback keeps unit
    tests and CPU-only environments usable.
    """
    x01 = to_01(x)
    y01 = to_01(y)
    try:
        import kornia.metrics as km

        ssim_map = km.ssim(x01, y01, window_size=window_size)
        return ssim_map.mean(dim=(1, 2, 3))
    except Exception:
        dims = (1, 2, 3)
        c1 = 0.01**2
        c2 = 0.03**2
        mu_x = x01.mean(dim=dims)
        mu_y = y01.mean(dim=dims)
        var_x = ((x01 - mu_x.view(-1, 1, 1, 1)) ** 2).mean(dim=dims)
        var_y = ((y01 - mu_y.view(-1, 1, 1, 1)) ** 2).mean(dim=dims)
        cov = (
            (x01 - mu_x.view(-1, 1, 1, 1))
            * (y01 - mu_y.view(-1, 1, 1, 1))
        ).mean(dim=dims)
        return ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / (
            (mu_x.square() + mu_y.square() + c1) * (var_x + var_y + c2)
        )


class LPIPSScorer:
    """Thin lazy wrapper around the `lpips` package."""

    def __init__(self, device: torch.device, net: str = "alex"):
        try:
            from lpips import LPIPS
        except ImportError as exc:
            raise ImportError("LPIPS requested, but the lpips package is not installed.") from exc
        self.model = LPIPS(net=net).to(device).eval()

    @torch.no_grad()
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.model(x, y).view(x.shape[0])


def selected_pixel_metrics(
    x: torch.Tensor,
    y: torch.Tensor,
    names: Iterable[str],
    lpips_scorer: LPIPSScorer | None = None,
) -> dict[str, torch.Tensor]:
    """Compute selected per-image pixel/perceptual metrics."""
    out: dict[str, torch.Tensor] = {}
    for name in names:
        if name == "mse":
            out["mse"] = mse_score_imagewise(x, y)
        elif name == "ssim":
            out["ssim"] = ssim_score_imagewise(x, y)
        elif name == "lpips":
            if lpips_scorer is None:
                raise ValueError("LPIPS was requested but no LPIPSScorer was supplied.")
            out["lpips"] = lpips_scorer(x, y)
        else:
            raise ValueError(f"Unknown pixel metric: {name}")
    return out


def _histogram_by_batch(
    values: torch.Tensor,
    bins: int,
    vmin: float,
    vmax: float,
    eps: float = 1e-10,
) -> torch.Tensor:
    if not math.isfinite(vmin) or not math.isfinite(vmax):
        raise ValueError("Histogram range contains non-finite values.")
    if vmin == vmax:
        vmin -= 0.5
        vmax += 0.5
    hist = torch.stack(
        [torch.histc(values[i], bins=bins, min=vmin, max=vmax) for i in range(values.shape[0])]
    )
    hist = hist + eps
    return hist / hist.sum(dim=1, keepdim=True)


@torch.no_grad()
def texture_metrics(
    orig_gray: torch.Tensor,
    other_gray: torch.Tensor,
    *,
    nbins: int = 256,
    include_lbp: bool = True,
    include_hog: bool = True,
    include_intensity: bool = True,
    include_dtcwt: bool = True,
) -> dict[str, torch.Tensor]:
    """Compute batched texture and frequency divergences on grayscale images."""
    device = orig_gray.device
    batch_size, _, height, width = orig_gray.shape
    eps = 1e-10
    out: dict[str, torch.Tensor] = {}

    if include_lbp:
        lbp_bins = 256
        p_o = F.pad(orig_gray, (1, 1, 1, 1), mode="replicate")
        p_n = F.pad(other_gray, (1, 1, 1, 1), mode="replicate")
        c_o = p_o[:, 0, 1:-1, 1:-1]
        c_n = p_n[:, 0, 1:-1, 1:-1]
        offsets = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)]
        neigh_o = torch.stack([p_o[:, 0, dy : dy + height, dx : dx + width] for dy, dx in offsets], dim=1)
        neigh_n = torch.stack([p_n[:, 0, dy : dy + height, dx : dx + width] for dy, dx in offsets], dim=1)
        weights = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], device=device, dtype=torch.int64)
        codes_o = ((neigh_o >= c_o.unsqueeze(1)).long() * weights.view(1, 8, 1, 1)).sum(dim=1)
        codes_n = ((neigh_n >= c_n.unsqueeze(1)).long() * weights.view(1, 8, 1, 1)).sum(dim=1)
        offs = (torch.arange(batch_size, device=device) * lbp_bins).unsqueeze(1)
        flat_o = torch.bincount((codes_o.view(batch_size, -1) + offs).reshape(-1), minlength=batch_size * lbp_bins)
        flat_n = torch.bincount((codes_n.view(batch_size, -1) + offs).reshape(-1), minlength=batch_size * lbp_bins)
        h1 = flat_o.float().view(batch_size, lbp_bins) + eps
        h2 = flat_n.float().view(batch_size, lbp_bins) + eps
        h1 = h1 / h1.sum(dim=1, keepdim=True)
        h2 = h2 / h2.sum(dim=1, keepdim=True)
        out["lbp_TVD"] = 0.5 * (h1 - h2).abs().sum(dim=1)
        h1c = h1.clamp_min(eps)
        h2c = h2.clamp_min(eps)
        m = 0.5 * (h1c + h2c)
        out["lbp_KL"] = (h1c * (h1c / h2c).log()).sum(dim=1)
        out["lbp_JS"] = 0.5 * (
            (h1c * (h1c / m).log()).sum(dim=1) + (h2c * (h2c / m).log()).sum(dim=1)
        )
        out["lbp_Wasserstein"] = (h1.cumsum(dim=1) - h2.cumsum(dim=1)).abs().sum(dim=1)

    if include_hog:
        go_y, go_x = torch.gradient(orig_gray.squeeze(1), dim=(1, 2))
        gn_y, gn_x = torch.gradient(other_gray.squeeze(1), dim=(1, 2))
        mag_o = torch.sqrt(go_x.square() + go_y.square()).view(batch_size, -1)
        mag_n = torch.sqrt(gn_x.square() + gn_y.square()).view(batch_size, -1)
        ang_o = (torch.atan2(go_y, go_x) % math.pi).view(batch_size, -1)
        ang_n = (torch.atan2(gn_y, gn_x) % math.pi).view(batch_size, -1)
        bins = 9
        ids_o = (ang_o / (math.pi / bins)).floor().long().clamp(0, bins - 1)
        ids_n = (ang_n / (math.pi / bins)).floor().long().clamp(0, bins - 1)
        offs = (torch.arange(batch_size, device=device) * bins).unsqueeze(1)
        hist_o = torch.bincount(
            (ids_o + offs).reshape(-1), weights=mag_o.reshape(-1), minlength=batch_size * bins
        ).view(batch_size, bins)
        hist_n = torch.bincount(
            (ids_n + offs).reshape(-1), weights=mag_n.reshape(-1), minlength=batch_size * bins
        ).view(batch_size, bins)
        h1 = hist_o + eps
        h2 = hist_n + eps
        h1 = h1 / h1.sum(dim=1, keepdim=True)
        h2 = h2 / h2.sum(dim=1, keepdim=True)
        h1c = h1.clamp_min(eps)
        h2c = h2.clamp_min(eps)
        m = 0.5 * (h1c + h2c)
        out["hog_TVD"] = 0.5 * (h1 - h2).abs().sum(dim=1)
        out["hog_KL"] = (h1c * (h1c / h2c).log()).sum(dim=1)
        out["hog_JS"] = 0.5 * (
            (h1c * (h1c / m).log()).sum(dim=1) + (h2c * (h2c / m).log()).sum(dim=1)
        )
        out["hog_Wasserstein"] = (h1.cumsum(dim=1) - h2.cumsum(dim=1)).abs().sum(dim=1)

    if include_intensity:
        out["intensity_Euclidean"] = torch.norm(orig_gray.flatten(1) - other_gray.flatten(1), dim=1)
        out["intensity_Cosine"] = 1 - F.cosine_similarity(orig_gray.flatten(1), other_gray.flatten(1), dim=1)

    if include_dtcwt:
        try:
            from pytorch_wavelets import DTCWTForward
        except ImportError as exc:
            raise ImportError("DTCWT metrics requested, but pytorch-wavelets is not installed.") from exc
        xfm = DTCWTForward(J=3, biort="near_sym_b", qshift="qshift_b").to(device)
        yl_o, yh_o = xfm(orig_gray)
        yl_n, yh_n = xfm(other_gray)
        flat_lo = yl_o.flatten(1)
        flat_ln = yl_n.flatten(1)
        out["dtcwt_LL_MAD"] = (flat_lo - flat_ln).abs().mean(dim=1)
        vmin = min(flat_lo.min().item(), flat_ln.min().item())
        vmax = max(flat_lo.max().item(), flat_ln.max().item())
        h1 = _histogram_by_batch(flat_lo, nbins, vmin, vmax, eps).clamp_min(eps)
        h2 = _histogram_by_batch(flat_ln, nbins, vmin, vmax, eps).clamp_min(eps)
        out["dtcwt_LL_KL"] = (h1 * (h1 / h2).log()).sum(dim=1)
        for level, (ho, hn) in enumerate(zip(yh_o, yh_n), start=1):
            flat_ho = ho.flatten(1)
            flat_hn = hn.flatten(1)
            out[f"dtcwt_level{level}_MAD"] = (flat_ho - flat_hn).abs().mean(dim=1)
            vmin = min(flat_ho.min().item(), flat_hn.min().item())
            vmax = max(flat_ho.max().item(), flat_hn.max().item())
            h1 = _histogram_by_batch(flat_ho, nbins, vmin, vmax, eps).clamp_min(eps)
            h2 = _histogram_by_batch(flat_hn, nbins, vmin, vmax, eps).clamp_min(eps)
            out[f"dtcwt_level{level}_KL"] = (h1 * (h1 / h2).log()).sum(dim=1)

    return out


def _config_section(config: dict[str, Any]) -> dict[str, Any]:
    """Accept full repo config, `features`, or a dedicated `metrics` section."""
    if "metrics" in config and isinstance(config["metrics"], dict):
        return dict(config["metrics"])
    if "features" in config and isinstance(config["features"], dict):
        return dict(config["features"])
    return dict(config)


def _flatten_model(model: torch.nn.Module) -> tuple[list[str], list[torch.nn.Module]]:
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


def _orthogonal_hulls(
    x: torch.Tensor,
    *,
    hull_points: int,
    radius: float,
    seed: int | None,
) -> torch.Tensor:
    if hull_points < 2 or hull_points % 2 != 0:
        raise ValueError("local_complexity.hull_points must be an even integer >= 2.")
    generator = torch.Generator(device=x.device)
    if seed is not None:
        generator.manual_seed(seed)
    half = hull_points // 2
    flat_dim = int(torch.tensor(x.shape[1:]).prod().item())
    random_matrix = torch.randn(flat_dim, half, device=x.device, dtype=x.dtype, generator=generator)
    q, _ = torch.linalg.qr(random_matrix, mode="reduced")
    directions = radius * F.normalize(q.T, dim=1)
    flat = x.flatten(1).unsqueeze(1)
    directions = directions.unsqueeze(0).expand(x.shape[0], -1, -1)
    hull = torch.cat([flat + directions, flat - directions], dim=1)
    return hull.reshape(x.shape[0], hull_points, *x.shape[1:])


class LocalComplexityMetric:
    """Activation sign-change local complexity metric.

    LC samples an orthogonal hull around a noisy image, runs the diffusion model
    at the corresponding timestep, and counts whether pre-SiLU layer activation
    signs changed within the hull. Because LC depends on the diffusion model
    internals, this metric requires a runtime model/reconstructor or a callable
    forward function in config.
    """

    def __init__(self, config: dict[str, Any], device: torch.device):
        self.config = dict(config)
        self.device = device
        self.hull_points = int(self.config.get("hull_points", self.config.get("n", 32)))
        self.radius = float(self.config.get("radius", self.config.get("r", 0.1)))
        self.batch_size = int(self.config.get("batch_size", 1))
        self.seed = self.config.get("seed", 42)
        self.timestep = self.config.get("timestep")
        self.forward_fn: Callable[[torch.Tensor, int | None], Any] | None = self.config.get("forward_fn")
        self.reconstructor = self.config.get("reconstructor") or self.config.get("diffusion_reconstructor")
        self.model = self.config.get("model") or getattr(self.reconstructor, "model", None)
        if self.forward_fn is None and self.model is None and self.reconstructor is None:
            raise ValueError(
                "local_complexity is enabled, but no model, reconstructor, or forward_fn "
                "was supplied. LC is optional; disable it in config for feature-only runs."
            )
        if self.timestep is None and self.forward_fn is None:
            raise ValueError(
                "local_complexity requires a timestep when using a diffusion model/reconstructor."
            )
        self.activation_buffer: dict[str, torch.Tensor] = {}
        self.layer_names: list[str] = []
        if self.model is not None:
            self._register_preactivation_hooks(self.model)

    def _register_preactivation_hooks(self, model: torch.nn.Module) -> None:
        names, modules = _flatten_model(model)
        for idx, module in enumerate(modules[:-1]):
            if modules[idx + 1].__class__.__name__ != "SiLU":
                continue
            layer_name = names[idx]
            self.layer_names.append(layer_name)

            def hook(_: torch.nn.Module, __: Any, output: torch.Tensor, name: str = layer_name) -> None:
                self.activation_buffer[name] = output.detach()

            module.register_forward_hook(hook)
        if not self.layer_names:
            raise ValueError("local_complexity could not find pre-SiLU layers to hook.")

    @staticmethod
    def _intersections(activation: torch.Tensor, batch_size: int) -> torch.Tensor:
        signs = torch.sign(activation).reshape(batch_size, activation.shape[0] // batch_size, -1)
        changed = (signs[:, 1:] != signs[:, :1]).any(dim=1)
        return changed.sum(dim=1).float()

    def _run_forward(self, hull_batch: torch.Tensor) -> None:
        concat = hull_batch.reshape(hull_batch.shape[0] * hull_batch.shape[1], *hull_batch.shape[2:])
        timestep = None if self.timestep is None else int(self.timestep)
        if self.forward_fn is not None:
            self.forward_fn(concat, timestep)
            return
        t = torch.full((concat.shape[0],), timestep, dtype=torch.long, device=concat.device)
        if self.reconstructor is not None and hasattr(self.reconstructor, "_p_sample_output"):
            self.reconstructor._p_sample_output(concat, t)
            return
        if self.model is None:
            raise ValueError("local_complexity has no model to run.")
        self.model(concat, t)

    @torch.no_grad()
    def __call__(self, x_noisy: torch.Tensor) -> torch.Tensor:
        hulls = _orthogonal_hulls(
            x_noisy,
            hull_points=self.hull_points,
            radius=self.radius,
            seed=None if self.seed is None else int(self.seed),
        )
        scores = torch.zeros(x_noisy.shape[0], device=x_noisy.device)
        for start in range(0, hulls.shape[0], self.batch_size):
            hull_batch = hulls[start : start + self.batch_size]
            current_bs = hull_batch.shape[0]
            self.activation_buffer.clear()
            self._run_forward(hull_batch)
            if not self.layer_names:
                raise ValueError("local_complexity has no hooked layer activations.")
            layer_values = [
                self._intersections(self.activation_buffer[name], current_bs)
                for name in self.layer_names
            ]
            scores[start : start + current_bs] = torch.stack(layer_values, dim=1).mean(dim=1)
        return scores


class MetricComputer:
    """Compute DISC metrics for one batch without reducing across samples.

    Args:
        config: Full config, a `features` section, or a dedicated `metrics`
            section. The default ImageNet DISC metrics are MSE/SSIM/LPIPS with
            Alex LPIPS and optional texture/frequency descriptors.
        device: Device used for LPIPS, wavelets, and optional LC.
    """

    def __init__(self, config: dict[str, Any], device: torch.device | str):
        self.config = _config_section(config)
        self.device = torch.device(device)
        self.pixel_metrics = list(
            self.config.get("pixel_metrics", self.config.get("metrics", ["mse", "ssim"]))
        )
        self.texture_cfg = dict(self.config.get("texture", {}))
        self.texture_enabled = bool(self.texture_cfg.get("enabled", False))
        self.texture_optional = bool(self.texture_cfg.get("optional", False))
        self.local_complexity_cfg = dict(self.config.get("local_complexity", {}))
        self.local_complexity_enabled = bool(self.local_complexity_cfg.get("enabled", False))
        self.lpips_scorer: LPIPSScorer | None = None
        self.local_complexity: LocalComplexityMetric | None = None

        if "lpips" in self.pixel_metrics:
            lpips_cfg = dict(self.config.get("lpips", {}))
            backbone = lpips_cfg.get("backbone", lpips_cfg.get("net", "alex"))
            optional = bool(lpips_cfg.get("optional", self.config.get("lpips_optional", False)))
            try:
                self.lpips_scorer = LPIPSScorer(self.device, net=backbone)
            except ImportError:
                if optional:
                    self.pixel_metrics = [m for m in self.pixel_metrics if m != "lpips"]
                else:
                    raise

        if self.local_complexity_enabled:
            self.local_complexity = LocalComplexityMetric(self.local_complexity_cfg, self.device)

    @staticmethod
    def _validate_outputs(outputs: dict[str, torch.Tensor], batch_size: int) -> dict[str, torch.Tensor]:
        stable: dict[str, torch.Tensor] = {}
        for name, value in outputs.items():
            if value.shape[0] != batch_size:
                raise ValueError(
                    f"Metric '{name}' returned shape {tuple(value.shape)} for batch size {batch_size}."
                )
            if not torch.isfinite(value).all():
                raise FloatingPointError(f"Metric '{name}' produced NaN or Inf values.")
            stable[name] = value
        return stable

    @torch.no_grad()
    def compute(
        self,
        x_noisy: torch.Tensor,
        x_recon: torch.Tensor,
        x_clean: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute configured metrics for a batch.

        Metrics compare `x_recon` against `x_clean` when available. If
        `x_clean` is absent, `x_noisy` is used as the reference.
        """
        reference = x_noisy if x_clean is None else x_clean
        if reference.shape != x_recon.shape or x_noisy.shape != x_recon.shape:
            raise ValueError("x_noisy, x_recon, and x_clean must share shape [B, C, H, W].")
        batch_size = x_recon.shape[0]
        outputs: dict[str, torch.Tensor] = {}

        outputs.update(
            selected_pixel_metrics(
                reference,
                x_recon,
                self.pixel_metrics,
                lpips_scorer=self.lpips_scorer,
            )
        )

        if self.texture_enabled:
            try:
                outputs.update(
                    texture_metrics(
                        rgb_to_gray_01(reference),
                        rgb_to_gray_01(x_recon),
                        nbins=int(self.texture_cfg.get("nbins", 256)),
                        include_lbp=bool(self.texture_cfg.get("include_lbp", True)),
                        include_hog=bool(self.texture_cfg.get("include_hog", True)),
                        include_intensity=bool(self.texture_cfg.get("include_intensity", True)),
                        include_dtcwt=bool(self.texture_cfg.get("include_dtcwt", False)),
                    )
                )
            except ImportError:
                if not self.texture_optional:
                    raise

        if self.local_complexity_enabled:
            assert self.local_complexity is not None
            outputs["local_complexity"] = self.local_complexity(x_noisy)

        return self._validate_outputs(outputs, batch_size)
