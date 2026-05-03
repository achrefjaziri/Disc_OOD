"""Diffusion reconstruction wrapper for ImageNet DISC.

The default backend uses OpenAI's ``improved_diffusion`` package with the
ImageNet-64 unconditional checkpoint ``imagenet64_uncond_100M_1500K.pt``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


ACTUAL_IMAGENET64_MODEL_ARGS: dict[str, Any] = {
    "image_size": 64,
    "num_channels": 128,
    "num_res_blocks": 3,
    "num_heads": 4,
    "num_heads_upsample": -1,
    "attention_resolutions": "16,8",
    "dropout": 0.0,
    "learn_sigma": True,
    "sigma_small": False,
    "class_cond": False,
    "diffusion_steps": 4000,
    "noise_schedule": "cosine",
    "timestep_respacing": "",
    "use_kl": False,
    "predict_xstart": False,
    "rescale_timesteps": True,
    "rescale_learned_sigmas": True,
    "use_checkpoint": False,
    "use_scale_shift_norm": True,
}


@dataclass
class ReconstructionBatch:
    """Outputs for one requested diffusion timestep.

    Attributes:
        x_noisy: The current noisy/reverse-chain sample at timestep ``t``.
        x_recon: Reconstruction target:
            ``improved_diffusion.p_sample(...)[\"sample\"]``.
        x0_hat: The model's optional predicted clean image,
            ``p_sample(...)[\"pred_xstart\"]``.
        pred_noise: Predicted epsilon when the backend can derive it.
        timestep: Raw improved-diffusion timestep index.
        metadata: Convention and backend information for this reconstruction.
    """

    x_noisy: torch.Tensor
    x_recon: torch.Tensor
    x0_hat: torch.Tensor | None
    pred_noise: torch.Tensor | None
    timestep: int
    metadata: dict[str, Any]


class DiffusionReconstructor:
    """Model-agnostic reconstruction interface with an improved-diffusion backend."""

    def __init__(self, config: dict[str, Any], device: torch.device | str):
        self.full_config = config
        self.config = dict(config.get("diffusion", config))
        configured_device = self.config.get("device")
        self.device = torch.device(device if configured_device in (None, "", "auto") else configured_device)
        self.backend = self.config.get("backend", "improved_diffusion")
        self.scheduler = self.config.get("scheduler", "improved_diffusion_gaussian")
        self.prediction_type = self.config.get("prediction_type", "epsilon")
        self.reconstruction_mode = self.config.get(
            "reconstruction_mode",
            config.get("features", {}).get("mode", "full_reverse"),
        )
        self.mixed_precision = bool(self.config.get("mixed_precision", False))
        self.model: torch.nn.Module | None = None
        self.diffusion: Any | None = None

        if self.backend in {"dummy", "smoke_test"}:
            self.num_timesteps = int(self.config.get("diffusion_steps", 4000))
            return
        if self.backend not in {"improved_diffusion", "openai_improved_diffusion", "custom"}:
            raise ValueError(
                f"Unsupported diffusion backend '{self.backend}'. Use 'improved_diffusion' "
                "for the ImageNet DISC pipeline."
            )
        if self.scheduler not in {None, "improved_diffusion_gaussian", "improved_diffusion_ddim"}:
            raise ValueError(
                f"Unsupported scheduler '{self.scheduler}'. Use improved_diffusion's "
                "GaussianDiffusion or DDIM samplers."
            )
        if self.config.get("num_inference_steps") not in (None, "", 0):
            raise ValueError(
                "num_inference_steps is not used by this improved_diffusion wrapper. "
                "Configure explicit integer timesteps instead."
            )
        self._load_improved_diffusion()

    def _checkpoint_path(self) -> Path:
        raw_path = self.config.get("checkpoint_path") or self.config.get("model_name_or_path")
        if not raw_path:
            raise ValueError(
                "diffusion.checkpoint_path or diffusion.model_name_or_path is required "
                "for the improved_diffusion backend."
            )
        path = Path(str(raw_path))
        if not path.exists():
            raise FileNotFoundError(
                f"Improved-diffusion checkpoint not found: {path}. "
                "Set diffusion.checkpoint_path to the ImageNet-64 checkpoint, e.g. "
                "imagenet64_uncond_100M_1500K.pt."
            )
        return path

    def _load_improved_diffusion(self) -> None:
        try:
            from improved_diffusion.script_util import create_model_and_diffusion
        except ImportError as exc:
            raise ImportError(
                "Could not import improved_diffusion. Install the OpenAI "
                "improved-diffusion package or make a local copy "
                "importable before running feature extraction."
            ) from exc

        model_args = dict(ACTUAL_IMAGENET64_MODEL_ARGS)
        model_args.update(dict(self.config.get("model_args", {})))
        checkpoint_path = self._checkpoint_path()
        model, diffusion = create_model_and_diffusion(**model_args)
        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state)
        model.to(self.device).eval()
        self.model = model
        self.diffusion = diffusion
        self.num_timesteps = int(getattr(diffusion, "num_timesteps", model_args["diffusion_steps"]))

    def _validate_timesteps(self, timesteps: list[int]) -> list[int]:
        if not timesteps:
            raise ValueError("At least one diffusion timestep is required.")
        clean = sorted({int(t) for t in timesteps}, reverse=True)
        bad = [t for t in clean if t < 0 or t >= self.num_timesteps]
        if bad:
            raise ValueError(
                f"Timesteps {bad} are outside valid improved_diffusion range "
                f"[0, {self.num_timesteps - 1}]."
            )
        return clean

    def _scaled_timestep(self, timestep: int) -> float:
        if self.diffusion is None:
            return float(timestep)
        tensor = torch.tensor([timestep], dtype=torch.long, device=self.device)
        scaled = self.diffusion._scale_timesteps(tensor)
        return float(scaled.item())

    @torch.no_grad()
    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Add noise using the configured backend's forward process."""
        if self.backend in {"dummy", "smoke_test"}:
            return x_start if noise is None else x_start + 0.01 * noise
        assert self.diffusion is not None
        return self.diffusion.q_sample(x_start, t, noise=noise)

    @torch.no_grad()
    def _p_sample_output(self, x: torch.Tensor, t: torch.Tensor) -> dict[str, torch.Tensor | None]:
        if self.backend in {"dummy", "smoke_test"}:
            return {
                "sample": x * 0.95,
                "pred_xstart": x,
                "pred_noise": torch.zeros_like(x),
            }
        assert self.model is not None and self.diffusion is not None
        with torch.autocast(device_type=self.device.type, enabled=self.mixed_precision):
            out = self.diffusion.p_sample(self.model, x, t)
        pred_noise = None
        if out.get("pred_xstart") is not None and hasattr(self.diffusion, "_predict_eps_from_xstart"):
            pred_noise = self.diffusion._predict_eps_from_xstart(x, t, out["pred_xstart"])
        return {
            "sample": out["sample"],
            "pred_xstart": out.get("pred_xstart"),
            "pred_noise": pred_noise,
        }

    @torch.no_grad()
    def _ddim_sample_output(self, x: torch.Tensor, t: torch.Tensor) -> dict[str, torch.Tensor | None]:
        if self.backend in {"dummy", "smoke_test"}:
            return {
                "sample": x * 0.95,
                "pred_xstart": x,
                "pred_noise": torch.zeros_like(x),
            }
        assert self.model is not None and self.diffusion is not None
        eta = float(self.config.get("ddim_eta", 0.0))
        with torch.autocast(device_type=self.device.type, enabled=self.mixed_precision):
            out = self.diffusion.ddim_sample(self.model, x, t, eta=eta)
        pred_noise = None
        if out.get("pred_xstart") is not None and hasattr(self.diffusion, "_predict_eps_from_xstart"):
            pred_noise = self.diffusion._predict_eps_from_xstart(x, t, out["pred_xstart"])
        return {
            "sample": out["sample"],
            "pred_xstart": out.get("pred_xstart"),
            "pred_noise": pred_noise,
        }

    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Return the metric reconstruction sample."""
        return self._p_sample_output(x, t)["sample"]  # type: ignore[return-value]

    def _metadata(self, timestep: int) -> dict[str, Any]:
        x_recon_convention = (
            "improved_diffusion.ddim_sample(...)[\"sample\"]"
            if self.reconstruction_mode == "ddim_reverse"
            else "improved_diffusion.p_sample(...)[\"sample\"]"
        )
        return {
            "backend": self.backend,
            "scheduler": self.scheduler,
            "prediction_type": self.prediction_type,
            "reconstruction_mode": self.reconstruction_mode,
            "timestep": timestep,
            "original_timestep": self._original_timestep(timestep),
            "scaled_timestep": self._scaled_timestep(timestep),
            "x_recon_convention": x_recon_convention,
            "x0_hat_convention": "improved_diffusion.p_sample(...)[\"pred_xstart\"]",
        }

    def _original_timestep(self, timestep: int) -> int:
        if self.diffusion is not None and hasattr(self.diffusion, "timestep_map"):
            return int(self.diffusion.timestep_map[timestep])
        return int(timestep)

    @torch.no_grad()
    def _reconstruct_single_step(
        self, x: torch.Tensor, timesteps: list[int]
    ) -> dict[int, ReconstructionBatch]:
        out: dict[int, ReconstructionBatch] = {}
        x = x.to(self.device)
        batch_size = x.shape[0]
        for timestep in timesteps:
            t = torch.full((batch_size,), timestep, dtype=torch.long, device=self.device)
            x_noisy = self.q_sample(x, t, noise=torch.randn_like(x))
            step = self._p_sample_output(x_noisy, t)
            out[timestep] = ReconstructionBatch(
                x_noisy=x_noisy,
                x_recon=step["sample"],  # type: ignore[arg-type]
                x0_hat=step["pred_xstart"],
                pred_noise=step["pred_noise"],
                timestep=timestep,
                metadata=self._metadata(timestep),
            )
        return out

    @torch.no_grad()
    def _reconstruct_full_reverse(
        self, x: torch.Tensor, timesteps: list[int]
    ) -> dict[int, ReconstructionBatch]:
        """Run q_sample at max(t), then p_sample downwards."""
        out: dict[int, ReconstructionBatch] = {}
        x = x.to(self.device)
        requested = set(timesteps)
        batch_size = x.shape[0]
        start_t = max(requested)
        end_t = min(requested)
        t_start = torch.full((batch_size,), start_t, dtype=torch.long, device=self.device)
        sample = self.q_sample(x, t_start, noise=torch.randn_like(x))

        for timestep in range(start_t, end_t - 1, -1):
            t = torch.full((batch_size,), timestep, dtype=torch.long, device=self.device)
            x_noisy = sample
            step = self._p_sample_output(x_noisy, t)
            sample = step["sample"]  # type: ignore[assignment]
            if timestep in requested:
                metadata = self._metadata(timestep)
                metadata["x_noisy_convention"] = (
                    "current reverse-chain x_t; at max timestep this is q_sample(x0, t)"
                )
                out[timestep] = ReconstructionBatch(
                    x_noisy=x_noisy,
                    x_recon=sample,
                    x0_hat=step["pred_xstart"],
                    pred_noise=step["pred_noise"],
                    timestep=timestep,
                    metadata=metadata,
                )
        return out

    @torch.no_grad()
    def _reconstruct_ddim_reverse(
        self, x: torch.Tensor, timesteps: list[int]
    ) -> dict[int, ReconstructionBatch]:
        """Run a DDIM reverse trajectory and record requested timesteps."""
        out: dict[int, ReconstructionBatch] = {}
        x = x.to(self.device)
        requested = set(timesteps)
        batch_size = x.shape[0]
        start_t = max(requested)
        end_t = min(requested)
        t_start = torch.full((batch_size,), start_t, dtype=torch.long, device=self.device)
        sample = self.q_sample(x, t_start, noise=torch.randn_like(x))

        for timestep in range(start_t, end_t - 1, -1):
            t = torch.full((batch_size,), timestep, dtype=torch.long, device=self.device)
            x_noisy = sample
            step = self._ddim_sample_output(x_noisy, t)
            sample = step["sample"]  # type: ignore[assignment]
            if timestep in requested:
                metadata = self._metadata(timestep)
                metadata["x_noisy_convention"] = (
                    "current DDIM reverse-chain x_t; at max timestep this is q_sample(x0, t)"
                )
                out[timestep] = ReconstructionBatch(
                    x_noisy=x_noisy,
                    x_recon=sample,
                    x0_hat=step["pred_xstart"],
                    pred_noise=step["pred_noise"],
                    timestep=timestep,
                    metadata=metadata,
                )
        return out

    @torch.no_grad()
    def reconstruct(
        self,
        x: torch.Tensor,
        timesteps: list[int],
    ) -> dict[int, ReconstructionBatch]:
        """Return noisy and reconstructed batches for requested timesteps.

        ``full_reverse`` adds noise once at the largest requested timestep,
        runs stochastic ``p_sample`` down to the smallest requested timestep,
        and returns the ``p_sample`` sample at requested indices. ``single_step``
        treats each timestep independently by starting from ``q_sample(x0, t)``.
        ``ddim_reverse`` follows the same trajectory interface with DDIM steps.
        """
        clean_timesteps = self._validate_timesteps(timesteps)
        if self.reconstruction_mode == "single_step":
            return self._reconstruct_single_step(x, clean_timesteps)
        if self.reconstruction_mode == "full_reverse":
            return self._reconstruct_full_reverse(x, clean_timesteps)
        if self.reconstruction_mode == "ddim_reverse":
            return self._reconstruct_ddim_reverse(x, clean_timesteps)
        raise ValueError(
            f"Unsupported reconstruction_mode '{self.reconstruction_mode}'. "
            "Use 'full_reverse', 'single_step', or 'ddim_reverse'."
        )


def build_diffusion_backend(cfg: dict[str, Any], device: torch.device) -> DiffusionReconstructor:
    """Create the configured DISC diffusion reconstructor."""
    return DiffusionReconstructor(cfg, device)
