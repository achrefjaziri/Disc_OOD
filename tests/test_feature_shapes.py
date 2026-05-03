"""Shape tests for DISC feature extraction using a fake diffusion backend."""

import torch

from disc.features import compute_disc_features_for_batch


class FakeDiffusionBackend:
    """Small deterministic backend for feature-shape tests."""

    def q_sample(self, x_start, t, noise=None):
        noise = torch.zeros_like(x_start) if noise is None else noise
        return x_start + 0.01 * noise

    def p_sample(self, x, t):
        return x * 0.9


def test_single_step_feature_shape_without_heavy_metrics():
    images = torch.randn(5, 3, 16, 16)
    cfg = {
        "mode": "single_step",
        "timesteps": [10, 5],
        "comparison": "orig_vs_recon",
        "pixel_metrics": ["mse", "ssim"],
        "texture": {"enabled": False},
    }
    batch = compute_disc_features_for_batch(images, FakeDiffusionBackend(), cfg)
    assert batch.X.shape == (5, 4)
    assert batch.feature_names == ["t10/mse", "t10/ssim", "t5/mse", "t5/ssim"]

