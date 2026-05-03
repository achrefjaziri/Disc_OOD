"""Tests for per-image DISC metric helpers."""

import torch

from disc.metrics import MetricComputer, mse_score_imagewise, rgb_to_gray_01, ssim_score_imagewise


def test_mse_shape_and_zero_for_identical_inputs():
    x = torch.randn(4, 3, 16, 16)
    values = mse_score_imagewise(x, x)
    assert values.shape == (4,)
    assert torch.allclose(values, torch.zeros_like(values))


def test_ssim_shape_for_identical_inputs():
    x = torch.rand(3, 3, 16, 16) * 2 - 1
    values = ssim_score_imagewise(x, x)
    assert values.shape == (3,)
    assert torch.all(values > 0.99)


def test_rgb_to_gray_range_and_shape():
    x = torch.rand(2, 3, 8, 8) * 2 - 1
    gray = rgb_to_gray_01(x)
    assert gray.shape == (2, 1, 8, 8)
    assert float(gray.min()) >= 0.0
    assert float(gray.max()) <= 1.0


def test_metric_computer_preserves_batch_dimension():
    x_clean = torch.randn(4, 3, 16, 16).clamp(-1, 1)
    x_noisy = x_clean + 0.05 * torch.randn_like(x_clean)
    x_recon = x_clean + 0.02 * torch.randn_like(x_clean)
    computer = MetricComputer(
        {
            "pixel_metrics": ["mse", "ssim"],
            "texture": {
                "enabled": True,
                "include_lbp": True,
                "include_hog": True,
                "include_intensity": True,
                "include_dtcwt": False,
            },
            "local_complexity": {"enabled": False},
        },
        torch.device("cpu"),
    )

    outputs = computer.compute(x_noisy, x_recon, x_clean=x_clean)

    assert "mse" in outputs
    assert "ssim" in outputs
    assert "lbp_KL" in outputs
    assert "hog_KL" in outputs
    assert "intensity_Euclidean" in outputs
    for value in outputs.values():
        assert value.shape[0] == 4
        assert torch.isfinite(value).all()


def test_metric_computer_no_nans_for_flat_inputs():
    x = torch.zeros(3, 3, 16, 16)
    computer = MetricComputer(
        {
            "pixel_metrics": ["mse", "ssim"],
            "texture": {
                "enabled": True,
                "include_lbp": True,
                "include_hog": True,
                "include_intensity": True,
                "include_dtcwt": False,
            },
        },
        "cpu",
    )

    outputs = computer.compute(x, x, x_clean=x)

    for name, value in outputs.items():
        assert value.shape[0] == 3, name
        assert torch.isfinite(value).all(), name
