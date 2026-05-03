"""Smoke tests for the diffusion reconstructor interface."""

import torch

from disc.diffusion import DiffusionReconstructor, ReconstructionBatch


def test_dummy_reconstructor_returns_requested_batches():
    cfg = {
        "diffusion": {
            "backend": "dummy",
            "diffusion_steps": 16,
            "reconstruction_mode": "single_step",
        }
    }
    reconstructor = DiffusionReconstructor(cfg, torch.device("cpu"))
    x = torch.randn(2, 3, 8, 8)
    out = reconstructor.reconstruct(x, [4, 1])

    assert sorted(out) == [1, 4]
    assert isinstance(out[4], ReconstructionBatch)
    assert out[4].x_noisy.shape == x.shape
    assert out[4].x_recon.shape == x.shape
    assert out[4].x0_hat is not None
    assert out[4].pred_noise is not None
    assert out[4].metadata["x_recon_convention"] == 'improved_diffusion.p_sample(...)["sample"]'

