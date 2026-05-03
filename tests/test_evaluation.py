"""Tests for DISC-only evaluation routines."""

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from disc.evaluation import binary_iforest_aurocs, run_disc_evaluations


def _cluster(center, n=24, seed=0):
    rng = np.random.RandomState(seed)
    return (np.asarray(center, dtype=np.float32) + 0.05 * rng.randn(n, 3)).astype(np.float32)


def _feature_sets():
    return {
        "imagenet_val": {
            "features": _cluster([0.0, 0.0, 0.0], n=32, seed=1),
            "family": "ImageNet",
        },
        "imagenet_a": {
            "features": _cluster([4.0, 0.0, 0.0], seed=2),
            "family": "ImageNet-A",
        },
        "imagenet_o": {
            "features": _cluster([0.0, 4.0, 0.0], seed=3),
            "family": "ImageNet-O",
        },
        "cifar10": {
            "features": _cluster([0.0, 0.0, 4.0], seed=4),
            "family": "CIFAR-10",
        },
    }


def test_binary_iforest_reports_average_and_ood_score_direction():
    rows = binary_iforest_aurocs(
        _feature_sets(),
        id_dataset="imagenet_id",
        ood_datasets=["imagenet_a", "imagenet_o", "cifar10"],
        train_fraction=1.0,
        n_estimators=25,
        seed=0,
    )

    assert rows.iloc[-1]["dataset"] == "average"
    assert rows["score_direction"].unique().tolist() == ["higher_is_more_ood"]
    assert np.isclose(rows.iloc[-1]["auroc"], rows.iloc[:-1]["auroc"].mean())
    assert rows["auroc"].between(0.0, 1.0).all()


def test_run_disc_evaluations_repeats_seeded_protocols():
    cfg = {
        "id_dataset": "imagenet_val",
        "ood_datasets": ["imagenet_a", "imagenet_o", "cifar10"],
        "scale_features": True,
        "seeds": [0, 1],
        "iforest": {"train_fraction": 1.0, "n_estimators": 25},
        "kmeans": {"n_init": 5},
        "supervised": {"test_size": 0.25, "hidden_layer_sizes": [12], "max_iter": 300},
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        binary, clustering, supervised, summary = run_disc_evaluations(_feature_sets(), cfg)

    assert binary.iloc[-1]["dataset"] == "average"
    assert clustering[clustering["row_type"] == "run"].shape[0] == 2
    assert supervised[supervised["row_type"] == "run"].shape[0] == 2
    assert summary["binary_auroc"]["average"] == binary.iloc[-1]["auroc"]
    assert summary["clustering_accuracy"]["n_repeats"] == 2
    assert summary["supervised_accuracy"]["n_repeats"] == 2
