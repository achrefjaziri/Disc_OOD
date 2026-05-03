"""Tests for DISC feature archive IO."""

import numpy as np

from disc.io import (
    feature_file_path,
    load_feature_file,
    merge_feature_shards,
    save_feature_shard,
    shard_file_path,
)


def test_feature_archive_schema_and_shard_merge(tmp_path):
    cfg = {"features": {"timesteps": [4, 1]}}
    feature_names = ["t4/mse", "t1/mse"]
    feature_dir = tmp_path / "features"

    save_feature_shard(
        shard_file_path(feature_dir, "imagenet_val", 0),
        features=np.ones((2, 2), dtype=np.float32),
        feature_names=feature_names,
        dataset_name="imagenet_val",
        family="ImageNet",
        role="id",
        sample_ids=np.asarray([0, 1]),
        labels=np.asarray([3, 4]),
        paths=["a.JPEG", "b.JPEG"],
        source_names=["imagenet_val", "imagenet_val"],
        timesteps=[4, 1],
        config=cfg,
    )
    save_feature_shard(
        shard_file_path(feature_dir, "imagenet_val", 1),
        features=np.zeros((1, 2), dtype=np.float32),
        feature_names=feature_names,
        dataset_name="imagenet_val",
        family="ImageNet",
        role="id",
        sample_ids=np.asarray([2]),
        labels=np.asarray([5]),
        paths=["c.JPEG"],
        source_names=["imagenet_val"],
        timesteps=[4, 1],
        config=cfg,
    )

    out_path = feature_file_path(feature_dir, "imagenet_val")
    merged = merge_feature_shards(
        sorted((feature_dir / "_shards" / "imagenet_id").glob("shard_*.npz")),
        out_path,
        config=cfg,
    )
    loaded = load_feature_file(out_path)

    assert out_path.name == "imagenet_id_disc_features.npz"
    assert merged["features"].shape == (3, 2)
    assert loaded["X"].shape == (3, 2)
    assert loaded["labels"].tolist() == [3, 4, 5]
    assert loaded["targets"].tolist() == [3, 4, 5]
    assert loaded["sample_ids"].tolist() == [0, 1, 2]
    assert loaded["feature_names"] == feature_names
    assert loaded["timesteps"].tolist() == [4, 1]
    assert loaded["family"] == "ImageNet"
    assert loaded["config_hash"]
