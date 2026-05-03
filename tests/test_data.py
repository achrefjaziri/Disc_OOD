"""Smoke tests for DISC dataloaders and metadata batches."""

from pathlib import Path

from PIL import Image

from disc.data import build_dataloaders, discover_imagenet_c_folders, smoke_check_dataloaders


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 10), color=(128, 64, 32)).save(path)


def _make_imagefolder(root: Path, count: int = 3) -> None:
    for idx in range(count):
        _write_image(root / "n00000000" / f"{idx}.JPEG")


def _base_cfg(tmp_path: Path) -> dict:
    return {
        "data": {
            "imagenet_val_root": "",
            "imagenet_a_root": "",
            "imagenet_o_root": "",
            "imagenet_c_root": "",
            "cifar10_root": "",
            "mnist_root": "",
            "image_size": 16,
            "batch_size": 2,
            "num_workers": 0,
            "max_samples_per_dataset": None,
            "seed": 7,
            "normalization": "minus_one_one",
        }
    }


def test_imagenet_folder_loader_returns_bchw_and_metadata(tmp_path):
    root = tmp_path / "imagenet_val"
    _make_imagefolder(root)
    cfg = _base_cfg(tmp_path)
    cfg["data"]["imagenet_val_root"] = str(root)

    loaders = build_dataloaders(cfg)
    shapes = smoke_check_dataloaders(loaders)
    batch = next(iter(loaders["imagenet_val"]))

    assert shapes["imagenet_val"] == (2, 3, 16, 16)
    assert batch["dataset_name"] == ["imagenet_val", "imagenet_val"]
    assert batch["ood_family"] == ["ImageNet", "ImageNet"]
    assert batch["label"].shape[0] == 2
    assert batch["original_index"].tolist() == [0, 1]


def test_imagenet_c_nested_root_is_combined_as_one_family(tmp_path):
    root = tmp_path / "imagenet_c"
    _make_imagefolder(root / "gaussian_noise" / "1", count=1)
    _make_imagefolder(root / "brightness" / "5", count=1)
    cfg = _base_cfg(tmp_path)
    cfg["data"]["imagenet_c_root"] = str(root)
    cfg["data"]["batch_size"] = 4

    discovered = discover_imagenet_c_folders(root)
    assert [source for _, source in discovered] == [
        "imagenet_c/brightness/5",
        "imagenet_c/gaussian_noise/1",
    ]

    loaders = build_dataloaders(cfg)
    shapes = smoke_check_dataloaders(loaders)
    batch = next(iter(loaders["imagenet_c"]))

    assert shapes["imagenet_c"] == (2, 3, 16, 16)
    assert set(batch["dataset_name"]) == {"imagenet_c"}
    assert set(batch["ood_family"]) == {"ImageNet-C"}
    assert set(batch["source_name"]) == {
        "imagenet_c/brightness/5",
        "imagenet_c/gaussian_noise/1",
    }


def test_max_samples_per_dataset_limits_loader(tmp_path):
    root = tmp_path / "imagenet_a"
    _make_imagefolder(root, count=5)
    cfg = _base_cfg(tmp_path)
    cfg["data"]["imagenet_a_root"] = str(root)
    cfg["data"]["max_samples_per_dataset"] = 2
    cfg["data"]["batch_size"] = 8

    loaders = build_dataloaders(cfg)
    batch = next(iter(loaders["imagenet_a"]))

    assert batch["image"].shape == (2, 3, 16, 16)
    assert batch["original_index"].tolist() == [0, 1]

