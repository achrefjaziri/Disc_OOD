#!/usr/bin/env python
"""Run DISC evaluations on saved feature files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from disc.config import load_config
from disc.evaluation import run_disc_evaluations
from disc.io import load_feature_dir
from disc.utils import ensure_dir, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/imagenet_disc.yaml")
    parser.add_argument(
        "--feature_dir",
        default=None,
        help="Override paths.feature_dir from the YAML config.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Override paths.results_dir from the YAML config.",
    )
    return parser.parse_args()


def _results_dir(cfg: dict, override: str | None) -> Path:
    if override:
        return ensure_dir(override)
    paths = cfg.get("paths", {})
    if paths.get("results_dir"):
        return ensure_dir(paths["results_dir"])
    output_dir = Path(paths.get("output_dir", "outputs"))
    return ensure_dir(output_dir / "results")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    feature_dir = Path(args.feature_dir or cfg["paths"]["feature_dir"])
    out_dir = _results_dir(cfg, args.output_dir)
    feature_sets = load_feature_dir(feature_dir)
    if not feature_sets:
        raise FileNotFoundError(f"No feature archives found under {feature_dir}")

    binary_df, clustering_df, supervised_df, summary = run_disc_evaluations(
        feature_sets,
        cfg["evaluation"],
        default_seed=int(cfg.get("seed", 42)),
    )

    binary_path = out_dir / "binary_auroc.csv"
    clustering_path = out_dir / "clustering_accuracy.csv"
    supervised_path = out_dir / "supervised_accuracy.csv"
    summary_path = out_dir / "summary.json"

    binary_df.to_csv(binary_path, index=False)
    clustering_df.to_csv(clustering_path, index=False)
    supervised_df.to_csv(supervised_path, index=False)
    write_json(summary_path, summary)

    print("Binary ID-vs-OOD AUROC")
    print(binary_df.to_string(index=False))
    print("\nMulti-OOD clustering accuracy")
    print(clustering_df.to_string(index=False))
    print("\nSupervised OOD-family accuracy")
    print(supervised_df.to_string(index=False))
    print(f"\nSaved results -> {out_dir}")


if __name__ == "__main__":
    main()
