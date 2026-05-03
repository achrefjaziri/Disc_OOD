#!/usr/bin/env python
"""Create paper-facing tables from DISC evaluation outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from disc.config import load_config
from disc.utils import ensure_dir


def _format_cell(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    """Write a small GitHub-flavored Markdown table without optional deps."""
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_format_cell(row[col]) for col in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "±": r"$\pm$",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def write_latex_table(
    df: pd.DataFrame,
    path: Path,
    *,
    caption: str,
    label: str,
) -> None:
    """Write a self-contained LaTeX table using only standard tabular syntax."""
    align = "l" * len(df.columns)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{_latex_escape(caption)}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{align}}}",
        r"\hline",
        " & ".join(_latex_escape(col) for col in df.columns) + r" \\",
        r"\hline",
    ]
    for _, row in df.iterrows():
        lines.append(" & ".join(_latex_escape(row[col]) for col in df.columns) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _format_metric(mean: float, std: float | None = None, repeats: int = 1) -> str:
    if std is not None and repeats > 1 and not pd.isna(std):
        return f"{mean:.4f} ± {std:.4f}"
    return f"{mean:.4f}"


def _load_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing evaluation summary: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _results_dir(cfg: dict[str, Any], override: str | None) -> Path:
    if override:
        return Path(override)
    paths = cfg.get("paths", {})
    return Path(paths.get("results_dir", paths.get("evaluation_dir", "outputs/results")))


def _tables_dir(cfg: dict[str, Any], override: str | None) -> Path:
    if override:
        return ensure_dir(override)
    paths = cfg.get("paths", {})
    output_dir = Path(paths.get("output_dir", "outputs"))
    return ensure_dir(output_dir / "tables")


def _id_display_name(summary: dict[str, Any]) -> str:
    dataset = summary.get("datasets", {}).get("id_dataset", "ImageNet")
    if dataset in {"imagenet_val", "imagenet_id", "id", "imagenet"}:
        return "ImageNet"
    return str(dataset)


def _ood_family_list(summary: dict[str, Any]) -> list[str]:
    datasets = summary.get("datasets", {})
    ood_names = list(datasets.get("ood_datasets", []))
    family_map = dict(datasets.get("ood_families", {}))
    families = [str(family_map.get(name, name)) for name in ood_names]
    return families


def _summary_row(
    *,
    summary: dict[str, Any],
    binary: pd.DataFrame,
    clustering: pd.DataFrame,
    supervised: pd.DataFrame,
) -> dict[str, str]:
    binary_average = binary[binary["dataset"] == "average"]
    if not binary_average.empty:
        avg_auroc = float(binary_average.iloc[0]["auroc"])
    else:
        avg_auroc = float(summary["binary_auroc"]["average"])

    clustering_summary = clustering[clustering["row_type"] == "summary"].iloc[0]
    supervised_summary = supervised[supervised["row_type"] == "summary"].iloc[0]
    clustering_repeats = int((clustering["row_type"] == "run").sum())
    supervised_repeats = int((supervised["row_type"] == "run").sum())
    families = _ood_family_list(summary)

    return {
        "Method": "DISC (ours)",
        "ID dataset": _id_display_name(summary),
        "OOD families": ", ".join(families),
        "Avg AUROC": f"{avg_auroc:.4f}",
        "Clust. Acc": _format_metric(
            float(clustering_summary["accuracy_mean"]),
            float(clustering_summary["accuracy_std"]),
            clustering_repeats,
        ),
        "Sup. Acc": _format_metric(
            float(supervised_summary["accuracy_mean"]),
            float(supervised_summary["accuracy_std"]),
            supervised_repeats,
        ),
    }


def _per_ood_table(binary: pd.DataFrame) -> pd.DataFrame:
    rows = binary[binary["dataset"] != "average"].copy()
    if rows.empty:
        raise ValueError("binary_auroc.csv does not contain per-OOD rows.")
    out = pd.DataFrame(
        {
            "OOD family": rows["family"].astype(str),
            "Dataset": rows["dataset"].astype(str),
            "AUROC": rows["auroc"].map(lambda value: f"{float(value):.4f}"),
        }
    )
    return out.sort_values("OOD family").reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/imagenet_disc.yaml")
    parser.add_argument("--results_dir", default=None, help="Override paths.results_dir.")
    parser.add_argument("--output_dir", default=None, help="Override the tables output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    results_dir = _results_dir(cfg, args.results_dir)
    out_dir = _tables_dir(cfg, args.output_dir)

    binary_path = results_dir / "binary_auroc.csv"
    clustering_path = results_dir / "clustering_accuracy.csv"
    supervised_path = results_dir / "supervised_accuracy.csv"
    summary_path = results_dir / "summary.json"
    missing = [
        path
        for path in [binary_path, clustering_path, supervised_path, summary_path]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing evaluation result files: {missing}")

    binary = pd.read_csv(binary_path)
    clustering = pd.read_csv(clustering_path)
    supervised = pd.read_csv(supervised_path)
    summary = _load_summary(summary_path)

    compact = pd.DataFrame(
        [
            _summary_row(
                summary=summary,
                binary=binary,
                clustering=clustering,
                supervised=supervised,
            )
        ]
    )
    per_ood = _per_ood_table(binary)

    compact_md = out_dir / "imagenet_disc_results.md"
    compact_tex = out_dir / "imagenet_disc_results.tex"
    per_ood_md = out_dir / "imagenet_disc_per_ood_auroc.md"

    write_markdown_table(compact, compact_md)
    write_latex_table(
        compact,
        compact_tex,
        caption="Minimal ImageNet DISC replication results.",
        label="tab:imagenet_disc_minimal",
    )
    write_markdown_table(per_ood, per_ood_md)

    print(f"Wrote tables -> {out_dir}")
    print(f"  {compact_md}")
    print(f"  {compact_tex}")
    print(f"  {per_ood_md}")


if __name__ == "__main__":
    main()
