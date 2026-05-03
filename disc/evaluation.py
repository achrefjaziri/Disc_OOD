"""Evaluation routines for saved DISC feature matrices.

The evaluator uses only saved DISC features. It does not train classifier-based
OOD detectors and does not use ImageNet class labels, except as archive
metadata carried through from feature extraction.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


DATASET_ALIASES = {
    "id": "imagenet_val",
    "imagenet": "imagenet_val",
    "imagenet_id": "imagenet_val",
}


def _resolve_dataset_name(feature_sets: dict[str, dict[str, Any]], name: str) -> str:
    """Resolve common dataset aliases against loaded feature archive keys."""
    if name in feature_sets:
        return name
    alias = DATASET_ALIASES.get(name, name)
    if alias in feature_sets:
        return alias
    if name == "imagenet_val" and "imagenet_id" in feature_sets:
        return "imagenet_id"
    available = ", ".join(sorted(feature_sets)) or "none"
    raise KeyError(f"Missing feature file for dataset '{name}'. Available: {available}")


def resolve_dataset_names(
    feature_sets: dict[str, dict[str, Any]],
    *,
    id_dataset: str,
    ood_datasets: list[str],
) -> tuple[str, list[str]]:
    """Resolve configured ID/OOD dataset names to loaded feature-set keys."""
    resolved_id = _resolve_dataset_name(feature_sets, id_dataset)
    resolved_ood = [_resolve_dataset_name(feature_sets, name) for name in ood_datasets]
    if resolved_id in resolved_ood:
        raise ValueError("The ID dataset is also listed as OOD.")
    return resolved_id, resolved_ood


def _feature_matrix(item: dict[str, Any], dataset_name: str) -> np.ndarray:
    """Return a finite 2D feature matrix from a loaded feature archive."""
    X = np.asarray(item.get("features", item.get("X")), dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"Dataset '{dataset_name}' has feature shape {X.shape}, expected [N, D].")
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError(f"Dataset '{dataset_name}' has an empty feature matrix {X.shape}.")
    if not np.isfinite(X).all():
        raise FloatingPointError(f"Dataset '{dataset_name}' contains NaN or Inf features.")
    return X


def _family_name(item: dict[str, Any], dataset_name: str) -> str:
    family = str(item.get("family", dataset_name))
    return family if family else dataset_name


def _maybe_scale(X: np.ndarray, enabled: bool) -> tuple[np.ndarray, StandardScaler | None]:
    if not enabled:
        return X, None
    scaler = StandardScaler()
    return scaler.fit_transform(X), scaler


def _configured_seeds(seed_or_seeds: int | list[int] | tuple[int, ...]) -> list[int]:
    if isinstance(seed_or_seeds, (list, tuple)):
        if not seed_or_seeds:
            raise ValueError("At least one evaluation seed is required.")
        return [int(seed) for seed in seed_or_seeds]
    return [int(seed_or_seeds)]


def _safe_std(values: list[float]) -> float:
    return 0.0 if len(values) <= 1 else float(np.std(values, ddof=1))


def binary_iforest_aurocs(
    feature_sets: dict[str, dict[str, Any]],
    *,
    id_dataset: str,
    ood_datasets: list[str],
    train_fraction: float,
    n_estimators: int,
    seed: int,
    scale_features: bool = True,
) -> pd.DataFrame:
    """Compute ID-vs-each-OOD AUROC using IsolationForest fitted on ID features.

    Scores are negated IsolationForest normality scores, so larger values mean
    more OOD-like before `roc_auc_score` is called.
    """
    id_dataset, ood_datasets = resolve_dataset_names(
        feature_sets,
        id_dataset=id_dataset,
        ood_datasets=ood_datasets,
    )
    X_id = _feature_matrix(feature_sets[id_dataset], id_dataset)
    rng = np.random.RandomState(seed)
    if train_fraction <= 0 or train_fraction > 1:
        raise ValueError("iforest.train_fraction must be in (0, 1].")
    train_size = max(1, int(round(train_fraction * X_id.shape[0])))
    train_idx = rng.choice(X_id.shape[0], size=train_size, replace=False)

    if scale_features:
        scaler = StandardScaler().fit(X_id[train_idx])
        X_id_model = scaler.transform(X_id[train_idx])
    else:
        scaler = None
        X_id_model = X_id[train_idx]

    clf = IsolationForest(n_estimators=n_estimators, random_state=seed)
    clf.fit(X_id_model)

    rows = []
    for name in ood_datasets:
        X_ood = _feature_matrix(feature_sets[name], name)
        X_eval = np.vstack([X_id, X_ood])
        if scaler is not None:
            X_eval = scaler.transform(X_eval)
        y = np.concatenate([np.zeros(X_id.shape[0]), np.ones(X_ood.shape[0])])
        scores = -clf.decision_function(X_eval)
        rows.append(
            {
                "dataset": name,
                "family": _family_name(feature_sets[name], name),
                "auroc": float(roc_auc_score(y, scores)),
                "n_id": X_id.shape[0],
                "n_ood": X_ood.shape[0],
                "fit_n_id": train_size,
                "seed": int(seed),
                "score_direction": "higher_is_more_ood",
            }
        )
    rows.append(
        {
            "dataset": "average",
            "family": "average",
            "auroc": float(np.mean([row["auroc"] for row in rows])),
            "n_id": X_id.shape[0],
            "n_ood": int(sum(row["n_ood"] for row in rows)),
            "fit_n_id": train_size,
            "seed": int(seed),
            "score_direction": "higher_is_more_ood",
        }
    )
    return pd.DataFrame(rows)


def clustering_accuracy(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_init: int,
    seed: int,
) -> float:
    """Compute best-mapped KMeans clustering accuracy."""
    labels = KMeans(n_clusters=len(np.unique(y)), n_init=n_init, random_state=seed).fit_predict(X)
    classes = np.unique(y)
    confusion = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for true, pred in zip(y, labels):
        confusion[int(true), int(pred)] += 1
    row_ind, col_ind = linear_sum_assignment(confusion.max() - confusion)
    mapping = {col: row for row, col in zip(row_ind, col_ind)}
    mapped = np.asarray([mapping[label] for label in labels])
    return accuracy_score(y, mapped)


def _ood_matrix_and_labels(
    feature_sets: dict[str, dict[str, Any]],
    *,
    ood_datasets: list[str],
) -> tuple[np.ndarray, np.ndarray, LabelEncoder, dict[str, int]]:
    """Stack OOD features and encode OOD-family labels."""
    if len(ood_datasets) < 2:
        raise ValueError("At least two OOD datasets are required for multi-family evaluation.")
    X_parts = [_feature_matrix(feature_sets[name], name) for name in ood_datasets]
    families = np.concatenate(
        [
            np.repeat(_family_name(feature_sets[name], name), X_parts[idx].shape[0])
            for idx, name in enumerate(ood_datasets)
        ]
    )
    encoder = LabelEncoder()
    y = encoder.fit_transform(families)
    if len(encoder.classes_) < 2:
        raise ValueError("OOD datasets must contain at least two distinct family names.")
    counts = {str(family): int((families == family).sum()) for family in encoder.classes_}
    return np.vstack(X_parts), y, encoder, counts


def multi_ood_clustering(
    feature_sets: dict[str, dict[str, Any]],
    *,
    ood_datasets: list[str],
    n_init: int,
    seeds: list[int] | tuple[int, ...] | int,
    scale_features: bool = True,
) -> pd.DataFrame:
    """Cluster only OOD feature vectors and summarize accuracy over seeds."""
    resolved_ood = [_resolve_dataset_name(feature_sets, name) for name in ood_datasets]
    X, y, encoder, _ = _ood_matrix_and_labels(feature_sets, ood_datasets=resolved_ood)
    X, _ = _maybe_scale(X, scale_features)

    rows = []
    accuracies: list[float] = []
    for seed in _configured_seeds(seeds):
        accuracy = float(clustering_accuracy(X, y, n_init=n_init, seed=seed))
        accuracies.append(accuracy)
        rows.append(
            {
                "row_type": "run",
                "seed": seed,
                "accuracy": accuracy,
                "accuracy_mean": np.nan,
                "accuracy_std": np.nan,
                "n_samples": X.shape[0],
                "n_families": len(encoder.classes_),
            }
        )
    rows.append(
        {
            "row_type": "summary",
            "seed": np.nan,
            "accuracy": np.nan,
            "accuracy_mean": float(np.mean(accuracies)),
            "accuracy_std": _safe_std(accuracies),
            "n_samples": X.shape[0],
            "n_families": len(encoder.classes_),
        }
    )
    return pd.DataFrame(rows)


def supervised_ood_family_accuracy(
    feature_sets: dict[str, dict[str, Any]],
    *,
    ood_datasets: list[str],
    hidden_layer_sizes: tuple[int, ...],
    test_size: float,
    max_iter: int,
    seeds: list[int] | tuple[int, ...] | int,
    scale_features: bool = True,
    early_stopping: bool = False,
    learning_rate_init: float = 0.001,
) -> pd.DataFrame:
    """Train/test a small sklearn MLP to classify OOD families from DISC features."""
    resolved_ood = [_resolve_dataset_name(feature_sets, name) for name in ood_datasets]
    X, y, encoder, _ = _ood_matrix_and_labels(feature_sets, ood_datasets=resolved_ood)
    class_counts = np.bincount(y)
    if np.any(class_counts < 2):
        raise ValueError("Each OOD family needs at least two samples for stratified splitting.")
    n_test = int(np.ceil(float(test_size) * y.shape[0])) if test_size < 1 else int(test_size)
    n_train = int(y.shape[0] - n_test)
    if n_test < len(encoder.classes_) or n_train < len(encoder.classes_):
        raise ValueError("The supervised stratified split is too small for the number of OOD families.")

    rows = []
    accuracies: list[float] = []
    for seed in _configured_seeds(seeds):
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            stratify=y,
            random_state=seed,
        )
        mlp = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            random_state=seed,
            early_stopping=early_stopping,
            learning_rate_init=learning_rate_init,
        )
        model = make_pipeline(StandardScaler(), mlp) if scale_features else mlp
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        accuracy = float(accuracy_score(y_test, preds))
        accuracies.append(accuracy)
        rows.append(
            {
                "row_type": "run",
                "seed": seed,
                "accuracy": accuracy,
                "accuracy_mean": np.nan,
                "accuracy_std": np.nan,
                "n_train": X_train.shape[0],
                "n_test": X_test.shape[0],
                "n_families": len(encoder.classes_),
            }
        )
    rows.append(
        {
            "row_type": "summary",
            "seed": np.nan,
            "accuracy": np.nan,
            "accuracy_mean": float(np.mean(accuracies)),
            "accuracy_std": _safe_std(accuracies),
            "n_train": np.nan,
            "n_test": np.nan,
            "n_families": len(encoder.classes_),
        }
    )
    return pd.DataFrame(rows)


def evaluation_summary(
    *,
    feature_sets: dict[str, dict[str, Any]],
    id_dataset: str,
    ood_datasets: list[str],
    binary_auroc: pd.DataFrame,
    clustering_accuracy_df: pd.DataFrame,
    supervised_accuracy_df: pd.DataFrame,
) -> dict[str, Any]:
    """Create a compact JSON-serializable summary of all DISC evaluations."""
    resolved_id, resolved_ood = resolve_dataset_names(
        feature_sets,
        id_dataset=id_dataset,
        ood_datasets=ood_datasets,
    )
    binary_rows = binary_auroc[binary_auroc["dataset"] != "average"]
    binary_average = binary_auroc.loc[binary_auroc["dataset"] == "average", "auroc"].iloc[0]
    clustering_summary = clustering_accuracy_df[clustering_accuracy_df["row_type"] == "summary"].iloc[0]
    supervised_summary = supervised_accuracy_df[supervised_accuracy_df["row_type"] == "summary"].iloc[0]

    return {
        "datasets": {
            "id_dataset": resolved_id,
            "ood_datasets": resolved_ood,
            "n_id": int(_feature_matrix(feature_sets[resolved_id], resolved_id).shape[0]),
            "n_ood_by_dataset": {
                name: int(_feature_matrix(feature_sets[name], name).shape[0])
                for name in resolved_ood
            },
            "ood_families": {
                name: _family_name(feature_sets[name], name)
                for name in resolved_ood
            },
        },
        "binary_auroc": {
            "average": float(binary_average),
            "score_direction": "higher_is_more_ood",
            "per_family": [
                {
                    "dataset": str(row["dataset"]),
                    "family": str(row["family"]),
                    "auroc": float(row["auroc"]),
                    "n_ood": int(row["n_ood"]),
                }
                for _, row in binary_rows.iterrows()
            ],
        },
        "clustering_accuracy": {
            "mean": float(clustering_summary["accuracy_mean"]),
            "std": float(clustering_summary["accuracy_std"]),
            "n_repeats": int((clustering_accuracy_df["row_type"] == "run").sum()),
        },
        "supervised_accuracy": {
            "mean": float(supervised_summary["accuracy_mean"]),
            "std": float(supervised_summary["accuracy_std"]),
            "n_repeats": int((supervised_accuracy_df["row_type"] == "run").sum()),
        },
    }


def run_disc_evaluations(
    feature_sets: dict[str, dict[str, Any]],
    eval_cfg: dict[str, Any],
    *,
    default_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run the complete DISC-only evaluation protocol from a config section."""
    id_dataset = str(eval_cfg["id_dataset"])
    ood_datasets = list(eval_cfg["ood_datasets"])
    resolved_id, resolved_ood = resolve_dataset_names(
        feature_sets,
        id_dataset=id_dataset,
        ood_datasets=ood_datasets,
    )
    scale_features = bool(eval_cfg.get("scale_features", True))
    seeds = _configured_seeds(eval_cfg.get("seeds", default_seed))
    iforest_cfg = dict(eval_cfg.get("iforest", {}))
    kmeans_cfg = dict(eval_cfg.get("kmeans", {}))
    supervised_cfg = dict(eval_cfg.get("supervised", {}))

    binary_df = binary_iforest_aurocs(
        feature_sets,
        id_dataset=resolved_id,
        ood_datasets=resolved_ood,
        train_fraction=float(iforest_cfg.get("train_fraction", 1.0)),
        n_estimators=int(iforest_cfg.get("n_estimators", 200)),
        seed=int(iforest_cfg.get("seed", seeds[0])),
        scale_features=scale_features,
    )
    clustering_df = multi_ood_clustering(
        feature_sets,
        ood_datasets=resolved_ood,
        n_init=int(kmeans_cfg.get("n_init", 20)),
        seeds=seeds,
        scale_features=scale_features,
    )
    supervised_df = supervised_ood_family_accuracy(
        feature_sets,
        ood_datasets=resolved_ood,
        hidden_layer_sizes=tuple(supervised_cfg.get("hidden_layer_sizes", [128, 64])),
        test_size=float(supervised_cfg.get("test_size", 0.2)),
        max_iter=int(supervised_cfg.get("max_iter", 500)),
        seeds=seeds,
        scale_features=scale_features,
        early_stopping=bool(supervised_cfg.get("early_stopping", False)),
        learning_rate_init=float(supervised_cfg.get("learning_rate_init", 0.001)),
    )
    summary = evaluation_summary(
        feature_sets=feature_sets,
        id_dataset=resolved_id,
        ood_datasets=resolved_ood,
        binary_auroc=binary_df,
        clustering_accuracy_df=clustering_df,
        supervised_accuracy_df=supervised_df,
    )
    return binary_df, clustering_df, supervised_df, summary
