#!/usr/bin/env python3
"""
03_cross_dataset_analysis.py

Tahap 03:
- Ambil model reference, scaler, dan feature_names.
- Uji ke dataset target yang sudah disiapkan oleh Script 00.
- Simpan metrik cross-dataset.
- Opsional: hitung SHAP global per target untuk stability analysis.

Output:
cross_dataset_results/
  cross_dataset_metrics.csv
  reports/<reference>_to_<target>.txt
  shap_stability_report.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    classification_report,
    confusion_matrix,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_reference_artifacts(model_dir: Path, reference_dataset: str, model_name: str):
    mdir = model_dir / reference_dataset
    feature_names = joblib.load(mdir / "feature_names.pkl")
    scaler = joblib.load(mdir / "scaler.pkl")
    model_file = "best_model.pkl" if model_name == "best_model" else f"{model_name}.pkl"
    model = joblib.load(mdir / model_file)
    return model, scaler, feature_names


def load_target_dataset(prepared_dir: Path, dataset: str):
    d = prepared_dir / dataset
    X_test = joblib.load(d / "X_test_raw.pkl")
    y_test = joblib.load(d / "y_test.pkl")
    le = joblib.load(d / "label_encoder.pkl")
    return X_test, y_test, le


def align_features(X: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
    aligned = X.copy()
    for f in feature_names:
        if f not in aligned.columns:
            aligned[f] = 0.0
    return aligned[feature_names]


def eval_predictions(y_true, y_pred, class_names):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "report": classification_report(y_true, y_pred, target_names=class_names, zero_division=0),
    }


def shap_global_for_target(model, X_scaled: pd.DataFrame, max_rows: int):
    try:
        import shap
    except Exception:
        return None

    X_sample = X_scaled.iloc[: min(max_rows, len(X_scaled))]
    try:
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(X_sample)
    except Exception:
        return None

    if isinstance(values, list):
        values = values[1] if len(values) == 2 else np.abs(np.stack(values, axis=-1)).mean(axis=-1)
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, 1] if values.shape[-1] == 2 else np.abs(values).mean(axis=-1)

    return pd.Series(np.abs(values).mean(axis=0), index=X_sample.columns).sort_values(ascending=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-dir", default="prepared_data")
    parser.add_argument("--model-dir", default="model_outputs")
    parser.add_argument("--reference-dataset", required=True)
    parser.add_argument("--model-name", default="Random_Forest")
    parser.add_argument("--output-dir", default="cross_dataset_results")
    parser.add_argument("--targets", nargs="*", default=None)
    parser.add_argument("--shap-max-rows", type=int, default=300)
    args = parser.parse_args()

    prepared_dir = Path(args.prepared_dir)
    model_dir = Path(args.model_dir)
    out_dir = Path(args.output_dir)
    report_dir = out_dir / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    model, scaler, feature_names = load_reference_artifacts(
        model_dir,
        args.reference_dataset,
        args.model_name,
    )

    dataset_names = sorted([p.name for p in prepared_dir.iterdir() if p.is_dir()])
    if args.targets:
        dataset_names = [d for d in dataset_names if d in set(args.targets)]

    rows = []
    shap_rows = []

    for target in dataset_names:
        print(f"[03] Evaluating {args.reference_dataset} -> {target}")

        X_test, y_test, le = load_target_dataset(prepared_dir, target)
        class_names = list(le.classes_)

        X_aligned = align_features(X_test, feature_names)
        X_scaled = pd.DataFrame(
            scaler.transform(X_aligned),
            columns=feature_names,
            index=X_aligned.index,
        )

        y_pred = model.predict(X_scaled)
        metrics = eval_predictions(y_test, y_pred, class_names)

        row = {
            "reference_dataset": args.reference_dataset,
            "target_dataset": target,
            "model_name": args.model_name,
            "accuracy": metrics["accuracy"],
            "f1_weighted": metrics["f1_weighted"],
            "f1_macro": metrics["f1_macro"],
            "mcc": metrics["mcc"],
            "n_test": len(y_test),
        }
        rows.append(row)

        with open(report_dir / f"{args.reference_dataset}_to_{target}.txt", "w", encoding="utf-8") as f:
            f.write(metrics["report"])
            f.write("\n\nConfusion matrix:\n")
            f.write(str(metrics["confusion_matrix"]))

        shap_imp = shap_global_for_target(model, X_scaled, args.shap_max_rows)
        if shap_imp is not None:
            for rank, (feature, value) in enumerate(shap_imp.items(), start=1):
                shap_rows.append({
                    "reference_dataset": args.reference_dataset,
                    "target_dataset": target,
                    "feature": feature,
                    "rank": rank,
                    "mean_abs_shap": float(value),
                })

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(out_dir / "cross_dataset_metrics.csv", index=False)

    if shap_rows:
        shap_df = pd.DataFrame(shap_rows)
        shap_df.to_csv(out_dir / "shap_stability_report.csv", index=False)

    # Plot generalization gap sederhana.
    if not metrics_df.empty:
        plt.figure(figsize=(12, 6))
        plot_df = metrics_df.set_index("target_dataset")[["f1_weighted", "f1_macro", "mcc"]]
        plot_df.plot(kind="bar")
        plt.title(f"Cross-dataset metrics: {args.reference_dataset} -> targets")
        plt.ylabel("score")
        plt.ylim(-0.05, 1.05)
        plt.tight_layout()
        plt.savefig(out_dir / "generalization_gap.png", dpi=160)
        plt.close()

    print(f"[03] Saved cross-dataset results: {out_dir}")


if __name__ == "__main__":
    main()
