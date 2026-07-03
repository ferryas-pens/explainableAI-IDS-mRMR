#!/usr/bin/env python3
"""
02_load_explain_models.py

Tahap 02:
- Load model, scaler, dan feature_names.
- Load test set prepared.
- Jalankan SHAP untuk penjelasan global/lokal.
- Jalankan LIME untuk penjelasan lokal.
- Simpan CSV dan plot.

Default:
- model-name Random_Forest, tetapi dapat diganti best_model atau model lain.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_artifacts(prepared_dir: Path, model_dir: Path, dataset: str, model_name: str):
    pdir = prepared_dir / dataset
    mdir = model_dir / dataset

    X_train = joblib.load(pdir / "X_train_raw.pkl")
    X_test = joblib.load(pdir / "X_test_raw.pkl")
    y_test = joblib.load(pdir / "y_test.pkl")
    le = joblib.load(pdir / "label_encoder.pkl")

    feature_names = joblib.load(mdir / "feature_names.pkl")
    scaler = joblib.load(mdir / "scaler.pkl")

    model_file = "best_model.pkl" if model_name == "best_model" else f"{model_name}.pkl"
    model = joblib.load(mdir / model_file)

    X_train_scaled = pd.DataFrame(
        scaler.transform(X_train[feature_names]),
        columns=feature_names,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test[feature_names]),
        columns=feature_names,
        index=X_test.index,
    )

    return model, X_train_scaled, X_test_scaled, y_test, le, feature_names


def normalize_shap_values(shap_values):
    values = shap_values
    if isinstance(values, list):
        if len(values) == 2:
            values = values[1]
        else:
            values = np.abs(np.stack(values, axis=-1)).mean(axis=-1)

    values = np.asarray(values)

    # SHAP baru dapat mengembalikan shape [n, features, classes]
    if values.ndim == 3:
        if values.shape[-1] == 2:
            values = values[:, :, 1]
        else:
            values = np.abs(values).mean(axis=-1)

    return values


def run_shap(model, X_test: pd.DataFrame, out_dir: Path, max_rows: int):
    try:
        import shap
    except Exception as exc:
        print(f"[WARN] SHAP tidak tersedia: {exc}")
        return None

    X_sample = X_test.iloc[: min(max_rows, len(X_test))].copy()
    print(f"[02] Running SHAP on {len(X_sample)} rows")

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
    except Exception:
        # Fallback untuk model non-tree
        background = shap.sample(X_sample, min(100, len(X_sample)), random_state=42)
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(X_sample, nsamples=100)

    values = normalize_shap_values(shap_values)

    global_importance = pd.Series(
        np.abs(values).mean(axis=0),
        index=X_sample.columns,
        name="mean_abs_shap",
    ).sort_values(ascending=False)

    global_importance.to_csv(out_dir / "shap_global_importance.csv")

    # Bar plot sederhana agar tetap robust.
    plt.figure(figsize=(10, 7))
    global_importance.head(20).sort_values().plot(kind="barh")
    plt.title("SHAP Global Importance")
    plt.tight_layout()
    plt.savefig(out_dir / "shap_global_importance.png", dpi=160)
    plt.close()

    # Local explanation CSV untuk beberapa instance awal.
    local = pd.DataFrame(values[: min(10, len(values))], columns=X_sample.columns)
    local.to_csv(out_dir / "shap_local_values_first10.csv", index=False)

    try:
        shap.summary_plot(values, X_sample, show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(out_dir / "shap_summary_beeswarm.png", dpi=160)
        plt.close()
    except Exception as exc:
        print(f"[WARN] SHAP summary plot gagal: {exc}")

    return global_importance


def run_lime(model, X_train: pd.DataFrame, X_test: pd.DataFrame, class_names: List[str], out_dir: Path, n_instances: int):
    try:
        import lime.lime_tabular
    except Exception as exc:
        print(f"[WARN] LIME tidak tersedia: {exc}")
        return None

    print(f"[02] Running LIME for {n_instances} instances")

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=list(X_train.columns),
        class_names=class_names,
        mode="classification",
        discretize_continuous=True,
        random_state=42,
    )

    records = []
    for i in range(min(n_instances, len(X_test))):
        exp = explainer.explain_instance(
            X_test.iloc[i].values,
            model.predict_proba,
            num_features=min(15, X_test.shape[1]),
        )
        pred_proba = model.predict_proba(X_test.iloc[[i]])[0]
        pred = int(np.argmax(pred_proba))

        for feature_rule, weight in exp.as_list():
            records.append({
                "instance": i,
                "predicted_class_index": pred,
                "confidence": float(pred_proba[pred]),
                "feature_rule": feature_rule,
                "lime_weight": float(weight),
            })

        try:
            fig = exp.as_pyplot_figure()
            fig.tight_layout()
            fig.savefig(out_dir / f"lime_instance_{i}.png", dpi=160)
            plt.close(fig)
        except Exception as exc:
            print(f"[WARN] Plot LIME instance {i} gagal: {exc}")

    lime_df = pd.DataFrame(records)
    lime_df.to_csv(out_dir / "lime_local_explanations.csv", index=False)

    if not lime_df.empty:
        agg = (
            lime_df.assign(abs_weight=lime_df["lime_weight"].abs())
            .groupby("feature_rule")["abs_weight"]
            .mean()
            .sort_values(ascending=False)
        )
        agg.to_csv(out_dir / "lime_aggregate_importance.csv")

        plt.figure(figsize=(10, 7))
        agg.head(20).sort_values().plot(kind="barh")
        plt.title("LIME Aggregate Importance")
        plt.tight_layout()
        plt.savefig(out_dir / "lime_aggregate_importance.png", dpi=160)
        plt.close()

    return lime_df


def compare_shap_lime(out_dir: Path):
    shap_path = out_dir / "shap_global_importance.csv"
    lime_path = out_dir / "lime_aggregate_importance.csv"

    if not shap_path.exists() or not lime_path.exists():
        return

    shap_imp = pd.read_csv(shap_path, index_col=0).iloc[:, 0]
    lime_imp = pd.read_csv(lime_path, index_col=0).iloc[:, 0]

    # LIME feature_rule sering berbentuk interval; comparison ini hanya pendekatan textual.
    rows = []
    for f, v in shap_imp.items():
        lime_match = lime_imp[lime_imp.index.to_series().str.contains(str(f), regex=False)]
        rows.append({
            "feature": f,
            "shap_importance": float(v),
            "lime_importance_matched": float(lime_match.mean()) if not lime_match.empty else 0.0,
        })

    comp = pd.DataFrame(rows)
    if comp["shap_importance"].max() > 0:
        comp["shap_norm"] = comp["shap_importance"] / comp["shap_importance"].max()
    if comp["lime_importance_matched"].max() > 0:
        comp["lime_norm"] = comp["lime_importance_matched"] / comp["lime_importance_matched"].max()
    comp.to_csv(out_dir / "shap_vs_lime_comparison.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-dir", default="prepared_data")
    parser.add_argument("--model-dir", default="model_outputs")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model-name", default="Random_Forest")
    parser.add_argument("--output-dir", default="xai_outputs")
    parser.add_argument("--shap-max-rows", type=int, default=500)
    parser.add_argument("--lime-instances", type=int, default=5)
    args = parser.parse_args()

    out_dir = Path(args.output_dir) / args.dataset / args.model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    model, X_train, X_test, y_test, le, feature_names = load_artifacts(
        prepared_dir=Path(args.prepared_dir),
        model_dir=Path(args.model_dir),
        dataset=args.dataset,
        model_name=args.model_name,
    )

    class_names = list(le.classes_)

    with open(out_dir / "xai_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "dataset": args.dataset,
            "model_name": args.model_name,
            "class_names": class_names,
            "n_features": len(feature_names),
            "feature_names": feature_names,
        }, f, indent=2)

    run_shap(model, X_test, out_dir, args.shap_max_rows)
    run_lime(model, X_train, X_test, class_names, out_dir, args.lime_instances)
    compare_shap_lime(out_dir)

    print(f"[02] Saved XAI outputs: {out_dir}")


if __name__ == "__main__":
    main()
