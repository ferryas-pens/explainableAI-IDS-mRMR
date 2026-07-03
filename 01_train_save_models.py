#!/usr/bin/env python3
"""
01_train_save_models.py

Tahap 01:
- Load prepared_data/<dataset>/.
- Feature selection pada TRAIN ONLY.
- Default: Random Forest feature_importances_ seperti repo.
- Opsi tambahan: mRMR.
- Fit scaler pada TRAIN ONLY.
- SMOTE hanya pada TRAIN.
- Train beberapa model dan simpan artifact.

Output:
model_outputs/<dataset>/
  feature_names.pkl
  scaler.pkl
  Random_Forest.pkl
  best_model.pkl
  model_comparison.csv
  feature_ranking.csv
  classification_report_<model>.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    classification_report,
    confusion_matrix,
)

try:
    from imblearn.over_sampling import SMOTE
except Exception:
    SMOTE = None


def load_prepared_dataset(path: Path):
    X_train = joblib.load(path / "X_train_raw.pkl")
    X_test = joblib.load(path / "X_test_raw.pkl")
    y_train = joblib.load(path / "y_train.pkl")
    y_test = joblib.load(path / "y_test.pkl")
    le = joblib.load(path / "label_encoder.pkl")
    return X_train, X_test, y_train, y_test, le


def rf_importance_selection(X_train: pd.DataFrame, y_train: pd.Series, top_n: int, random_state: int):
    selector = RandomForestClassifier(
        n_estimators=50,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    selector.fit(X_train, y_train)
    ranking = pd.Series(
        selector.feature_importances_,
        index=X_train.columns,
        name="rf_importance",
    ).sort_values(ascending=False)
    selected = ranking.head(top_n).index.tolist()
    return selected, ranking


def mrmr_selection(X_train: pd.DataFrame, y_train: pd.Series, top_n: int, random_state: int):
    from sklearn.feature_selection import mutual_info_classif

    relevance = pd.Series(
        mutual_info_classif(X_train, y_train, random_state=random_state),
        index=X_train.columns,
        name="mutual_information",
    )
    corr = X_train.corr(numeric_only=True).abs().fillna(0.0)

    selected: List[str] = []
    candidates = list(X_train.columns)

    while len(selected) < min(top_n, len(candidates)):
        scores = {}
        for f in candidates:
            redundancy = 0.0 if not selected else corr.loc[f, selected].mean()
            scores[f] = relevance[f] - redundancy
        best = max(scores, key=scores.get)
        selected.append(best)
        candidates.remove(best)

    ranking = pd.Series(
        {
            f: relevance[f] - (0.0 if not selected else corr.loc[f, selected].mean())
            for f in X_train.columns
        },
        name="mrmr_score",
    ).sort_values(ascending=False)

    return selected, ranking


def build_models(random_state: int) -> Dict[str, object]:
    models = {
        "Decision_Tree": DecisionTreeClassifier(
            max_depth=15,
            random_state=random_state,
            class_weight="balanced",
        ),
        "Random_Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced_subsample",
        ),
        "Gradient_Boosting": GradientBoostingClassifier(random_state=random_state),
        "Logistic_Regression": LogisticRegression(
            max_iter=1000,
            n_jobs=None,
            class_weight="balanced",
        ),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Gaussian_NB": GaussianNB(),
    }

    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        )
    except Exception:
        pass

    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            random_state=random_state,
            n_jobs=-1,
        )
    except Exception:
        pass

    return models


def maybe_smote(X, y, random_state: int):
    if SMOTE is None:
        print("[WARN] imbalanced-learn tidak tersedia. SMOTE dilewati.")
        return X, y
    if pd.Series(y).value_counts().min() < 2:
        print("[WARN] Kelas minoritas terlalu kecil untuk SMOTE. SMOTE dilewati.")
        return X, y
    k_neighbors = min(5, int(pd.Series(y).value_counts().min()) - 1)
    if k_neighbors < 1:
        return X, y
    smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    return smote.fit_resample(X, y)


def evaluate_model(model, X_test, y_test, class_names: List[str]) -> Dict:
    y_pred = model.predict(X_test)
    result = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "report": classification_report(
            y_test,
            y_pred,
            target_names=class_names,
            zero_division=0,
        ),
    }
    return result


def train_one_dataset(
    dataset_dir: Path,
    out_dir: Path,
    top_features: int,
    feature_selection: str,
    random_state: int,
):
    dataset_name = dataset_dir.name
    print(f"[01] Training dataset: {dataset_name}")

    X_train, X_test, y_train, y_test, le = load_prepared_dataset(dataset_dir)
    class_names = list(le.classes_)

    if feature_selection == "mrmr":
        selected, ranking = mrmr_selection(X_train, y_train, top_features, random_state)
    else:
        selected, ranking = rf_importance_selection(X_train, y_train, top_features, random_state)

    scaler = StandardScaler()
    X_train_sel = pd.DataFrame(
        scaler.fit_transform(X_train[selected]),
        columns=selected,
        index=X_train.index,
    )
    X_test_sel = pd.DataFrame(
        scaler.transform(X_test[selected]),
        columns=selected,
        index=X_test.index,
    )

    X_res, y_res = maybe_smote(X_train_sel, y_train, random_state)

    out = out_dir / dataset_name
    out.mkdir(parents=True, exist_ok=True)

    joblib.dump(selected, out / "feature_names.pkl")
    joblib.dump(scaler, out / "scaler.pkl")
    ranking.to_csv(out / "feature_ranking.csv")

    rows = []
    best_name = None
    best_score = -np.inf
    best_model = None

    for name, model in build_models(random_state).items():
        print(f"[01]  - Training {name}")
        try:
            model.fit(X_res, y_res)
            metrics = evaluate_model(model, X_test_sel, y_test, class_names)
            joblib.dump(model, out / f"{name}.pkl")

            with open(out / f"classification_report_{name}.txt", "w", encoding="utf-8") as f:
                f.write(metrics["report"])

            row = {
                "dataset": dataset_name,
                "model": name,
                "accuracy": metrics["accuracy"],
                "f1_weighted": metrics["f1_weighted"],
                "f1_macro": metrics["f1_macro"],
                "mcc": metrics["mcc"],
                "feature_selection": feature_selection,
                "n_features": len(selected),
            }
            rows.append(row)

            # Prioritaskan MCC, lalu F1 macro.
            score = metrics["mcc"] + 0.01 * metrics["f1_macro"]
            if score > best_score:
                best_score = score
                best_name = name
                best_model = model

        except Exception as exc:
            print(f"[WARN] Model {name} gagal: {exc}")

    comparison = pd.DataFrame(rows).sort_values(
        by=["mcc", "f1_macro", "f1_weighted"],
        ascending=False,
    )
    comparison.to_csv(out / "model_comparison.csv", index=False)

    if best_model is not None:
        joblib.dump(best_model, out / "best_model.pkl")
        with open(out / "best_model.json", "w", encoding="utf-8") as f:
            json.dump({"best_model": best_name, "selection_metric": "mcc_then_macro_f1"}, f, indent=2)

    print(f"[01] Saved model outputs: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-dir", default="prepared_data")
    parser.add_argument("--output-dir", default="model_outputs")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--top-features", type=int, default=20)
    parser.add_argument("--feature-selection", choices=["rf_importance", "mrmr"], default="rf_importance")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    prepared_dir = Path(args.prepared_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_dirs = [p for p in prepared_dir.iterdir() if p.is_dir()]
    if args.datasets:
        keep = set(args.datasets)
        dataset_dirs = [p for p in dataset_dirs if p.name in keep]

    if not dataset_dirs:
        raise RuntimeError("Tidak ada dataset prepared yang ditemukan.")

    for d in sorted(dataset_dirs):
        train_one_dataset(
            dataset_dir=d,
            out_dir=out_dir,
            top_features=args.top_features,
            feature_selection=args.feature_selection,
            random_state=args.random_state,
        )

    print("[01] Done.")


if __name__ == "__main__":
    main()
