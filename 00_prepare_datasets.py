#!/usr/bin/env python3
"""
00_prepare_datasets.py

Tahap 00:
- Load dataset CSV/folder.
- Normalisasi nama kolom.
- Recover/normalisasi kolom label.
- Konversi label menjadi binary Normal/Attack jika diminta.
- Harmonisasi skema fitur mengikuti reference dataset.
- Stratified sampling.
- Train-test split.
- Simpan artifact prepared_data/<dataset>/.

Default ini mengikuti prinsip pipeline repo explainableAI-IDS:
preprocessing harus terkontrol dan bebas data leakage.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def clean_column_name(col: str) -> str:
    col = str(col).strip()
    col = re.sub(r"\s+", " ", col)
    return col


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [clean_column_name(c) for c in df.columns]
    return df.loc[:, ~df.columns.duplicated()]


def find_csv_files(path: Path) -> List[Path]:
    if path.is_file() and path.suffix.lower() == ".csv":
        return [path]
    if path.is_dir():
        return sorted(path.rglob("*.csv"))
    raise FileNotFoundError(f"Path tidak ditemukan atau bukan CSV/folder: {path}")


def load_dataset(path: str) -> pd.DataFrame:
    files = find_csv_files(Path(path))
    frames = []
    for f in files:
        try:
            part = pd.read_csv(f, low_memory=False)
            part = normalize_columns(part)
            part["__source_file"] = str(f)
            frames.append(part)
        except Exception as exc:
            print(f"[WARN] Gagal membaca {f}: {exc}")
    if not frames:
        raise RuntimeError(f"Tidak ada CSV valid pada {path}")
    return pd.concat(frames, ignore_index=True, sort=False)


def find_label_column(df: pd.DataFrame, target_col: str) -> str:
    if target_col in df.columns:
        return target_col
    lowered = {c.lower().strip(): c for c in df.columns}
    for candidate in [target_col.lower(), "label", "class", "attack", "category"]:
        if candidate in lowered:
            return lowered[candidate]
    raise KeyError(f"Kolom label tidak ditemukan. Target diminta: {target_col}")


def to_binary_label(y: pd.Series) -> pd.Series:
    def normalize_one(v: object) -> str:
        s = str(v).strip().upper()
        if s in {"BENIGN", "NORMAL", "0", "FALSE"}:
            return "Normal"
        return "Attack"
    return y.map(normalize_one)


def numeric_features(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    drop_cols = {label_col, "__source_file"}
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0.0)
    # buang kolom yang seluruhnya nol/konstan agar tidak memberi noise
    nunique = X.nunique(dropna=False)
    keep = nunique[nunique > 1].index.tolist()
    return X[keep]


def stratified_sample(
    X: pd.DataFrame,
    y: pd.Series,
    sample_frac: float,
    max_rows: int,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.Series]:
    n = len(X)
    target_n = min(max_rows, int(n * sample_frac)) if sample_frac < 1.0 else min(max_rows, n)
    if target_n <= 0 or target_n >= n:
        return X, y

    df = X.copy()
    df["__label"] = y.values
    sampled = (
        df.groupby("__label", group_keys=False)
        .apply(lambda g: g.sample(
            n=max(1, int(round(len(g) / n * target_n))),
            random_state=random_state
        ))
    )
    y_sample = sampled.pop("__label")
    return sampled.reset_index(drop=True), y_sample.reset_index(drop=True)


def align_to_reference(X: pd.DataFrame, reference_cols: List[str]) -> pd.DataFrame:
    aligned = X.copy()
    for c in reference_cols:
        if c not in aligned.columns:
            aligned[c] = 0.0
    return aligned[reference_cols]


def prepare_one_dataset(
    name: str,
    path: str,
    target_col: str,
    output_dir: Path,
    binary_label: bool,
    reference_cols: List[str] | None,
    sample_frac: float,
    max_rows: int,
    test_size: float,
    random_state: int,
) -> Tuple[List[str], Dict]:
    print(f"[00] Loading dataset {name}: {path}")
    df = load_dataset(path)
    label_col = find_label_column(df, target_col)

    y_raw = df[label_col]
    y_text = to_binary_label(y_raw) if binary_label else y_raw.astype(str).fillna("UNKNOWN")

    X = numeric_features(df, label_col)

    if reference_cols is None:
        reference_cols = list(X.columns)
    else:
        X = align_to_reference(X, reference_cols)

    X, y_text = stratified_sample(X, y_text, sample_frac, max_rows, random_state)

    le = LabelEncoder()
    y = pd.Series(le.fit_transform(y_text), name="label")

    stratify = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    out = output_dir / name
    out.mkdir(parents=True, exist_ok=True)

    joblib.dump(X_train, out / "X_train_raw.pkl")
    joblib.dump(X_test, out / "X_test_raw.pkl")
    joblib.dump(y_train, out / "y_train.pkl")
    joblib.dump(y_test, out / "y_test.pkl")
    joblib.dump(le, out / "label_encoder.pkl")
    joblib.dump(reference_cols, out / "all_numeric_cols.pkl")

    report = {
        "dataset": name,
        "input_path": path,
        "rows_after_sampling": int(len(X)),
        "n_features": int(X.shape[1]),
        "label_classes": list(le.classes_),
        "class_distribution": y_text.value_counts().to_dict(),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "binary_label": bool(binary_label),
        "reference_schema_features": len(reference_cols),
    }
    with open(out / "prep_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[00] Saved prepared dataset: {out}")
    return reference_cols, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="JSON berisi mapping dataset name -> path.")
    parser.add_argument("--output-dir", default="prepared_data")
    parser.add_argument("--reference-dataset", default=None)
    parser.add_argument("--target-col", default=None)
    parser.add_argument("--binary-label", action="store_true")
    parser.add_argument("--sample-frac", type=float, default=0.10)
    parser.add_argument("--max-rows", type=int, default=300_000)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    datasets = cfg["datasets"]
    reference_name = args.reference_dataset or cfg.get("reference_dataset") or next(iter(datasets))
    target_col = args.target_col or cfg.get("target_column", "Label")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_reports = {}
    reference_cols = None

    # Reference dataset harus diproses dulu agar schema fitur menjadi patokan.
    ordered_names = [reference_name] + [d for d in datasets if d != reference_name]

    for name in ordered_names:
        reference_cols, report = prepare_one_dataset(
            name=name,
            path=datasets[name],
            target_col=target_col,
            output_dir=output_dir,
            binary_label=args.binary_label,
            reference_cols=reference_cols,
            sample_frac=args.sample_frac,
            max_rows=args.max_rows,
            test_size=args.test_size,
            random_state=args.random_state,
        )
        all_reports[name] = report

    with open(output_dir / "prepare_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_reports, f, indent=2)

    print("[00] Done.")


if __name__ == "__main__":
    main()
