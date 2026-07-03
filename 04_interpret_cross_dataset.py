#!/usr/bin/env python3
"""
04_interpret_cross_dataset.py

Tahap 04:
- Membaca hasil cross_dataset_metrics.csv.
- Membaca shap_stability_report.csv jika tersedia.
- Membuat interpretasi agregat:
  - best/worst target
  - gap weighted-F1 vs MCC
  - stabilitas ranking SHAP
  - catatan risiko generalisasi.

Output:
cross_dataset_interpretation/
  interpretation_report.md
  metric_summary.csv
  shap_top_features.csv
  plots/*.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def write_metric_plots(metrics: pd.DataFrame, out_dir: Path):
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    if metrics.empty:
        return

    metrics_sorted = metrics.sort_values("mcc", ascending=False)

    plt.figure(figsize=(12, 6))
    metrics_sorted.set_index("target_dataset")[["f1_weighted", "f1_macro", "mcc"]].plot(kind="bar")
    plt.title("Cross-dataset metric comparison")
    plt.ylabel("score")
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(plot_dir / "cross_dataset_metric_comparison.png", dpi=160)
    plt.close()

    if "accuracy" in metrics.columns:
        gap = metrics.copy()
        gap["weighted_f1_minus_mcc"] = gap["f1_weighted"] - gap["mcc"]
        plt.figure(figsize=(12, 6))
        gap.sort_values("weighted_f1_minus_mcc", ascending=False).set_index("target_dataset")["weighted_f1_minus_mcc"].plot(kind="bar")
        plt.title("Weighted F1 - MCC gap")
        plt.ylabel("gap")
        plt.tight_layout()
        plt.savefig(plot_dir / "weighted_f1_mcc_gap.png", dpi=160)
        plt.close()


def summarize_shap_stability(cross_dir: Path, out_dir: Path):
    path = cross_dir / "shap_stability_report.csv"
    if not path.exists():
        return None, "SHAP stability file tidak ditemukan."

    shap_df = pd.read_csv(path)
    if shap_df.empty:
        return None, "SHAP stability file kosong."

    top = (
        shap_df[shap_df["rank"] <= 10]
        .groupby("feature")
        .agg(
            appear_in_top10=("target_dataset", "nunique"),
            mean_rank=("rank", "mean"),
            mean_abs_shap=("mean_abs_shap", "mean"),
        )
        .sort_values(["appear_in_top10", "mean_abs_shap"], ascending=[False, False])
    )
    top.to_csv(out_dir / "shap_top_features.csv")

    note = (
        "Fitur dengan frekuensi tinggi pada Top-10 SHAP cenderung stabil secara global. "
        "Namun, stabilitas fitur tidak otomatis membuktikan generalisasi model jika MCC tetap rendah."
    )
    return top, note


def build_report(metrics: pd.DataFrame, shap_summary, shap_note: str) -> str:
    lines = []
    lines.append("# Cross-dataset Interpretation Report\n")

    if metrics.empty:
        lines.append("Tidak ada metrik cross-dataset yang dapat dibaca.\n")
        return "\n".join(lines)

    lines.append("## Metric summary\n")
    summary = metrics[["accuracy", "f1_weighted", "f1_macro", "mcc"]].describe().T
    lines.append(summary.to_markdown())
    lines.append("\n")

    best = metrics.sort_values("mcc", ascending=False).iloc[0]
    worst = metrics.sort_values("mcc", ascending=True).iloc[0]

    lines.append("## Best/Worst target by MCC\n")
    lines.append(f"- Best target: `{best['target_dataset']}` with MCC = {best['mcc']:.4f}")
    lines.append(f"- Worst target: `{worst['target_dataset']}` with MCC = {worst['mcc']:.4f}\n")

    metrics = metrics.copy()
    metrics["weighted_f1_minus_mcc"] = metrics["f1_weighted"] - metrics["mcc"]
    suspicious = metrics.sort_values("weighted_f1_minus_mcc", ascending=False).head(5)

    lines.append("## Warning: weighted-F1 vs MCC gap\n")
    lines.append(
        "Jika weighted-F1 tinggi tetapi MCC rendah, model kemungkinan condong ke kelas mayoritas. "
        "Ini umum pada IDS yang sangat imbalance."
    )
    lines.append(suspicious[["target_dataset", "f1_weighted", "f1_macro", "mcc", "weighted_f1_minus_mcc"]].to_markdown(index=False))
    lines.append("\n")

    lines.append("## SHAP stability\n")
    lines.append(shap_note)
    if shap_summary is not None:
        lines.append("\nTop SHAP-stable features:\n")
        lines.append(shap_summary.head(15).to_markdown())
    lines.append("\n")

    lines.append("## Methodological interpretation\n")
    lines.append(
        "- Feature selection, scaler, dan SMOTE harus fit pada training set saja.\n"
        "- Cross-dataset evaluation wajib untuk melihat robustness terhadap distribution shift.\n"
        "- Accuracy dan weighted-F1 tidak cukup untuk IDS imbalance; MCC dan recall per kelas harus dilaporkan.\n"
        "- SHAP/LIME menjelaskan perilaku model, bukan sebab kausal serangan.\n"
        "- Jika MCC cross-dataset mendekati nol, model belum layak diklaim generalizable walaupun within-dataset sangat tinggi.\n"
    )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cross-dir", default="cross_dataset_results")
    parser.add_argument("--output-dir", default="cross_dataset_interpretation")
    args = parser.parse_args()

    cross_dir = Path(args.cross_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = cross_dir / "cross_dataset_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Tidak ditemukan: {metrics_path}")

    metrics = pd.read_csv(metrics_path)
    metrics.to_csv(out_dir / "metric_summary.csv", index=False)

    write_metric_plots(metrics, out_dir)

    shap_summary, shap_note = summarize_shap_stability(cross_dir, out_dir)
    report = build_report(metrics, shap_summary, shap_note)

    with open(out_dir / "interpretation_report.md", "w", encoding="utf-8") as f:
        f.write(report)

    print(f"[04] Saved interpretation: {out_dir / 'interpretation_report.md'}")


if __name__ == "__main__":
    main()
