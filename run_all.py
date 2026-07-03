#!/usr/bin/env python3
"""
run_all.py

Menjalankan pipeline 00 -> 04 secara berurutan.

Contoh:
python run_all.py --config config_example.json --reference-dataset CIC17
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(cmd):
    print("\n[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_example.json")
    parser.add_argument("--prepared-dir", default="prepared_data")
    parser.add_argument("--model-dir", default="model_outputs")
    parser.add_argument("--xai-dir", default="xai_outputs")
    parser.add_argument("--cross-dir", default="cross_dataset_results")
    parser.add_argument("--interpret-dir", default="cross_dataset_interpretation")
    parser.add_argument("--reference-dataset", default=None)
    parser.add_argument("--model-name", default="Random_Forest")
    parser.add_argument("--feature-selection", choices=["rf_importance", "mrmr"], default="rf_importance")
    parser.add_argument("--top-features", type=int, default=20)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    reference = args.reference_dataset or cfg.get("reference_dataset") or next(iter(cfg["datasets"]))

    run([
        sys.executable, "00_prepare_datasets.py",
        "--config", args.config,
        "--output-dir", args.prepared_dir,
        "--reference-dataset", reference,
        "--binary-label",
    ])

    run([
        sys.executable, "01_train_save_models.py",
        "--prepared-dir", args.prepared_dir,
        "--output-dir", args.model_dir,
        "--feature-selection", args.feature_selection,
        "--top-features", str(args.top_features),
    ])

    run([
        sys.executable, "02_load_explain_models.py",
        "--prepared-dir", args.prepared_dir,
        "--model-dir", args.model_dir,
        "--dataset", reference,
        "--model-name", args.model_name,
        "--output-dir", args.xai_dir,
    ])

    run([
        sys.executable, "03_cross_dataset_analysis.py",
        "--prepared-dir", args.prepared_dir,
        "--model-dir", args.model_dir,
        "--reference-dataset", reference,
        "--model-name", args.model_name,
        "--output-dir", args.cross_dir,
    ])

    run([
        sys.executable, "04_interpret_cross_dataset.py",
        "--cross-dir", args.cross_dir,
        "--output-dir", args.interpret_dir,
    ])

    print("\n[DONE] Pipeline selesai.")


if __name__ == "__main__":
    main()
