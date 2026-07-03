# Ordered Python Scripts for explainableAI-IDS Style Pipeline

Paket ini disusun mengikuti urutan file pada repo:

1. `00_prepare_datasets.py`
2. `01_train_save_models.py`
3. `02_load_explain_models.py`
4. `03_cross_dataset_analysis.py`
5. `04_interpret_cross_dataset.py`

Default pipeline mengikuti metode repo: feature selection menggunakan `RandomForestClassifier.feature_importances_` pada data train saja. Opsi `mrmr` disediakan sebagai eksperimen tambahan, bukan default repo.

## 0. Install dependency

```bash
pip install -r requirements.txt
```

## 1. Siapkan konfigurasi dataset

Edit `config_example.json`:

```json
{
  "datasets": {
    "CIC17": "data/CIC17",
    "CIC18_Friday": "data/CIC18/Friday"
  },
  "reference_dataset": "CIC17",
  "target_column": "Label"
}
```

Path dapat berupa file CSV tunggal atau folder berisi banyak CSV.

## 2. Jalankan berurutan

```bash
python 00_prepare_datasets.py --config config_example.json --output-dir prepared_data --binary-label
python 01_train_save_models.py --prepared-dir prepared_data --output-dir model_outputs --feature-selection rf_importance --top-features 20
python 02_load_explain_models.py --prepared-dir prepared_data --model-dir model_outputs --dataset CIC17 --model-name Random_Forest
python 03_cross_dataset_analysis.py --prepared-dir prepared_data --model-dir model_outputs --reference-dataset CIC17 --output-dir cross_dataset_results
python 04_interpret_cross_dataset.py --cross-dir cross_dataset_results --output-dir cross_dataset_interpretation
```

Atau langsung:

```bash
python run_all.py --config config_example.json
```

## Catatan penting

- Feature selection, scaler, dan SMOTE harus fit hanya pada data training.
- Jangan melakukan normalisasi atau sampling dengan informasi dari test set.
- Untuk cross-dataset, gunakan scaler dan feature list dari model reference.
- Laporkan MCC, macro-F1, recall per kelas, dan confusion matrix, bukan accuracy saja.
