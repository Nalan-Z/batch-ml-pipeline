# Batch ML Pipeline — Nutrient Deficiency Classification & Auto-Approval

A Python pipeline for automated processing of micronutrient assay batch data.  
Applies machine learning classifiers and 4-parameter logistic regression (4PLR) curve fitting to predict analyte P3 values, classify deficiencies and borderlines, and filter cases eligible for auto-approval.

---

## Overview

```
Raw instrument Excel
        │
        ▼
 Machine_learning.py       ← LDA + GaussianNB prediction, 4PLR curve fitting
        │
        ▼
 auto_approve.py           ← quality filter, repeat/missing case removal
        │
        ▼
 _auto_approve.xlsx        ← upload-ready approved case list
```

`batch_stats.py` is a standalone utility for quick descriptive statistics on any P1 file.

---

## Features

- Extracts raw assay data directly from instrument-generated Excel workbooks
- Z-score normalisation with per-row mean imputation for missing analytes
- Ensemble ML prediction: LDA and GaussianNB each run 10 iterations; results averaged
- Per-analyte 4PLR curve fitting with optional z-score-based weighting
- Deficiency and borderline classification using configurable bin thresholds (35 analytes)
- Uncertainty flagging: analytes > mean + 2 SD marked per case
- Plate swap detection between adjacent cases
- Auto-approval filter: validation score, plate count, age range, deficiency count, plate ratio checks
- All output paths configurable via a single `config.yaml` — no hardcoded paths in code

---

## Requirements

```
python >= 3.8
pandas
numpy
scipy
scikit-learn
matplotlib
openpyxl
pyyaml
```

Install dependencies:

```bash
pip install pandas numpy scipy scikit-learn matplotlib openpyxl pyyaml
```

---

## Setup

1. Clone the repository.
2. Copy `config.yaml` and edit the paths to match your environment:

```yaml
paths:
  working_dir:      "."
  results_dir:      "./Results"
  repeats_dir:      "./Repeats"
  log_dir:          "./Log_files/4PLR_log_files"
  historical_data:  "./Run_files/historical_normalized_zonly_labels.csv"

files:
  batch_template:   "batch_template.xlsx"
  auto_approve_log: "Auto_approve_log.csv"
```

3. Place your historical training CSV and analyte dictionary files under `Run_files/`:

```
Run_files/
  historical_normalized_zonly_labels.csv   ← training data (not included, see below)
  predict_dict.py
  learn_dict.py
```

> **Note:** The historical training dataset is not included in this repository as it contains private assay records. The pipeline will not run without it. You will need to provide your own labelled dataset in the same normalised z-score format.

---

## Usage

### Step 1 — ML prediction and 4PLR fitting

```bash
python Machine_learning.py
```

A file picker dialog will open. Select the instrument Excel file for the batch.

Outputs written to `working_dir` (then moved to `results_dir`):

| File | Description |
|---|---|
| `<batch>_processed_summary.xlsx` | Summary, analyte results, batch stats, AA input |
| `<batch>_filtered_P3_results.xlsx` | Upload-ready P3 values |
| `<batch>_processed_uncertainty_plot.png` | Reference count vs uncertainty scatter |
| `Log_files/…/*.csv` | Appended log entries (4PLR params, batch stats, deficiencies) |

### Step 2 — Auto-approval filter

```bash
python auto_approve.py <batch_number>
```

Reads the `_processed_summary.xlsx` from Step 1 and the corresponding repeat list from `repeats_dir`.

Output:

| File | Description |
|---|---|
| `<batch>_auto_approve.xlsx` | Cases sheet + P3 Values sheet ready for upload |

### Utility — Batch statistics

```bash
python batch_stats.py <batch>_P1.xlsx
```

Writes `<batch>_batch_stats.xlsx` with min/max/mean/std/count per analyte.

---

## Repository Structure

```
├── Machine_learning.py       # Main pipeline
├── auto_approve.py           # Auto-approval filter
├── batch_stats.py            # Standalone batch stats utility
├── config.yaml               # All environment paths (edit before running)
├── Run_files/
│   ├── predict_dict.py       # Analyte feature columns for prediction
│   └── learn_dict.py         # Analyte feature columns for training
├── Results/                  # Auto-created; intermediate files moved here
├── Repeats/                  # Repeat case Excel files (one per batch)
└── Log_files/
    └── 4PLR_log_files/       # Appended CSV logs per run
```

---

## Notes on the analyte classification bins

Deficiency and borderline thresholds in `ANALYTE_BINS` are set for a specific assay system and patient population. If you apply this pipeline to a different assay platform, these bins will need to be recalibrated against your own reference ranges.

The two composite indices (`Immunidex`, `Spectrox`) are computed upstream by the assay software and treated here as ordinary numeric analytes.

---

## What is not included

| Item | Reason |
|---|---|
| Historical training CSV | Contains private patient assay records |
| Instrument Excel examples | Proprietary file format with internal identifiers |
| `Repeats/` files | Lab-internal QC records |
| `batch_template.xlsx` | Internal reporting template |

If you want to adapt this pipeline to your own data, you will need to provide all of the above in the formats described in Setup.

---

## Author

Nalan-Z

## Acknowledgements
This pipeline was developed by refactoring and extending an existing 
internal batch processing script. Core algorithm logic and analyte 
thresholds are inherited from the original implementation.