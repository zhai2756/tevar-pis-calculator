# TEVAR-PIS Risk Calculator

Code, locked model parameters, aggregate results, and a browser-based calculator accompanying:

**Development and External Validation of a Penalized Logistic Regression Model to Predict Post-Implantation Syndrome After Thoracic Endovascular Aortic Repair in Patients With Type B Aortic Dissection.**

## Overview

The repository includes:

- **Locked-model verification:** discrimination, calibration, threshold classification, and risk-stratum results for the three study cohorts.
- **Fully nested sensitivity analysis:** code and aggregate summaries for Supplementary Tables S21–S23.
- **Figure 2 reconstruction:** code, frozen feature-level inputs, the voting table, and the rebuilt figure.

The public release contains the code and aggregate artifacts for these analyses. Patient-level datasets and the historical raw-data feature-selection pipeline are not included.

## Repository contents

| Path | Content |
|---|---|
| `index.html` | Standalone browser-based calculator |
| `model/ridge_model_coefficients.json` | Coefficients, intercept, hyperparameters, and locked performance summary |
| `model/preprocessing_params.json` | Clinical bounds, RobustScaler parameters, TBAD encoding, and squared-term definition |
| `model/ridge_final_model.pkl` | Frozen scikit-learn model |
| `code/verify_locked_model.py` | Locked-model verification using the processed cohort CSVs |
| `code/bootstrap_or_ci.py` | Table 5 odds-ratio bootstrap refitting |
| `results/locked_model_verification.json` | Aggregate locked-model verification report |
| `results/table5_or_bootstrap_ci.json` | Table 5 point estimates and confidence intervals from 2,000 bootstrap refits |
| `fully_nested_l2/code/` | Preprocessing, feature selection, VIF filtering, inner tuning, outer evaluation, and patient-cluster bootstrap |
| `fully_nested_l2/results/` | Nested-analysis summaries, tuning results, selection frequencies, bootstrap AUC distribution, and leakage audit |
| `figure2_rebuild/` | Figure 2 code and frozen aggregate inputs |
| `LICENSE` | MIT license |

## Locked model

- L2-penalized logistic regression; `C = 0.6672367170464207`, solver `lbfgs`, random seed 42.
- Predictors: age, operation time, contrast volume, TBAD stratum, preoperative neutrophil count, preoperative APTT, preoperative fibrinogen, squared neutrophil count, and preoperative platelet count.
- Coefficients apply after clinical-bound clipping and RobustScaler transformation. The squared neutrophil term is calculated after clipping and then standardized. TBAD stratum is ordinal-encoded as 1 → 0, 2 → 1, 3 → 2 and is not scaled.
- Reported threshold: 0.321 (stored exact value: 0.32094380290459346). In the model-selection set, this gives 14 TP, 6 FP, 106 TN, and 10 FN; sensitivity 58.3%, specificity 94.6%, PPV 70.0%, and NPV 91.4%.
- Risk strata: low < 15%; intermediate 15%–30%, inclusive; high > 30%. External-cohort observed PIS rates were 8.9% (12/135), 10.7% (3/28), and 54.3% (19/35), respectively.

Open `index.html` in a modern browser. Calculations are performed locally in the browser.

## Verify the locked model

Place the processed cohort CSVs in a local directory:

- `stage1_train.csv`
- `stage1_internal_val_fixed.csv`
- `stage1_ext_val_v3.csv`

Each file contains `pis` and the nine processed model columns. Run from the repository root:

```bash
python code/verify_locked_model.py \
  --data-dir /path/to/stage1_cohorts \
  --output results/locked_model_verification.json
```

The script scores the frozen JSON coefficients directly and checks the aggregate values against the locked results.

| Cohort | n | Events | AUC | Brier |
|---|---:|---:|---:|---:|
| Training | 540 | 97 | 0.8386 | 0.1039 |
| Model selection | 136 | 24 | 0.7839 | 0.0972 |
| External | 198 | 34 | 0.8006 | 0.0970 |

## Reproduce Table 5 odds-ratio intervals

The included JSON contains the completed 2,000 stratified patient-level bootstrap refits. To rerun the analysis, supply the processed training CSV and a `selected_features.json` file with a `final_features` array containing the nine model columns:

```bash
python code/bootstrap_or_ci.py \
  --train-csv /path/to/preprocessed_train.csv \
  --features-json /path/to/selected_features.json \
  --model-json model/ridge_model_coefficients.json \
  --output results/table5_or_bootstrap_ci.json
```

The script first verifies that a full-training refit reproduces the frozen coefficients.

## Fully nested sensitivity analysis (S21–S23)

The design uses 50 outer splits (stratified 10-fold cross-validation with 5 repeats), inner 5-fold tuning, and 1,000 patient-cluster bootstrap resamples. The mean repeat-level pooled out-of-fold AUC is 0.770389, with 95% CI 0.713004–0.822847. The median number of selected features per outer fold was 18 (range 10–24), and the median pairwise Jaccard similarity was 0.375.

The included summaries can be inspected directly. A full rerun uses the de-identified input at `fully_nested_l2/data/raw_training_540.csv`:

```bash
python fully_nested_l2/code/run_fully_nested_l2.py --workers 4
python fully_nested_l2/code/finalize_results.py
```

`finalize_results.py` uses the 50 fold-result JSON files and the raw-training CSV generated during the governed analysis.

## Rebuild Figure 2

```bash
python figure2_rebuild/generate_figure2.py
```

See `figure2_rebuild/README.md` for input definitions and provenance. The script reconstructs the figure from frozen aggregate artifacts.

## Requirements

Python 3.10 or later; see `requirements.txt`. The calculator requires only a modern browser. Locked-result and nested-summary verification used numpy 2.4.4, pandas 3.0.2, scipy 1.18.1, and scikit-learn 1.9.0.

## Data availability

Patient-level datasets are available from the corresponding author upon reasonable request, subject to institutional data-governance requirements.

## Archived versions

Current release:

> zhai2756. (2026). zhai2756/tevar-pis-calculator: v1.1.0: Revised manuscript files (v1.1.0). Zenodo. https://doi.org/10.5281/zenodo.24304506

Earlier release: v1.0.0, https://doi.org/10.5281/zenodo.21466817. The concept DOI for all versions is https://doi.org/10.5281/zenodo.21466816.

## Use

For research and educational use. Predictions are probabilistic estimates and do not replace clinical judgment.
