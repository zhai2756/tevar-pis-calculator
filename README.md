# TEVAR-PIS Risk Calculator

Code, locked model parameters, aggregate results, and a browser-based calculator accompanying:

**Development and External Validation of a Penalized Logistic Regression Model to Predict Post-Implantation Syndrome After Thoracic Endovascular Aortic Repair in Patients With Type B Aortic Dissection.**

**Live calculator:** https://zhai2756.github.io/tevar-pis-calculator/

## Overview

The repository provides:

- **Locked-model verification:** discrimination, calibration, threshold classification, and risk-stratum results for the three study cohorts.
- **Feature-selection analyses:** LASSO path generation, random-forest/XGBoost permutation importance, configuration files, and run instructions.
- **Performance uncertainty:** Table 4 point estimates and 95% confidence intervals for discrimination, calibration, and classification metrics.
- **Fully nested sensitivity analysis:** code and aggregate summaries for Supplementary Tables S21–S23.
- **Figure 2 reconstruction:** code, frozen feature-level inputs, the voting table, and the figure-generation script.

The public release contains code and aggregate results. Patient-level datasets are not included.

## Repository contents

| Path | Content |
|---|---|
| `index.html` | Standalone browser-based calculator with three-tier risk strata and a hypothetical research-monitoring framework (no drug, ICU/step-down, or other treatment recommendations) |
| `model/ridge_model_coefficients.json` | Coefficients, intercept, hyperparameters, and locked performance summary |
| `model/preprocessing_params.json` | Clinical bounds, RobustScaler parameters, TBAD encoding, and squared-term definition |
| `model/ridge_final_model.pkl` | Frozen scikit-learn model |
| `code/verify_locked_model.py` | Locked-model verification using the processed cohort CSVs |
| `code/table4_performance_ci.py` | Table 4 performance estimates and 95% confidence intervals |
| `code/bootstrap_or_ci.py` | Table 5 odds-ratio bootstrap refitting |
| `results/locked_model_verification.json` | Aggregate locked-model verification report |
| `results/table4_metric_cis.csv` and `.json` | Table 4 point estimates, 95% confidence intervals, methods, and resample counts |
| `results/table4_ci_metadata.json` | Table 4 analysis settings and aggregate point estimates |
| `results/table5_or_bootstrap_ci.json` | Table 5 point estimates and confidence intervals from 2,000 bootstrap refits |
| `original_pipeline/` | Feature-selection scripts, configuration, software requirements, and run instructions |
| `fully_nested_l2/code/` | Preprocessing, feature selection, VIF filtering, inner tuning, outer evaluation, and patient-cluster bootstrap |
| `fully_nested_l2/results/` | Nested-analysis summaries, tuning results, selection frequencies, bootstrap AUC distribution, and leakage audit |
| `figure2_rebuild/` | Figure 2 code and frozen aggregate inputs |
| `LICENSE` | MIT license |

## Locked model

- L2-penalized logistic regression; `C = 0.6672367170464207`, solver `lbfgs`, random seed 42.
- Predictors: age, operation time, contrast volume, TBAD stratum, preoperative neutrophil count, preoperative APTT, preoperative fibrinogen, squared neutrophil count, and preoperative platelet count.
- Coefficients apply after clinical-bound clipping and RobustScaler transformation. The squared neutrophil term is calculated after clipping and then standardized. TBAD stratum is ordinal-encoded as 1 → 0, 2 → 1, 3 → 2 and is not scaled.
- Reported model-selection-derived threshold: 0.321 (stored exact value: 0.32094380290459346). In the model-selection set, this gives 14 TP, 6 FP, 106 TN, and 10 FN; sensitivity 58.3%, specificity 94.6%, PPV 70.0%, and NPV 91.4%.
- Risk strata: low < 15%; intermediate 15%–30%, inclusive; high > 30%. External-cohort observed PIS rates were 8.9% (12/135), 10.7% (3/28), and 54.3% (19/35), respectively.

Open `index.html` in a modern browser or use the live calculator. Calculations are performed locally in the browser.

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

## Reproduce Table 4 performance intervals

With the three processed cohort CSVs in one directory, run:

```bash
python code/table4_performance_ci.py \
  --data-dir /path/to/stage1_cohorts \
  --output-dir results
```

AUC-PR, Brier score, balanced accuracy, and F1 intervals use 1,000 stratified patient-level bootstrap resamples with seed 42. PPV and NPV use Wilson score intervals. The included CSV and JSON files contain aggregate results only.

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

## Feature-selection analyses

`original_pipeline/` contains the LASSO and permutation-importance scripts, the 53-feature order, model settings, and software requirements. Supply the processed training CSV with `pis` and the feature columns listed in `original_pipeline/config.json`, then run:

```bash
python -m pip install -r original_pipeline/requirements.txt
python original_pipeline/code/lasso_cv.py --train-csv /path/to/preprocessed_train.csv
python original_pipeline/code/permutation_importance.py --train-csv /path/to/preprocessed_train.csv
```

The LASSO script reproduces the locked 40-point cross-validated path and compares it with `figure2_rebuild/lasso_path.npz`. The permutation-importance script reports both the apparent analysis and the separately labelled cross-validated sensitivity analysis. See `original_pipeline/README.md` for configuration details and the scope of the XGBoost aggregate.

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

The manuscript analyses used Python 3.10. The verification environment used Python 3.12.16 with numpy 2.4.4, pandas 3.0.2, scipy 1.18.1, scikit-learn 1.9.0, and joblib 1.5.3. Exact requirements for the feature-selection scripts, including XGBoost 3.3.0, are listed in `original_pipeline/requirements.txt`; general plotting and verification dependencies are listed in `requirements.txt`. The calculator requires only a modern browser.

## Data availability

Patient-level datasets are available from the corresponding author upon reasonable request, subject to institutional data-governance requirements.

## Version and citation

**Current release:** v2.2.

**Version DOI:** https://doi.org/10.5281/zenodo.24331526.

Cite the software using the v2.2 version-specific DOI above.

## Use

For research and educational use. Predictions are probabilistic estimates and do not replace clinical judgment.
