# TEVAR-PIS Risk Calculator and Verified Analysis Artifacts

Code, locked model parameters, aggregate results, and the web-based calculator accompanying:

**Development and External Validation of a Penalized Logistic Regression Model to Predict Post-Implantation Syndrome After Thoracic Endovascular Aortic Repair in Patients With Type B Aortic Dissection.**

> **Release status:** the revised source has been uploaded to GitHub. A revised Zenodo version has not yet been archived; until then, the DOI below refers only to v1.0.0.

## Scope and reproducibility boundary

This package supports three distinct tasks:

1. **Locked-model verification:** aggregate discrimination, calibration, threshold classification, and risk-stratum results can be recalculated when the governed, already processed cohort CSVs are supplied.
2. **Fully nested sensitivity analysis:** the implementation and non-patient-level summaries for Supplementary Tables S21–S23 are included. During local validation, all included nested summary artifacts were regenerated from the 50 frozen outer-fold results and matched byte for byte. The complete 50-fold retraining was not rerun during final packaging.
3. **Figure 2 reconstruction:** the figure and voting table can be rebuilt from frozen feature-level aggregate artifacts.

**This is not a complete raw-data-to-final-model reproduction package.** The exact configuration used to generate the locked historical XGBoost importance table was not archived, and the original generator for the main feature-selection artifacts is therefore not claimed to be reproducible here. The fully nested sensitivity-analysis code must not be misrepresented as that missing original generator.

Legacy `phase1_preprocessing.py`, `phase2_feature_selection.py`, and `phase4_final_model.py` scripts were intentionally excluded because their implementations did not faithfully match the current manuscript methods. Other legacy all-algorithm evaluation and figure scripts were also excluded from this focused revised release.

## Repository contents

| Path | Content |
|---|---|
| `index.html` | Standalone client-side calculator; descriptive risk tiers only, with no treatment or monitoring recommendations |
| `model/ridge_model_coefficients.json` | Nine coefficients, intercept, hyperparameters, and locked performance summary |
| `model/preprocessing_params.json` | Clinical bounds, RobustScaler median/IQR values, TBAD encoding, and squared-term definition |
| `model/ridge_final_model.pkl` | Frozen scikit-learn model; load pickle files only from a trusted source |
| `code/verify_locked_model.py` | Aggregate verification on the three governed stage-1 cohort CSVs |
| `code/bootstrap_or_ci.py` | Table 5 odds-ratio bootstrap refitting using a governed processed training CSV |
| `results/locked_model_verification.json` | Aggregate locked-model verification report; no patient-level rows |
| `results/table5_or_bootstrap_ci.json` | Completed 2,000-refit Table 5 point estimates and confidence intervals |
| `fully_nested_l2/code/` | Fully nested preprocessing, four-method selection, VIF filtering, inner tuning, outer evaluation, and patient-cluster bootstrap |
| `fully_nested_l2/results/` | Non-patient-level nested-analysis summaries, tuning results, selection frequencies, bootstrap AUC distribution, and leakage audit |
| `figure2_rebuild/` | Figure 2 reconstruction code and frozen feature-level aggregate inputs |
| `LICENSE` | MIT license |

## Locked model

- L2-penalized logistic regression; `C = 0.6672367170464207`, solver `lbfgs`, random seed 42.
- Predictors: age, operation time, contrast volume, TBAD stratum, preoperative neutrophil count, preoperative APTT, preoperative fibrinogen, squared neutrophil count, and preoperative platelet count.
- Coefficients apply after clinical-bound clipping and RobustScaler transformation. The squared neutrophil term is calculated after clipping and then standardized. TBAD stratum is ordinal-encoded as 1 → 0, 2 → 1, 3 → 2 and is not scaled.
- Reported threshold: 0.321 (the stored exact model-selection-set threshold is 0.32094380290459346). In the model-selection set, this gives 14 TP, 6 FP, 106 TN, and 10 FN; sensitivity 58.3%, specificity 94.6%, PPV 70.0%, and NPV 91.4%.
- Descriptive research strata: low < 15%; intermediate 15%–30%, inclusive; high > 30%. External-cohort observed PIS rates were 8.9% (12/135), 10.7% (3/28), and 54.3% (19/35), respectively.

Open `index.html` directly in a browser. All calculation is local to the browser; no patient information is transmitted by this file.

## Verify the locked model

Obtain the three governed, already processed CSVs and place them in one directory outside this public repository:

- `stage1_train.csv`
- `stage1_internal_val_fixed.csv`
- `stage1_ext_val_v3.csv`

Each file must contain `pis` and the nine processed model columns. Then run:

```bash
python code/verify_locked_model.py \
  --data-dir /governed/path/to/stage1_cohorts \
  --output results/locked_model_verification.json
```

The script scores the frozen JSON coefficients directly, checks aggregate values against the locked results, and writes no patient-level output. The expected aggregate results are:

| Cohort | n | Events | AUC | Brier |
|---|---:|---:|---:|---:|
| Training | 540 | 97 | 0.8386 | 0.1039 |
| Model selection | 136 | 24 | 0.7839 | 0.0972 |
| External | 198 | 34 | 0.8006 | 0.0970 |

## Reproduce Table 5 odds-ratio intervals

The included JSON contains the completed 2,000 stratified patient-level bootstrap refits. To rerun them, supply the governed processed training CSV and a `selected_features.json` file with a `final_features` array containing the nine model columns:

```bash
python code/bootstrap_or_ci.py \
  --train-csv /governed/path/preprocessed_train.csv \
  --features-json /governed/path/selected_features.json \
  --model-json model/ridge_model_coefficients.json \
  --output results/table5_or_bootstrap_ci.json
```

The script first verifies that a full-training refit reproduces the frozen coefficients before starting the bootstrap.

## Fully nested sensitivity analysis (S21–S23)

The design uses 50 outer splits (stratified 10-fold cross-validation with 5 repeats), inner 5-fold tuning, and 1,000 patient-cluster bootstrap resamples. The mean repeat-level pooled out-of-fold AUC is 0.770389, with 95% CI 0.713004–0.822847. The median number of selected features per outer fold was 18 (range 10–24), the median pairwise Jaccard similarity was 0.375, and no outer fold reproduced all original nine predictors.

To rerun the analysis, place the governed de-identified input at `fully_nested_l2/data/raw_training_540.csv` and run from the repository root:

```bash
python fully_nested_l2/code/run_fully_nested_l2.py --workers 4
python fully_nested_l2/code/finalize_results.py
```

The public package intentionally omits `data/`, logs, fold-level files containing patient identifiers, and `outer_fold_predictions.csv`. `finalize_results.py` requires all 50 governed fold-result JSON files and the governed raw-training CSV; these are not included. The included summary JSON, repeat-level metrics, selection frequencies, tuning summaries, bootstrap AUC distribution, and leakage audit can be inspected without patient data.

## Rebuild Figure 2

```bash
python figure2_rebuild/generate_figure2.py
```

See `figure2_rebuild/README.md` for input definitions and the explicit XGBoost provenance limitation. This step reconstructs a figure from aggregate artifacts; it does not regenerate the original feature-selection analysis from raw data.

## Requirements

Python 3.10 or later; see `requirements.txt`. The calculator itself requires only a modern browser. Exact numerical reproduction can depend on library versions and platform. Local locked-result and nested-summary verification used Python with numpy 2.4.4, pandas 3.0.2, scipy 1.18.1, and scikit-learn 1.9.0.

## Data governance and privacy

Patient-level datasets are available only from the corresponding author upon reasonable request and subject to institutional data-governance requirements. Do not commit governed CSVs, patient identifiers, fold-level predictions, raw spreadsheets, or logs to the public repository. Keep controlled inputs outside the repository root.

## Previous archive and citation

The earlier archived release remains:

> zhai2756. (2026). zhai2756/tevar-pis-calculator: v1.0.0 — Initial release (v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.21466816

After this revised source is uploaded as a new version, update this section and the manuscript's **Archived version** entry with the new version and DOI. Do not describe these revised files as part of v1.0.0.

## Disclaimer

For research use only. Predictions are probabilistic estimates and do not replace clinical judgment. Risk tiers and observed event rates are descriptive; the calculator provides no treatment or monitoring recommendations.
