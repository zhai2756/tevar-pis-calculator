# TEVAR-PIS Risk Calculator & Analysis Pipeline

Code, model parameters, and the web-based calculator accompanying the study:

> **Machine learning–based prediction of post-implantation syndrome (PIS) after thoracic endovascular aortic repair (TEVAR) in patients with type B aortic dissection: development and external validation of a Ridge logistic regression model.**

**Live calculator:** https://zhai2756.github.io/tevar-pis-calculator/

## Repository contents

| Path | Content |
|------|---------|
| `index.html` | Web-based PIS risk calculator (standalone, fully client-side; English UI with three-tier management recommendations) |
| `model/ridge_final_model.pkl` | Final trained Ridge logistic regression model (scikit-learn) |
| `model/ridge_model_coefficients.json` | All 9 coefficients + intercept, hyperparameters, and performance metrics |
| `model/preprocessing_params.json` | Clinical bounds (clipping), RobustScaler median/IQR per feature, TBAD stratum encoding, squared-term definition |
| `code/phase1_preprocessing.py` … `code/phase6_interpretability.py` | Analysis pipeline: preprocessing, feature selection, model building, final model, external validation, interpretability (SHAP) |
| `code/run_optimal_params.py` | Unified three-cohort evaluation of all 8 algorithms with tuned hyperparameters |
| `code/generate_full_evaluation.py` | ROC / calibration / DCA figure generation for all cohorts |
| `LICENSE` | MIT |

## Model summary

- **Algorithm**: Ridge logistic regression (L2 penalty, C = 0.667, solver = lbfgs, random_state = 42)
- **Predictors (9)**: age, operation time, contrast volume, TBAD stratum, preoperative neutrophil count, preoperative APTT, preoperative fibrinogen, neutrophil count squared, preoperative platelet count
- **Development**: Northern Jiangsu People's Hospital (training n = 540; 10-fold CV AUC 0.806; training AUC 0.839)
- **Internal validation**: n = 136, AUC 0.784 (95% CI 0.684–0.884)
- **External validation**: n = 198 (The Second Affiliated Hospital of Nanjing Medical University), AUC 0.801 (95% CI 0.725–0.877), Brier 0.097
- **Decision threshold**: 0.321 (Youden-optimal; sensitivity 58.3%, specificity 94.6%, PPV 70.0%, NPV 91.4%)
- **Risk strata**: low < 15% | intermediate 15–30% | high > 30% (observed PIS rates in the external cohort: 10.7% / 28.6% / 42.9%)

## Prediction formula

```
p = sigmoid( intercept + Σ coef_i · z_i )
z_i = (x_i − median_i) / IQR_i          # RobustScaler, after clipping x_i to clinical bounds
neut_pre_sq = (clipped neut_pre)²        # then standardized with its own median/IQR
TBAD stratum: 1 (Complicated) → 0, 2 (High-risk) → 1, 3 (Uncomplicated) → 2   # ordinal, not scaled
```

All constants are in `model/preprocessing_params.json` and `model/ridge_model_coefficients.json`.

## Data

The de-identified patient-level datasets are available from the corresponding author upon reasonable request, subject to institutional data governance requirements (not included in this repository).

## Requirements

Python ≥ 3.10 with `scikit-learn ≥ 1.2`, `pandas`, `numpy`, `xgboost ≥ 1.7`, `lightgbm ≥ 3.0`, `shap ≥ 0.40`, `matplotlib`, `seaborn`, `statsmodels`.

## Disclaimer

For research use only. Predictions are probabilistic estimates and do not replace clinical judgment.
