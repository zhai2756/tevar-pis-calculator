# Recovered feature-selection analyses

These scripts were recovered from the frozen revision workspace. They document the LASSO regularization path and the random-forest/XGBoost permutation-importance analyses used during model development and subsequent sensitivity checks.

## Scope

- `code/lasso_cv.py` reproduces the locked 40-point LASSO path, including `C_min = 0.05878016072274915`, `C_1se = 0.00701703828670383`, and the three nonzero terms at `C_1se`.
- `code/permutation_importance.py` computes apparent and cross-validated permutation importance. The apparent random-forest analysis uses the configuration associated with Figure 2. The XGBoost configuration is explicitly labelled as a later sensitivity-analysis configuration.
- The original end-to-end driver and the computing configuration that generated the historical 13-feature XGBoost set were not preserved. Figure 2 therefore continues to use the locked aggregate table in `figure2_rebuild/xgb_perm_importance.csv`.

`config.json` records the feature order, random seeds, model settings, cross-validation design, voting threshold, and VIF threshold.

## Input

Supply the processed training CSV used for model development. It contains one `pis` outcome column and the 53 processed feature columns listed in `config.json`; an optional index column beginning with `Unnamed` is ignored. The expected training dimensions are 540 rows and 97 PIS events.

Patient-level data are available from the corresponding author upon reasonable request, subject to institutional data-governance requirements.

## Setup

The manuscript analyses used Python 3.10. `requirements.txt` records the later package versions used for revision verification.

```bash
python -m pip install -r original_pipeline/requirements.txt
```

## LASSO path

Run from the repository root:

```bash
python original_pipeline/code/lasso_cv.py \
  --train-csv /path/to/preprocessed_train.csv
```

The script writes `lasso_cv_reanalysis.json` and `lasso_cv_curve.csv` to `original_pipeline/results/`. It also compares the regenerated path with `figure2_rebuild/lasso_path.npz`.

## Permutation importance

```bash
python original_pipeline/code/permutation_importance.py \
  --train-csv /path/to/preprocessed_train.csv
```

The script writes feature-level importance tables and `cv_permutation_importance_votes.json` to `original_pipeline/results/`. Apparent importance uses 20 permutations. The sensitivity analysis refits each model in stratified 10-fold cross-validation and computes 10 permutations on each held-out fold.

The cross-validated vote is a sensitivity analysis and differs from the locked full-training vote. The frozen model coefficients and reported performance are unchanged by these scripts.
