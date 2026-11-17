# Figure 2 reconstruction from frozen aggregate artifacts

Run from the repository root:

```bash
python figure2_rebuild/generate_figure2.py
```

The script writes `Figure2_feature_selection.png` (600 dpi),
`Figure2_feature_selection.pdf`, `vote_table.csv`, and `figure2_report.json`
into this directory. It uses only the feature-level aggregate files included
here and does not require patient-level data.

## Inputs

- `univariate_fdr.csv`: feature-level univariable p values and BH-adjusted q values.
- `lasso_path.npz` and `selection_sets.json`: frozen LASSO path, cross-validation summary, and selected sets.
- `rf_perm_importance.csv`: feature-level random-forest permutation-importance means and standard deviations.
- `xgb_perm_importance.csv`: feature-level XGBoost permutation-importance table extracted from the locked historical aggregate artifact.

The voting rule is FDR q < 0.05, nonzero LASSO coefficients at C_1se,
random-forest mean importance > 0, and XGBoost mean importance > 0.
The expected set sizes are 31, 3, 49, and 13, respectively; all nine final
predictors receive at least two votes.

## Provenance limitation

This is a **figure reconstruction script**, not the original raw-data feature
selection pipeline. The exact computing configuration that generated the
locked XGBoost importance table (including repeat/subsampling and class-weight
settings) was not archived. Consequently, that table cannot be regenerated
from raw data by this package. The included CSV preserves the locked aggregate
values so that Figure 2 and its voting table can be reconstructed without
substituting a differently configured XGBoost analysis.
