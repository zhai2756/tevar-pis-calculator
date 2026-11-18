# -*- coding: utf-8 -*-
"""Compute Table 4 point estimates and newly added 95% confidence intervals.

Previously reported intervals are retained unchanged. Usage from the repository root:
    python code/table4_performance_ci.py --data-dir /path/to/governed/processed/data

The data directory must contain stage1_train.csv,
stage1_internal_val_fixed.csv, and stage1_ext_val_v3.csv. These governed
patient-level files are intentionally not included in the public archive.
The locked model, feature order, thresholds, and point estimates are unchanged.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)

SEED = 42
N_BOOT = 1000
FEATURES = [
    "num__age", "num__operation_time_min", "num__contrast_ml",
    "cat_ordinal__tbad_stratum", "num__neut_pre", "num__aptt_pre",
    "num__fib_pre", "num__neut_pre_sq", "num__plt_pre",
]
FILES = {
    "train": "stage1_train.csv",
    "internal": "stage1_internal_val_fixed.csv",
    "external": "stage1_ext_val_v3.csv",
}
THRESHOLDS = {
    "train": 0.321,
    # Exact model-selection-set Youden cutoff; displayed as 0.321 in the manuscript.
    "internal": 0.32094380290459346,
    "external": 0.321,
}
COHORT_LABELS = {
    "train": "Training",
    "internal": "Model-selection",
    "external": "External validation",
}
BOOTSTRAP_METRICS = [
    "AUC-ROC", "AUC-PR", "Brier score", "Sensitivity", "Specificity",
    "Balanced accuracy", "F1 score",
]
BOOTSTRAP_INTERVAL_METHOD = (
    f"Percentile 95% CI; {N_BOOT} stratified patient-level bootstrap "
    f"resamples; seed {SEED}"
)
# Preserve the manuscript's existing intervals exactly; only missing CIs are added.
PREVIOUSLY_REPORTED_INTERVALS = {
    ("train", "AUC-ROC"): (0.791, 0.879),
    ("internal", "AUC-ROC"): (0.684, 0.884),
    ("external", "AUC-ROC"): (0.725, 0.877),
    ("train", "Sensitivity"): (0.443, 0.649),
    ("internal", "Sensitivity"): (0.368, 0.798),
    ("external", "Sensitivity"): (0.395, 0.711),
    ("train", "Specificity"): (0.910, 0.957),
    ("internal", "Specificity"): (0.884, 1.000),
    ("external", "Specificity"): (0.869, 0.953),
    ("internal", "Brier score"): (0.076, 0.118),
}


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return (float("nan"), float("nan"))
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (float(center - half), float(center + half))


def metric_values(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return {
        "AUC-ROC": float(roc_auc_score(y, p)),
        "AUC-PR": float(average_precision_score(y, p)),
        "Brier score": float(brier_score_loss(y, p)),
        "Sensitivity": float(tp / (tp + fn)),
        "Specificity": float(tn / (tn + fp)),
        "PPV": float(tp / (tp + fp)),
        "NPV": float(tn / (tn + fn)),
        "Balanced accuracy": float(balanced_accuracy_score(y, pred)),
        "F1 score": float(f1_score(y, pred, zero_division=0)),
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing the three governed processed CSVs")
    parser.add_argument("--output-dir", type=Path, default=Path("results"), help="Output directory (default: results)")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    with (repo / "model" / "ridge_final_model.pkl").open("rb") as handle:
        model = pickle.load(handle)

    rng = np.random.default_rng(SEED)
    rows: list[dict[str, object]] = []
    point_estimates: dict[str, dict[str, float]] = {}
    for cohort in ("train", "internal", "external"):
        source = args.data_dir / FILES[cohort]
        if not source.is_file():
            raise FileNotFoundError(f"Required governed input not found: {source}")
        df = pd.read_csv(source)
        y = df["pis"].astype(int).to_numpy()
        p = model.predict_proba(df[FEATURES].astype(float).to_numpy())[:, 1]
        threshold = THRESHOLDS[cohort]
        points = metric_values(y, p, threshold)
        point_estimates[cohort] = points

        positive = np.flatnonzero(y == 1)
        negative = np.flatnonzero(y == 0)
        boot = {metric: [] for metric in BOOTSTRAP_METRICS}
        for _ in range(N_BOOT):
            idx = np.concatenate([
                rng.choice(positive, size=len(positive), replace=True),
                rng.choice(negative, size=len(negative), replace=True),
            ])
            values = metric_values(y[idx], p[idx], threshold)
            for metric in BOOTSTRAP_METRICS:
                boot[metric].append(values[metric])

        for metric in BOOTSTRAP_METRICS:
            low, high = np.percentile(boot[metric], [2.5, 97.5])
            method = BOOTSTRAP_INTERVAL_METHOD
            if (cohort, metric) in PREVIOUSLY_REPORTED_INTERVALS:
                low, high = PREVIOUSLY_REPORTED_INTERVALS[(cohort, metric)]
                method = (
                    "Previously reported 95% CI retained unchanged; newly added "
                    f"Table 4 intervals use: {BOOTSTRAP_INTERVAL_METHOD}"
                )
            rows.append({
                "cohort": cohort, "cohort_label": COHORT_LABELS[cohort],
                "n": len(y), "events": int(y.sum()), "threshold": threshold,
                "metric": metric, "point_estimate": points[metric],
                "ci95_low": float(low), "ci95_high": float(high),
                "method": method,
                "valid_bootstrap_resamples": len(boot[metric]),
            })
        for metric, successes, total in (
            ("PPV", int(points["TP"]), int(points["TP"] + points["FP"])),
            ("NPV", int(points["TN"]), int(points["TN"] + points["FN"])),
        ):
            low, high = wilson_interval(successes, total)
            rows.append({
                "cohort": cohort, "cohort_label": COHORT_LABELS[cohort],
                "n": len(y), "events": int(y.sum()), "threshold": threshold,
                "metric": metric, "point_estimate": points[metric],
                "ci95_low": low, "ci95_high": high,
                "method": "Wilson score 95% CI", "valid_bootstrap_resamples": "not applicable",
            })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(args.output_dir / "table4_metric_cis.csv", index=False)
    result.to_json(args.output_dir / "table4_metric_cis.json", orient="records", indent=2)
    metadata = {
        "model": "Locked ridge logistic regression",
        "model_file": "model/ridge_final_model.pkl",
        "features": FEATURES,
        "thresholds": THRESHOLDS,
        "n_bootstrap": N_BOOT,
        "seed": SEED,
        "bootstrap_design": "Separate resampling with replacement within PIS-positive and PIS-negative patients; original class counts retained",
        "interval": "Percentile 2.5th and 97.5th",
        "ppv_npv_interval": "Wilson score 95% CI",
        "retained_interval_policy": "Previously reported Table 4 intervals are retained unchanged; only missing intervals were newly calculated",
        "previously_reported_intervals": {
            f"{cohort}|{metric}": list(interval)
            for (cohort, metric), interval in PREVIOUSLY_REPORTED_INTERVALS.items()
        },
        "point_estimates": point_estimates,
    }
    (args.output_dir / "table4_ci_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(result[["cohort_label", "metric", "point_estimate", "ci95_low", "ci95_high"]].to_string(index=False))


if __name__ == "__main__":
    main()
