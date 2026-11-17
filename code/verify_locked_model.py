"""Verify aggregate results for the locked nine-predictor ridge model.

The three governed input CSVs must be the already processed stage-1 cohorts:
  stage1_train.csv
  stage1_internal_val_fixed.csv
  stage1_ext_val_v3.csv

They must contain the binary outcome ``pis`` and the nine model columns listed
in model/ridge_model_coefficients.json. Features must already be clipped,
engineered, encoded, and RobustScaler-transformed exactly as in the locked
analysis. No patient-level output is written.

Example:
    python code/verify_locked_model.py \
        --data-dir /governed/path/to/stage1_cohorts \
        --output results/locked_model_verification.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
FILES = {
    "training": "stage1_train.csv",
    "model_selection": "stage1_internal_val_fixed.csv",
    "external": "stage1_ext_val_v3.csv",
}
EXPECTED = {
    "training": {
        "n": 540,
        "events": 97,
        "auc": 0.8386,
        "brier": 0.1039,
    },
    "model_selection": {
        "n": 136,
        "events": 24,
        "auc": 0.7839,
        "brier": 0.0972,
        "threshold": 0.32094380290459346,
        "confusion": {"tp": 14, "fp": 6, "tn": 106, "fn": 10},
    },
    "external": {
        "n": 198,
        "events": 34,
        "auc": 0.8006,
        "brier": 0.0970,
        "threshold": 0.321,
        "confusion": {"tp": 19, "fp": 13, "tn": 151, "fn": 15},
        "strata": {
            "low": {"n": 135, "events": 12},
            "intermediate": {"n": 28, "events": 3},
            "high": {"n": 35, "events": 19},
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Directory containing the three governed stage-1 CSVs")
    parser.add_argument("--model-json", type=Path,
                        default=ROOT / "model/ridge_model_coefficients.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/locked_model_verification.json")
    return parser.parse_args()


def classification(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    positive = p >= threshold
    tp = int(np.sum(positive & (y == 1)))
    fp = int(np.sum(positive & (y == 0)))
    tn = int(np.sum(~positive & (y == 0)))
    fn = int(np.sum(~positive & (y == 1)))
    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "sensitivity": tp / (tp + fn),
        "specificity": tn / (tn + fp),
        "ppv": tp / (tp + fp),
        "npv": tn / (tn + fn),
    }


def risk_strata(y: np.ndarray, p: np.ndarray) -> dict:
    labels = np.where(p < 0.15, "low", np.where(p <= 0.30, "intermediate", "high"))
    result = {}
    for name in ("low", "intermediate", "high"):
        mask = labels == name
        n = int(mask.sum())
        events = int(y[mask].sum())
        result[name] = {
            "n": n,
            "events": events,
            "observed_rate": events / n if n else None,
        }
    return result


def check_close(report: dict, expected: dict, key: str, tolerance: float = 0.000051) -> bool:
    return abs(report[key] - expected[key]) <= tolerance


def main() -> None:
    args = parse_args()
    frozen = json.loads(args.model_json.read_text(encoding="utf-8"))
    features = frozen["features"]
    beta = np.array([frozen["coefficients"][name] for name in features], dtype=float)
    intercept = float(frozen["intercept"])

    report = {
        "scope": "Aggregate point-estimate verification of the locked model; no model refitting",
        "input": "Already processed stage-1 cohort CSVs supplied under data governance",
        "model_json": str(args.model_json.name),
        "cohorts": {},
    }
    failures = []

    for cohort, filename in FILES.items():
        path = args.data_dir / filename
        frame = pd.read_csv(path)
        missing = [name for name in ["pis", *features] if name not in frame.columns]
        if missing:
            raise ValueError(f"{filename} is missing required columns: {missing}")
        y = frame["pis"].to_numpy(dtype=int)
        x = frame[features].to_numpy(dtype=float)
        if not np.isfinite(x).all():
            raise ValueError(f"{filename} contains non-finite values in model features")
        p = expit(intercept + x @ beta)
        expected = EXPECTED[cohort]
        result = {
            "n": int(len(y)),
            "events": int(y.sum()),
            "auc": float(roc_auc_score(y, p)),
            "brier": float(brier_score_loss(y, p)),
            "risk_strata": risk_strata(y, p),
        }
        if "threshold" in expected:
            result["classification"] = classification(y, p, expected["threshold"])
        report["cohorts"][cohort] = result

        if result["n"] != expected["n"] or result["events"] != expected["events"]:
            failures.append(f"{cohort}: cohort size or event count mismatch")
        for metric in ("auc", "brier"):
            if not check_close(result, expected, metric):
                failures.append(
                    f"{cohort}: {metric}={result[metric]:.8f}, expected approximately {expected[metric]}"
                )
        if "confusion" in expected:
            observed = {key: result["classification"][key] for key in ("tp", "fp", "tn", "fn")}
            if observed != expected["confusion"]:
                failures.append(f"{cohort}: confusion matrix {observed} != {expected['confusion']}")
        if "strata" in expected:
            for stratum, counts in expected["strata"].items():
                observed = result["risk_strata"][stratum]
                if any(observed[key] != value for key, value in counts.items()):
                    failures.append(f"{cohort}: {stratum} stratum mismatch: {observed} != {counts}")

    report["locked_values_reproduced"] = not failures
    report["failures"] = failures
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for cohort, result in report["cohorts"].items():
        print(
            f"{cohort}: n={result['n']}, events={result['events']}, "
            f"AUC={result['auc']:.6f}, Brier={result['brier']:.6f}"
        )
        if "classification" in result:
            print("  classification:", result["classification"])
    print("LOCKED-VALUES-REPRODUCED:", "YES" if not failures else "NO")
    print(f"Aggregate report saved to: {args.output}")
    if failures:
        raise RuntimeError("Locked-model verification failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
