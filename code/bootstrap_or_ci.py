"""Reproduce Table 5 odds-ratio confidence intervals for the locked ridge model.

The script refits the frozen nine-term L2-penalized logistic regression to
stratified patient-level bootstrap resamples of the processed training set.
Percentile intervals are calculated on the coefficient (log-odds) scale and
exponentiated to the odds-ratio scale.

Example:
    python code/bootstrap_or_ci.py \
        --train-csv analysis_data/preprocessed_train.csv \
        --features-json analysis_data/selected_features.json \
        --model-json model/ridge_model_coefficients.json \
        --output results/table5_or_bootstrap_ci.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--features-json", type=Path, required=True)
    parser.add_argument("--model-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train = pd.read_csv(args.train_csv, index_col=0)
    features = json.loads(args.features_json.read_text(encoding="utf-8"))["final_features"]
    frozen = json.loads(args.model_json.read_text(encoding="utf-8"))
    c_value = float(frozen["hyperparameters"]["C"])

    x = train[features].to_numpy(dtype=float)
    y = train["pis"].to_numpy(dtype=int)
    model = LogisticRegression(
        C=c_value,
        penalty="l2",
        solver="lbfgs",
        max_iter=1000,
        random_state=args.seed,
    )
    model.fit(x, y)

    expected = np.array([frozen["coefficients"][name] for name in features])
    point = model.coef_[0]
    max_difference = float(np.max(np.abs(point - expected)))
    if max_difference > 1e-5:
        raise RuntimeError(
            f"The refitted coefficients do not reproduce the frozen model; "
            f"maximum absolute difference = {max_difference:.6g}"
        )

    rng = np.random.RandomState(args.seed)
    positive_indices = np.flatnonzero(y == 1)
    negative_indices = np.flatnonzero(y == 0)
    bootstrap_coefficients = []
    for _ in range(args.n_bootstrap):
        sampled = np.concatenate([
            rng.choice(positive_indices, len(positive_indices), replace=True),
            rng.choice(negative_indices, len(negative_indices), replace=True),
        ])
        rng.shuffle(sampled)
        bootstrap_model = LogisticRegression(
            C=c_value,
            penalty="l2",
            solver="lbfgs",
            max_iter=1000,
            random_state=args.seed,
        )
        bootstrap_model.fit(x[sampled], y[sampled])
        bootstrap_coefficients.append(bootstrap_model.coef_[0])

    bootstrap_coefficients = np.asarray(bootstrap_coefficients)
    lower, upper = np.percentile(bootstrap_coefficients, [2.5, 97.5], axis=0)
    results = {
        "method": (
            "Stratified patient-level bootstrap refitting of the locked nine-term "
            "L2-penalized logistic-regression model; percentile 95% CIs calculated "
            "on the coefficient (log-odds) scale and exponentiated."
        ),
        "n_bootstrap": args.n_bootstrap,
        "random_seed": args.seed,
        "C": c_value,
        "n_training": int(len(train)),
        "n_events": int(y.sum()),
        "max_abs_coefficient_difference_from_frozen_model": max_difference,
        "features": {},
    }
    for index, name in enumerate(features):
        results["features"][name] = {
            "coefficient": float(point[index]),
            "or": float(np.exp(point[index])),
            "or_ci_low": float(np.exp(lower[index])),
            "or_ci_high": float(np.exp(upper[index])),
            "coefficient_ci_low": float(lower[index]),
            "coefficient_ci_high": float(upper[index]),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Frozen-model maximum coefficient difference: {max_difference:.3g}")
    print(f"Successful bootstrap refits: {len(bootstrap_coefficients)}")
    for name in features:
        row = results["features"][name]
        print(
            f"{name}: OR {row['or']:.3f} "
            f"(95% CI {row['or_ci_low']:.3f}–{row['or_ci_high']:.3f})"
        )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
