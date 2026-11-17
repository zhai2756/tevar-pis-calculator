"""Reproduce the locked LASSO path with stratified 10-fold AUC selection."""
from __future__ import annotations

import argparse
import json
import platform
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

CODE_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = CODE_DIR.parent
REPOSITORY_ROOT = PIPELINE_DIR.parent

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", message="Inconsistent values: penalty=l1", category=UserWarning)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True,
                        help="Processed training CSV containing pis and the 53 model features")
    parser.add_argument("--config", type=Path, default=PIPELINE_DIR / "config.json")
    parser.add_argument("--locked-npz", type=Path,
                        default=REPOSITORY_ROOT / "figure2_rebuild" / "lasso_path.npz")
    parser.add_argument("--output-dir", type=Path, default=PIPELINE_DIR / "results")
    parser.add_argument("--jobs", type=int, default=6,
                        help="Number of C-grid evaluations to run concurrently (default: 6)")
    return parser.parse_args()


def load_training_data(path: Path, config: dict) -> tuple[np.ndarray, np.ndarray, list[str]]:
    frame = pd.read_csv(path)
    outcome = config["outcome"]
    if outcome not in frame:
        raise ValueError(f"Missing outcome column: {outcome}")
    y = frame[outcome].to_numpy(dtype=int)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("The outcome must contain both binary classes 0 and 1")
    features = list(config["feature_order"])
    missing = [name for name in features if name not in frame]
    if missing:
        raise ValueError(f"Missing processed features: {missing}")
    X = frame.loc[:, features].to_numpy(dtype=float)
    if not np.isfinite(X).all():
        raise ValueError("The processed feature matrix contains missing or infinite values")
    return X, y, features


def main() -> None:
    args = arguments()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    X, y, features = load_training_data(args.train_csv, config)
    settings = config["lasso"]
    seed = int(config["random_state"])
    grid = settings["c_grid"]
    Cs = np.logspace(grid["start_exponent"], grid["stop_exponent"], grid["count"])
    n_splits = int(settings["cv_folds"])
    folds = list(StratifiedKFold(
        n_splits=n_splits,
        shuffle=settings["cv_shuffle"],
        random_state=seed,
    ).split(X, y))

    def fit_at_c(x: np.ndarray, outcome: np.ndarray, c: float) -> LogisticRegression:
        model = LogisticRegression(
            penalty=settings["penalty"],
            solver=settings["solver"],
            C=float(c),
            max_iter=int(settings["max_iter"]),
            random_state=seed,
        )
        model.fit(x, outcome)
        return model

    def evaluate_c(c: float):
        fold_aucs = []
        for train_index, validation_index in folds:
            model = fit_at_c(X[train_index], y[train_index], c)
            probability = model.predict_proba(X[validation_index])[:, 1]
            fold_aucs.append(roc_auc_score(y[validation_index], probability))
        full_model = fit_at_c(X, y, c)
        return (
            float(np.mean(fold_aucs)),
            float(np.std(fold_aucs, ddof=1) / np.sqrt(n_splits)),
            float(roc_auc_score(y, full_model.predict_proba(X)[:, 1])),
            full_model.coef_.ravel(),
        )

    evaluations = Parallel(n_jobs=args.jobs, backend="threading")(
        delayed(evaluate_c)(float(c)) for c in Cs
    )
    cv_mean = np.asarray([value[0] for value in evaluations])
    cv_se = np.asarray([value[1] for value in evaluations])
    apparent = np.asarray([value[2] for value in evaluations])
    coefficient_path = np.asarray([value[3] for value in evaluations])

    index_min = int(np.argmax(cv_mean))
    cutoff = cv_mean[index_min] - cv_se[index_min]
    eligible = np.where(cv_mean >= cutoff)[0]
    index_1se = int(eligible[0])
    c_min, c_1se = float(Cs[index_min]), float(Cs[index_1se])
    tolerance = float(settings["coefficient_tolerance"])
    nonzero = [
        features[j]
        for j in np.where(np.abs(coefficient_path[index_1se]) > tolerance)[0]
    ]

    with np.load(args.locked_npz) as locked:
        locked_cs = locked["Cs"]
        locked_mean = locked["auc_mean"]
        locked_c_min = float(locked["C_min"])
        locked_c_1se = float(locked["C_1se"])
        maximum_deviation = float(np.max(np.abs(cv_mean - locked_mean)))
        grid_matches = bool(np.allclose(Cs, locked_cs))

    result = {
        "protocol": {
            "n_rows": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "n_events": int(y.sum()),
            "c_grid": "np.logspace(-2.5, 2, 40)",
            "cross_validation": f"StratifiedKFold(n_splits={n_splits}, shuffle=True, random_state={seed})",
            "selection_rule": settings["selection_rule"],
            "solver": f"{settings['solver']}, max_iter={settings['max_iter']}, cold-start fits",
        },
        "C_min": c_min,
        "C_1se": c_1se,
        "auc_mean_at_Cmin": float(cv_mean[index_min]),
        "auc_se_at_Cmin": float(cv_se[index_min]),
        "auc_mean_at_C1se": float(cv_mean[index_1se]),
        "nonzero_at_C1se": nonzero,
        "matches_locked_npz": bool(grid_matches and maximum_deviation < 5e-3),
        "max_dev_from_locked": maximum_deviation,
        "locked_C_min": locked_c_min,
        "locked_C_1se": locked_c_1se,
        "apparent_auc_max": float(np.max(apparent)),
        "apparent_auc_argmax_C": float(Cs[int(np.argmax(apparent))]),
        "runtime": {"python": platform.python_version(), "scikit_learn": sklearn.__version__},
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "lasso_cv_reanalysis.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    pd.DataFrame({
        "C": Cs,
        "log10C": np.log10(Cs),
        "cv_auc_mean": cv_mean,
        "cv_auc_se": cv_se,
        "apparent_auc": apparent,
        "n_nonzero": (np.abs(coefficient_path) > tolerance).sum(axis=1),
    }).to_csv(args.output_dir / "lasso_cv_curve.csv", index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
