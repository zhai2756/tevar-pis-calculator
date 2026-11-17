"""Compute apparent and cross-validated RF/XGBoost permutation importance."""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

CODE_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = CODE_DIR.parent
REPOSITORY_ROOT = PIPELINE_DIR.parent
FINAL_FEATURES = [
    "num__age", "num__operation_time_min", "num__contrast_ml",
    "cat_ordinal__tbad_stratum", "num__neut_pre", "num__aptt_pre",
    "num__fib_pre", "num__neut_pre_sq", "num__plt_pre",
]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True,
                        help="Processed training CSV containing pis and the 53 model features")
    parser.add_argument("--config", type=Path, default=PIPELINE_DIR / "config.json")
    parser.add_argument("--locked-dir", type=Path,
                        default=REPOSITORY_ROOT / "figure2_rebuild")
    parser.add_argument("--output-dir", type=Path, default=PIPELINE_DIR / "results")
    return parser.parse_args()


def load_training_data(path: Path, config: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    outcome = config["outcome"]
    if outcome not in frame:
        raise ValueError(f"Missing outcome column: {outcome}")
    y = frame[outcome].to_numpy(dtype=int)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("The outcome must contain both binary classes 0 and 1")
    feature_names = list(config["feature_order"])
    missing = [name for name in feature_names if name not in frame]
    if missing:
        raise ValueError(f"Missing processed features: {missing}")
    X = frame.loc[:, feature_names].to_numpy(dtype=float)
    if not np.isfinite(X).all():
        raise ValueError("The processed feature matrix contains missing or infinite values")
    return X, y, np.asarray(feature_names)


def main() -> None:
    args = arguments()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    X, y, features = load_training_data(args.train_csv, config)
    seed = int(config["random_state"])
    importance_config = config["permutation_importance"]
    rf_params = dict(config["random_forest"])
    xgb_params = dict(config["xgboost_sensitivity"])
    makers = {
        "RF": lambda: RandomForestClassifier(**rf_params),
        "XGB": lambda: XGBClassifier(**xgb_params),
    }

    def apparent_importance(make_model):
        model = make_model().fit(X, y)
        result = permutation_importance(
            model, X, y,
            n_repeats=int(importance_config["apparent_repeats"]),
            random_state=seed,
            scoring=importance_config["scoring"],
            n_jobs=rf_params["n_jobs"],
        )
        return result.importances_mean, result.importances_std

    def cv_importance(make_model):
        folds = StratifiedKFold(
            n_splits=int(importance_config["cv_folds"]),
            shuffle=True,
            random_state=seed,
        )
        fold_means, fold_aucs = [], []
        for train_index, validation_index in folds.split(X, y):
            model = make_model().fit(X[train_index], y[train_index])
            result = permutation_importance(
                model, X[validation_index], y[validation_index],
                n_repeats=int(importance_config["cv_repeats"]),
                random_state=seed,
                scoring=importance_config["scoring"],
                n_jobs=rf_params["n_jobs"],
            )
            fold_means.append(result.importances_mean)
            probability = model.predict_proba(X[validation_index])[:, 1]
            fold_aucs.append(roc_auc_score(y[validation_index], probability))
        return np.mean(fold_means, axis=0), np.std(fold_means, axis=0), fold_aucs

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    tables = {}
    for name, make_model in makers.items():
        apparent_mean, apparent_std = apparent_importance(make_model)
        cv_mean, cv_std, fold_aucs = cv_importance(make_model)
        table = pd.DataFrame({
            "feature": features,
            "apparent_mean": apparent_mean,
            "apparent_std": apparent_std,
            "cv_mean": cv_mean,
            "cv_cvstd": cv_std,
        })
        table["apparent_pos"] = table["apparent_mean"] > 0
        table["cv_pos"] = table["cv_mean"] > 0
        table = table.sort_values("cv_mean", ascending=False)
        table.to_csv(args.output_dir / f"{name.lower()}_perm_importance_cv.csv", index=False)
        tables[name] = table
        results[name] = {
            "oof_fold_aucs": [float(value) for value in fold_aucs],
            "oof_auc_mean": float(np.mean(fold_aucs)),
            "n_apparent_pos": int(table["apparent_pos"].sum()),
            "n_cv_pos": int(table["cv_pos"].sum()),
        }

    locked_rf = pd.read_csv(args.locked_dir / "rf_perm_importance.csv")
    rf_table = tables["RF"].set_index("feature").loc[locked_rf["feature"]]
    rf_max_deviation = float(np.max(np.abs(
        rf_table["apparent_mean"].to_numpy() - locked_rf["mean"].to_numpy()
    )))
    rf_set_matches = set(tables["RF"].loc[tables["RF"]["apparent_pos"], "feature"]) == set(
        locked_rf.loc[locked_rf["mean"] > 0, "feature"]
    )

    selection_sets = json.loads((args.locked_dir / "selection_sets.json").read_text(encoding="utf-8"))
    vote_sets = {
        "FDR": set(selection_sets["sets"]["FDR"]),
        "LASSO": set(selection_sets["sets"]["LASSO"]),
        "RF": set(tables["RF"].loc[tables["RF"]["cv_pos"], "feature"]),
        "XGB": set(tables["XGB"].loc[tables["XGB"]["cv_pos"], "feature"]),
    }
    votes = {
        feature: {
            "votes": sum(feature in selected for selected in vote_sets.values()),
            "methods": [method for method, selected in vote_sets.items() if feature in selected],
        }
        for feature in sorted(set().union(*vote_sets.values()))
    }
    selected_cv = sorted(feature for feature, value in votes.items() if value["votes"] >= 2)

    output = {
        "scope": "Cross-validated sensitivity analysis; not a regeneration of the historical XGB vote",
        "protocol": {
            "folds": f"StratifiedKFold(n_splits={importance_config['cv_folds']}, shuffle=True, random_state={seed})",
            "apparent_repeats": importance_config["apparent_repeats"],
            "cv_repeats": importance_config["cv_repeats"],
            "rf_params": rf_params,
            "xgb_params": xgb_params,
        },
        "rf_locked_comparison": {
            "set_matches_locked": rf_set_matches,
            "max_abs_mean_deviation": rf_max_deviation,
        },
        "rf": results["RF"],
        "xgb": results["XGB"],
        "cv_vote_sets": {method: sorted(selected) for method, selected in vote_sets.items()},
        "final9_cv_votes": {feature: votes.get(feature, {"votes": 0, "methods": []})
                            for feature in FINAL_FEATURES},
        "ge2_set_cv": selected_cv,
        "runtime": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
    }
    (args.output_dir / "cv_permutation_importance_votes.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "rf": results["RF"],
        "xgb": results["XGB"],
        "rf_locked_comparison": output["rf_locked_comparison"],
        "runtime": output["runtime"],
    }, indent=2))


if __name__ == "__main__":
    main()
