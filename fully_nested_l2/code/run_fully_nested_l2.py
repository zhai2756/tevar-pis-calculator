# -*- coding: utf-8 -*-
"""
完全嵌套 L2 敏感性分析 —— 主脚本 run_fully_nested_l2.py
==========================================================
外层: RepeatedStratifiedKFold(10, 5, seed=42) -> 50 个外层折。
每个外层训练折内:
  内层 5 折分层 CV: 每个内层训练折重新做 插补/截断/特征工程/RobustScaler/
  四方法筛选/VIF；50 个 C 逐一拟合评估 -> 选最优 C (并列取小 C, 边界记录)。
外层训练折: 重新插补/缩放/筛选/VIF，用选定 C 拟合最终 L2 模型，预测外层测试折。
外层测试折只用于最终预测。
用法:
  python run_fully_nested_l2.py --folds 0          # 只跑外层折 0（试跑）
  python run_fully_nested_l2.py                    # 全部 50 折，逐折 checkpoint 并行
  python run_fully_nested_l2.py --summarize-only   # 汇总已完成的折
"""
import argparse
import json
import logging
import os
import sys
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

CODE = Path(__file__).resolve().parent
BASE = CODE.parent
DATA, RESULTS, LOGS = BASE / "data", BASE / "results", BASE / "logs"
FOLDS = RESULTS / "folds"
for d in (RESULTS, LOGS, FOLDS):
    d.mkdir(parents=True, exist_ok=True)

CONFIG = json.load(open(CODE / "config.json", encoding="utf-8"))
SEED = CONFIG["seed"]
C_GRID = np.logspace(-4, 2, 50)

sys.path.insert(0, str(CODE))
import preprocessing as pp  # noqa: E402
from feature_selection import four_method_selection  # noqa: E402

logging.basicConfig(filename=LOGS / "run.log", level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    encoding="utf-8")
logw = logging.getLogger("warnings")
fh = logging.FileHandler(LOGS / "warnings.log", encoding="utf-8")
logw.addHandler(fh)


def build_raw():
    df = pd.read_csv(DATA / "raw_training_540.csv")
    y = df["pis"].values.astype(int)
    feats = [c for c in df.columns if c not in ("patient_id", "pis")]
    return df, feats, y


def run_outer_fold(outer_id, df, feats, y):
    """运行单个外层折（outer_id 0..49）。返回可 JSON 序列化的结果 dict。"""
    outer = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=SEED)
    splits = list(outer.split(df[feats].values, y))
    otr, ote = splits[outer_id]
    y_tr, y_te = y[otr], y[ote]

    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + outer_id)
    inner_sel_seed = SEED + (outer_id % 10) * 10
    outer_sel_seed = SEED + outer_id * 100

    c_scores = {float(c): [] for c in C_GRID}
    inner_records = []
    raw_tr = df.iloc[otr]

    for j, (itr, iva) in enumerate(inner.split(raw_tr[feats].values, y_tr)):
        itr_abs = otr[itr]
        iva_abs = otr[iva]
        X_it, prep_it = pp.fit_transform(df.iloc[itr_abs][feats])
        y_it = y[itr_abs]
        X_iv = pp.transform(df.iloc[iva_abs][feats], prep_it)
        y_iv = y[iva_abs]
        sel = four_method_selection(X_it, y_it, seed=inner_sel_seed + j)
        cols = sel["selected_after_vif"]
        for c in C_GRID:
            m = LogisticRegression(C=c, penalty="l2", solver="lbfgs", max_iter=1000,
                                   class_weight=None)
            m.fit(X_it[cols].values, y_it)
            p = m.predict_proba(X_iv[cols].values)[:, 1]
            c_scores[float(c)].append(float(roc_auc_score(y_iv, p)))
        inner_records.append({
            "inner_fold": j,
            "n_inner_train": int(len(itr_abs)),
            "n_inner_valid": int(len(iva_abs)),
            "selected_before_vif": sel["selected_before_vif"],
            "selected_after_vif": sel["selected_after_vif"],
            "n_selected_after_vif": len(cols),
            "votes": sel["votes"],
            "fallback": sel["fallback"],
            "selector_logs": sel["selector_logs"],
            "vif_removed": sel["vif_removed"],
        })

    # ---- 选 C ----
    stats = []
    for c in C_GRID:
        a = np.array(c_scores[float(c)])
        stats.append({"C": float(c), "mean_auc": float(a.mean()), "sd_auc": float(a.std(ddof=1)),
                      "min_auc": float(a.min()), "max_auc": float(a.max())})
    max_mean = max(s["mean_auc"] for s in stats)
    tol = 1e-6
    best_stats = [s for s in stats if s["mean_auc"] >= max_mean - tol]
    best_stats.sort(key=lambda s: s["C"])
    best_c = best_stats[0]["C"]
    c_at_boundary = bool(best_c == C_GRID[0] or best_c == C_GRID[-1])

    # ---- 外层完整训练集重新拟合 ----
    X_ot, prep_ot = pp.fit_transform(raw_tr[feats])
    sel_o = four_method_selection(X_ot, y_tr, seed=outer_sel_seed)
    cols_o = sel_o["selected_after_vif"]
    model = LogisticRegression(C=best_c, penalty="l2", solver="lbfgs", max_iter=1000,
                               class_weight=None)
    model.fit(X_ot[cols_o].values, y_tr)
    X_ote = pp.transform(df.iloc[ote][feats], prep_ot)
    p_test = model.predict_proba(X_ote[cols_o].values)[:, 1]
    outer_test_auc = float(roc_auc_score(y_te, p_test))

    # 泄漏自检: 内外层索引无交集
    assert len(set(otr) & set(ote)) == 0
    assert len(set(itr_abs) & set(iva_abs)) == 0

    return {
        "outer_id": int(outer_id),
        "repeat": int(outer_id // 10),
        "outer_fold": int(outer_id % 10),
        "n_outer_train": int(len(otr)),
        "events_outer_train": int(y_tr.sum()),
        "n_outer_test": int(len(ote)),
        "events_outer_test": int(y_te.sum()),
        "selected_c": float(best_c),
        "c_at_boundary": c_at_boundary,
        "c_tuning_stats": stats,
        "inner_mean_auc_at_selected_c": float(np.mean(c_scores[best_c])),
        "n_selected_before_vif": len(sel_o["selected_before_vif"]),
        "n_selected_after_vif": len(cols_o),
        "selected_features": cols_o,
        "selected_before_vif": sel_o["selected_before_vif"],
        "vif_removed_features": [r["feature"] for r in sel_o["vif_removed"]],
        "vif_removed_log": sel_o["vif_removed"],
        "votes_outer": sel_o["votes"],
        "fallback": sel_o["fallback"],
        "selector_logs_outer": sel_o["selector_logs"],
        "outer_test_auc": outer_test_auc,
        "inner_records": inner_records,
        "predictions": {
            "patient_id": [int(i) for i in df.iloc[ote]["patient_id"].values],
            "repeat": int(outer_id // 10),
            "outer_fold": int(outer_id % 10),
            "y": [int(v) for v in y_te],
            "predicted_probability": [float(v) for v in p_test],
        },
        "prep_params_summary": {
            "n_impute_medians": len(prep_ot["impute"]["medians"]),
            "n_scaler_features": len(prep_ot["scaler"]["center"]),
        },
    }


def _worker(outer_id):
    df, feats, y = build_raw()
    try:
        return outer_id, run_outer_fold(outer_id, df, feats, y), None
    except Exception as e:
        return outer_id, None, traceback.format_exc()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=str, default="all",
                    help="'all' 或逗号分隔的外层折编号 (0..49)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()

    if args.folds == "all":
        ids = list(range(50))
    else:
        ids = [int(x) for x in args.folds.split(",")]

    if not args.summarize_only:
        todo = [i for i in ids if not (FOLDS / f"outer_fold_{i:02d}.json").exists()]
        logging.info(f"folds requested={ids} todo={len(todo)}")
        print(f"[run] {len(todo)} folds to compute, {len(ids)-len(todo)} cached")
        done = 0
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_worker, i): i for i in todo}
            for fut in as_completed(futs):
                i = futs[fut]
                try:
                    outer_id, res, err = fut.result()
                except Exception:
                    logging.error(f"fold {i} worker crashed: {traceback.format_exc()}")
                    continue
                if err:
                    logging.error(f"fold {i} failed: {err}")
                    logw.warning(f"fold {i} failed: {err}")
                    continue
                json.dump(res, open(FOLDS / f"outer_fold_{i:02d}.json", "w", encoding="utf-8"))
                done += 1
                logging.info(f"fold {outer_id} done ({done}/{len(todo)}): "
                             f"AUC={res['outer_test_auc']:.4f}, C={res['selected_c']:.6g}, "
                             f"nfeat={res['n_selected_after_vif']}")
                print(f"[fold {outer_id:02d}] AUC={res['outer_test_auc']:.4f} "
                      f"C={res['selected_c']:.6g} nfeat={res['n_selected_after_vif']} "
                      f"({done}/{len(todo)})")

    summarize()


def summarize():
    files = sorted(FOLDS.glob("outer_fold_*.json"))
    if not files:
        print("no fold results yet")
        return
    rows = []
    for f in files:
        r = json.load(open(f, encoding="utf-8"))
        rows.append({
            "repeat": r["repeat"], "outer_fold": r["outer_fold"],
            "n_outer_train": r["n_outer_train"], "events_outer_train": r["events_outer_train"],
            "n_outer_test": r["n_outer_test"], "events_outer_test": r["events_outer_test"],
            "selected_c": r["selected_c"], "c_at_boundary": r["c_at_boundary"],
            "inner_mean_auc_at_selected_c": r["inner_mean_auc_at_selected_c"],
            "n_selected_before_vif": r["n_selected_before_vif"],
            "n_selected_after_vif": r["n_selected_after_vif"],
            "selected_features": ";".join(r["selected_features"]),
            "vif_removed_features": ";".join(r["vif_removed_features"]),
            "fallback": r["fallback"] or "",
            "outer_test_auc": r["outer_test_auc"],
        })
    df = pd.DataFrame(rows).sort_values(["repeat", "outer_fold"]).reset_index(drop=True)
    df.to_csv(RESULTS / "outer_fold_summary.csv", index=False, encoding="utf-8-sig")
    print(f"[summary] {len(df)}/50 folds | "
          f"fold AUC mean={df['outer_test_auc'].mean():.4f} sd={df['outer_test_auc'].std(ddof=1):.4f}")
    auc_primary = df.groupby("repeat")["outer_test_auc"].mean()
    print(f"[summary] repeat-level AUC: {dict(np.round(auc_primary, 4))}")
    print(f"[summary] primary (mean of repeat AUCs) = {auc_primary.mean():.4f} "
          f"(sd={auc_primary.std(ddof=1):.4f})")


if __name__ == "__main__":
    main()
