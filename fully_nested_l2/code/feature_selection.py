# -*- coding: utf-8 -*-
"""
完全嵌套 L2 敏感性分析 —— 模块 feature_selection.py
======================================================
四方法特征筛选 + 迭代 VIF 过滤。全部只在当前训练子集内执行。
投票规则: >= 2/4 票; 无 9 个上限; 不强制保留原九个变量。
fallback 规则在运行前冻结（见 config.json）。
"""
import json
import numpy as np
import pandas as pd
from scipy import stats as sstats
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from preprocessing import ALL_53, DERIVED_FEATURES, SQUARE_OF

CONFIG = json.load(open(__file__.replace("feature_selection.py", "config.json"), encoding="utf-8"))
C_GRID_LASSO = np.logspace(-4, 2, 100)


# ---------------------------------------------------------------- FDR
def fdr_select(X: pd.DataFrame, y: np.ndarray, names=None):
    """单变量 logistic 回归 Wald p + Benjamini-Hochberg。返回 (保留集合, q值Series)。"""
    names = names or list(X.columns)
    pvals = {}
    for c in names:
        x = X[c].values.astype(float)
        if np.nanstd(x) < 1e-12 or len(np.unique(x)) < 2:
            pvals[c] = 1.0
            continue
        try:
            lr = LogisticRegression(penalty="l2", solver="lbfgs", max_iter=1000)
            lr.fit(x.reshape(-1, 1), y)
            beta = lr.coef_[0][0]
            # Wald 标准误: 用 Fisher 信息估计
            p1 = lr.predict_proba(x.reshape(-1, 1))[:, 1]
            w = p1 * (1 - p1)
            se = np.sqrt(1.0 / max(np.sum(w * (x - x.mean()) ** 2), 1e-12))
            z = abs(beta) / se
            pvals[c] = float(2 * sstats.norm.sf(z))
        except Exception:
            pvals[c] = 1.0
    pv = pd.Series(pvals)
    m = len(pv)
    order = pv.sort_values()
    bh = order * m / np.arange(1, m + 1)
    bh = bh.iloc[::-1].cummin().iloc[::-1].clip(upper=1.0)
    bh = bh[pv.index]  # 还原原顺序
    keep = set(bh[bh < 0.05].index)
    return keep, bh


# ---------------------------------------------------------------- LASSO 1-SE
def lasso_1se_select(X: pd.DataFrame, y: np.ndarray, seed: int):
    """10 折分层 CV，100 个 C，1-SE 规则。返回 (保留集合, 日志dict)。"""
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    aucs = np.zeros((10, len(C_GRID_LASSO)))
    for i, (tr, va) in enumerate(skf.split(X.values, y)):
        # C 升序路径 warm-start: 数值优化，不改变冻结的 estimator(l1 saga)/C 网格/1-SE 规则
        m = LogisticRegression(penalty="l1", solver="saga", C=C_GRID_LASSO[0],
                               max_iter=10000, class_weight=None, random_state=seed,
                               warm_start=True)
        for j, c in enumerate(C_GRID_LASSO):
            m.set_params(C=c)
            m.fit(X.values[tr], y[tr])
            p = m.predict_proba(X.values[va])[:, 1]
            aucs[i, j] = roc_auc_score(y[va], p) if len(np.unique(y[va])) == 2 else np.nan
    mean_auc = np.nanmean(aucs, axis=0)
    se = np.nanstd(aucs, axis=0, ddof=1) / np.sqrt(aucs.shape[0])
    best_j = int(np.argmax(mean_auc))
    # 并列（差 < 1e-12）取更小 C -> 从最小 C 侧扫描第一个达到最大值
    tol = 1e-12
    c_min_idx = int(np.argmax(mean_auc >= mean_auc[best_j] - tol))
    c_min = float(C_GRID_LASSO[c_min_idx])
    thresh = mean_auc[c_min_idx] - se[c_min_idx]
    eligible = np.where(mean_auc >= thresh - 1e-15)[0]
    c_1se_idx = int(eligible[0])  # 最小 C
    c_1se = float(C_GRID_LASSO[c_1se_idx])
    m = LogisticRegression(penalty="l1", solver="saga", C=c_1se, max_iter=10000,
                          class_weight=None, random_state=seed)
    m.fit(X.values, y)
    keep = set(np.array(list(X.columns))[np.abs(m.coef_[0]) > 1e-8])
    log = {"C_min": c_min, "mean_auc_at_C_min": float(mean_auc[c_min_idx]),
           "se_at_C_min": float(se[c_min_idx]),
           "C_1se": c_1se, "mean_auc_at_C_1se": float(mean_auc[c_1se_idx]),
           "se_at_C_1se": float(se[c_1se_idx]), "n_nonzero": len(keep)}
    return keep, log


# ---------------------------------------------------------------- RF / XGB 置换重要性
def _selector_holdout(X, y, seed):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr, ho = next(sss.split(X, y))
    return tr, ho


def rf_perm_select(X: pd.DataFrame, y: np.ndarray, seed: int):
    cfg = CONFIG["feature_selection"]["rf_permutation"]["config"]
    tr, ho = _selector_holdout(X.values, y, seed)
    rf = RandomForestClassifier(n_estimators=cfg["n_estimators"], max_depth=cfg["max_depth"],
                                min_samples_leaf=cfg["min_samples_leaf"],
                                max_features=cfg["max_features"],
                                random_state=cfg["random_state"],
                                class_weight=cfg["class_weight"], n_jobs=1)
    rf.fit(X.values[tr], y[tr])
    pi = permutation_importance(rf, X.values[ho], y[ho], scoring="roc_auc",
                                n_repeats=20, random_state=42)
    names = np.array(list(X.columns))
    keep = set(names[pi.importances_mean > 0])
    return keep, {"holdout_auc": float(roc_auc_score(y[ho], rf.predict_proba(X.values[ho])[:, 1])),
                  "n_positive": int(len(keep))}


def xgb_perm_select(X: pd.DataFrame, y: np.ndarray, seed: int):
    import xgboost as xgb
    cfg = CONFIG["feature_selection"]["xgb_permutation"]["config"]
    tr, ho = _selector_holdout(X.values, y, seed)
    spw = float((y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1))
    clf = xgb.XGBClassifier(n_estimators=cfg["n_estimators"], max_depth=cfg["max_depth"],
                            learning_rate=cfg["learning_rate"], scale_pos_weight=spw,
                            random_state=cfg["random_state"], eval_metric=cfg["eval_metric"],
                            n_jobs=1)
    clf.fit(X.values[tr], y[tr])
    pi = permutation_importance(clf, X.values[ho], y[ho], scoring="roc_auc",
                                n_repeats=20, random_state=42)
    names = np.array(list(X.columns))
    keep = set(names[pi.importances_mean > 0])
    return keep, {"holdout_auc": float(roc_auc_score(y[ho], clf.predict_proba(X.values[ho])[:, 1])),
                  "scale_pos_weight": spw, "n_positive": int(len(keep))}


# ---------------------------------------------------------------- VIF
def _vif_values(X: np.ndarray, names):
    """带截距的 VIF。返回 {name: vif}；完全共线性 -> inf。"""
    out = {}
    n, p = X.shape
    for i in range(p):
        others = [j for j in range(p) if j != i]
        A = np.column_stack([X[:, others], np.ones(n)])
        try:
            beta, *_ = np.linalg.lstsq(A, X[:, i], rcond=None)
            pred = A @ beta
            ss_res = ((X[:, i] - pred) ** 2).sum()
            ss_tot = ((X[:, i] - X[:, i].mean()) ** 2).sum()
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            out[names[i]] = float(1 / (1 - r2)) if r2 < 1 - 1e-12 else np.inf
        except np.linalg.LinAlgError:
            out[names[i]] = np.inf
    return out


def iterative_vif_filter(X: pd.DataFrame, votes: dict, threshold: float = 5.0):
    """
    迭代 VIF 过滤（无核心变量保护）。
    预定义删除规则:
      1) VIF=inf（完全共线性）: 删除与其他变量平均绝对相关最高、且不是临床主效应的派生项；
      2) 主效应与其平方项同时在候选中时，优先删除平方项；
      3) 否则删除 VIF 最大者（并列 -> 派生项优先 -> 名称序）。
    """
    feats = list(X.columns)
    removed_log = []
    fallback = None
    while True:
        if not feats:
            # VIF 删除后为空: 保留票数最高且 VIF 最低的一个
            fallback = "empty_vif_top_vote"
            break
        vm = _vif_values(X[feats].values, feats)
        vmax = max(vm.values())
        if vmax <= threshold:
            break
        infs = [f for f in feats if np.isinf(vm[f])]
        if infs:
            corr = X[feats].corr().abs().mean()
            derived_infs = [f for f in infs if f in DERIVED_FEATURES]
            pool = derived_infs if derived_infs else infs
            drop = max(pool, key=lambda f: (corr[f], f))
            reason = "perfect_collinearity"
        else:
            cands = [f for f in feats if vm[f] > threshold]
            drop = None
            for f in cands:
                sq = SQUARE_OF.get(f)
                if sq is None:
                    # f 是主效应，其平方项在候选中 -> 删平方项
                    for k, v in SQUARE_OF.items():
                        if v == f and k in cands:
                            drop = k
                            break
                if drop:
                    break
            if drop is None:
                drop = max(cands, key=lambda f: (vm[f], 0 if f in DERIVED_FEATURES else 1, f))
            reason = "vif_gt_threshold"
        removed_log.append({"feature": drop, "vif_before": vm[drop], "reason": reason,
                            "max_vif_after_pool": vmax})
        feats.remove(drop)
    if fallback:
        # 在全部被 VIF 删除前的候选中恢复: 用进入 VIF 前的全部特征中票数最高且 VIF 最低者
        all_prev = list(X.columns)
        vm = _vif_values(X[all_prev].values, all_prev)
        best = max(all_prev, key=lambda f: (votes.get(f, 0), -vm[f]))
        feats = [best]
    return feats, removed_log, fallback


# ---------------------------------------------------------------- 四方法投票
def four_method_selection(X: pd.DataFrame, y: np.ndarray, seed: int):
    """
    返回 dict: selected(投票>=2, 未经VIF), after_vif, votes, fallback,
               selector_logs, vif_removed, n_selected_before_vif, n_selected_after_vif
    投票为空 -> fallback='empty_vote_fdr'，取 FDR q 最小的一个特征。
    """
    fdr_keep, qvals = fdr_select(X, y)
    lasso_keep, lasso_log = lasso_1se_select(X, y, seed)
    rf_keep, rf_log = rf_perm_select(X, y, seed)
    xgb_keep, xgb_log = xgb_perm_select(X, y, seed)
    sets = {"fdr": {str(f) for f in fdr_keep}, "lasso": {str(f) for f in lasso_keep},
            "rf": {str(f) for f in rf_keep}, "xgb": {str(f) for f in xgb_keep}}
    votes = {str(f): int(sum(str(f) in s for s in sets.values())) for f in X.columns}
    voted = [str(f) for f in X.columns if votes[str(f)] >= 2]
    fallback = None
    if not voted:
        voted = [str(qvals.idxmin())]
        fallback = "empty_vote_fdr"
    Xv = X[voted]
    after_vif, vif_log, vif_fallback = iterative_vif_filter(Xv, votes)
    after_vif = [str(f) for f in after_vif]
    if vif_fallback:
        fallback = vif_fallback
    return {
        "selected_before_vif": voted,
        "selected_after_vif": after_vif,
        "votes": votes,
        "fallback": fallback,
        "selector_logs": {"fdr_n": len(fdr_keep), "lasso": lasso_log, "rf": rf_log, "xgb": xgb_log},
        "vif_removed": vif_log,
        "q_values": qvals.to_dict(),
    }
