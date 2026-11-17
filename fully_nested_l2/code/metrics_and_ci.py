# -*- coding: utf-8 -*-
"""
完全嵌套 L2 敏感性分析 —— 模块 metrics_and_ci.py
==================================================
不依赖阈值的指标 + 锁定阈值下的分类指标 + 患者聚类 bootstrap CI + 决策曲线。
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, roc_auc_score)


def cox_calibration(y, p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    lr.fit(logit, y)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def standard_metrics(y, p):
    slope, intercept = cox_calibration(y, p)
    return {
        "auc": float(roc_auc_score(y, p)),
        "auc_pr": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
    }


def classification_at_threshold(y, p, thr):
    yp = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yp, labels=[0, 1]).ravel()
    return {"threshold": float(thr), "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
            "specificity": tn / (tn + fp) if tn + fp else np.nan,
            "ppv": tp / (tp + fp) if tp + fp else np.nan,
            "npv": tn / (tn + fn) if tn + fn else np.nan,
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}


def decision_curve_net_benefit(y, p, thresholds):
    """返回各阈值下的净获益（ all 与 none 参照由调用方计算）。"""
    n = len(y)
    out = []
    for pt in thresholds:
        w = pt / (1 - pt)
        yp = p >= pt
        tp = int((yp & (y == 1)).sum())
        fp = int((yp & (y == 0)).sum())
        nb = tp / n - fp / n * w
        nb_all = (y == 1).sum() / n - (y == 0).sum() / n * w
        out.append({"threshold": float(pt), "net_benefit": float(nb),
                    "net_benefit_treat_all": float(nb_all)})
    return out


def patient_cluster_bootstrap_auc(pred_df, n_boot=1000, seed=20261106):
    """
    患者聚类 bootstrap 的主估计量 CI。

    每次以患者为单位有放回抽样，重复抽中的患者按重抽样倍数保留（加权）。
    在每个重样本内，分别计算 5 个重复级 pooled OOF AUC，再取它们的均值；
    对这组“5 次重复 AUC 均值”的 bootstrap 分布取 2.5%/97.5% 分位数。
    这与主估计 auc_primary_mean_of_repeat_aucs 的估计量一致，而不是把
    5 x n_resampled 条预测池化后计算单个 AUC。

    pred_df 列: patient_id, repeat, y, predicted_probability
    返回 (ci_low, ci_high, boot_mean_auc_array)
    """
    rng = np.random.default_rng(seed)
    piv = pred_df.pivot_table(index="patient_id", columns="repeat",
                              values=["y", "predicted_probability"])
    patient_ids = np.array(piv.index)
    y_mat = piv["y"].values          # (n_patient, n_repeat)
    p_mat = piv["predicted_probability"].values
    if np.isnan(y_mat).any() or np.isnan(p_mat).any():
        raise ValueError("每个患者必须在每次重复中恰好有一条有效预测")
    if not np.all(y_mat == y_mat[:, [0]]):
        raise ValueError("同一患者在不同重复中的结局标签不一致")

    n_pat = len(patient_ids)
    n_repeat = y_mat.shape[1]
    repeat_aucs = np.full((n_boot, n_repeat), np.nan)
    for b in range(n_boot):
        idx = rng.integers(0, n_pat, size=n_pat)
        yb = y_mat[idx, 0]
        if len(np.unique(yb)) < 2:
            continue
        for j in range(n_repeat):
            # idx 中的重复索引自然实现按重抽样倍数加权。
            repeat_aucs[b, j] = roc_auc_score(yb, p_mat[idx, j])

    aucs = np.nanmean(repeat_aucs, axis=1)
    valid = aucs[np.isfinite(aucs)]
    if len(valid) == 0:
        raise ValueError("没有同时包含阳性与阴性患者的有效 bootstrap 重样本")
    return (float(np.quantile(valid, 0.025)), float(np.quantile(valid, 0.975)), aucs)
