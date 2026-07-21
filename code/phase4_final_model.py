"""
Phase 4: Final Model Training & Evaluation
最终模型训练与评估
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, brier_score_loss, accuracy_score,
                             precision_score, recall_score, f1_score, confusion_matrix,
                             roc_curve, calibration_curve)
from scipy import stats
import json
import pickle

RANDOM_STATE = 42

def train_final_model(X_train, y_train, C=0.667, penalty='l2'):
    """训练最终Ridge模型"""
    model = LogisticRegression(
        C=C,
        penalty=penalty,
        solver='lbfgs',
        max_iter=1000,
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    return model

def bootstrap_auc_ci(y_true, y_proba, n_bootstrap=1000, confidence=0.95):
    """Bootstrap计算AUC的95%置信区间"""
    rng = np.random.RandomState(RANDOM_STATE)
    n = len(y_true)
    aucs = []
    
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_proba[idx]))
    
    aucs = np.array(aucs)
    alpha = 1 - confidence
    ci_lower = np.percentile(aucs, alpha/2 * 100)
    ci_upper = np.percentile(aucs, (1 - alpha/2) * 100)
    
    return np.mean(aucs), ci_lower, ci_upper

def calculate_calibration_slope(y_true, y_proba, n_bins=10):
    """
    计算校准斜率（Cox方法）
    标准方法：用glm拟合 observed ~ logit(predicted)
    """
    # 分箱
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy='quantile')
    
    # 避免logit(0)或logit(1)
    epsilon = 1e-7
    prob_true_clip = np.clip(prob_true, epsilon, 1 - epsilon)
    prob_pred_clip = np.clip(prob_pred, epsilon, 1 - epsilon)
    
    logit_obs = np.log(prob_true_clip / (1 - prob_true_clip))
    logit_pred = np.log(prob_pred_clip / (1 - prob_pred_clip))
    
    # 线性回归: logit_obs = intercept + slope * logit_pred
    if len(logit_pred) >= 2:
        slope, intercept, r_value, p_value, std_err = stats.linregress(logit_pred, logit_obs)
    else:
        slope, intercept = np.nan, np.nan
    
    return slope, intercept

def calculate_all_metrics(y_true, y_proba, threshold=0.5):
    """计算所有评估指标"""
    y_pred = (y_proba >= threshold).astype(int)
    
    # AUC和95% CI
    auc, auc_ci_low, auc_ci_high = bootstrap_auc_ci(y_true, y_proba)
    
    # 基础指标
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    sensitivity = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    brier = brier_score_loss(y_true, y_proba)
    
    # 特异性
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # 校准斜率
    cal_slope, cal_intercept = calculate_calibration_slope(y_true, y_proba)
    
    # Youden指数最优阈值
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    youden = tpr - fpr
    optimal_idx = np.argmax(youden)
    optimal_threshold = thresholds[optimal_idx]
    
    return {
        'AUC': auc,
        'AUC_95CI_Low': auc_ci_low,
        'AUC_95CI_High': auc_ci_high,
        'Accuracy': accuracy,
        'Precision': precision,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'F1_Score': f1,
        'Brier_Score': brier,
        'Calibration_Slope': cal_slope,
        'Calibration_Intercept': cal_intercept,
        'Optimal_Threshold': optimal_threshold
    }

def evaluate_model(model, X, y, dataset_name=''):
    """评估模型"""
    y_proba = model.predict_proba(X)[:, 1]
    metrics = calculate_all_metrics(y, y_proba)
    
    print(f"\n{dataset_name} Results:")
    print(f"  AUC: {metrics['AUC']:.3f} (95% CI: {metrics['AUC_95CI_Low']:.3f}-{metrics['AUC_95CI_High']:.3f})")
    print(f"  Brier: {metrics['Brier_Score']:.3f}")
    print(f"  Calibration Slope: {metrics['Calibration_Slope']:.3f}")
    print(f"  Sensitivity: {metrics['Sensitivity']:.3f}")
    print(f"  Specificity: {metrics['Specificity']:.3f}")
    
    return metrics

if __name__ == '__main__':
    pass
