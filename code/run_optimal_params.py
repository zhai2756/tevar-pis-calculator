"""
使用最优超参数重新运行统一评估
加载各模型最优超参数，重新训练，评估三个数据集
"""
import pandas as pd
import numpy as np
import json
import pickle
import warnings
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, accuracy_score,
    precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings('ignore')

GLOBAL_RANDOM_STATE = 42
np.random.seed(GLOBAL_RANDOM_STATE)

WORKSPACE = Path(r'C:\Users\zjc170\Documents\kimi\workspace')
BASE_DIR = WORKSPACE / 'output' / 'pis' / 'stage3_unified_eval'
FIGURES_DIR = BASE_DIR / 'figures'
OUTPUT_DIR = BASE_DIR / 'results'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 加载数据
STAGE1_DIR = WORKSPACE / 'output' / 'pis' / 'stage1_optimized'
train_df = pd.read_csv(STAGE1_DIR / 'preprocessed_train.csv', index_col=0)
val_df = pd.read_csv(STAGE1_DIR / 'preprocessed_internal_val_fixed.csv', index_col=0)
ext_df = pd.read_csv(STAGE1_DIR / 'preprocessed_ext_val_v3.csv', index_col=0)

with open(WORKSPACE / 'output' / 'pis' / 'stage2_v3' / 'selected_features.json', 'r', encoding='utf-8') as f:
    selected_features = json.load(f)['final_features']

X_train = train_df.drop(columns=['pis'])[selected_features].values
y_train = train_df['pis'].values
X_val = val_df.drop(columns=['pis'])[selected_features].values
y_val = val_df['pis'].values
X_ext = ext_df.drop(columns=['pis'])[selected_features].values
y_ext = ext_df['pis'].values

datasets = {
    'Train': (X_train, y_train),
    'Validation': (X_val, y_val),
    'External': (X_ext, y_ext)
}

# ============ 最优超参数配置 ============
# 基于之前阶段三10折CV优化的结果

# 计算scale_pos_weight
n_neg = (y_train == 0).sum()
n_pos = (y_train == 1).sum()
spw = n_neg / n_pos
print(f"[INFO] scale_pos_weight = {spw:.4f}")

MODEL_CONFIGS = {
    'LogisticRegression': {
        'model': LogisticRegression,
        'params': {'C': 0.6688747907702062, 'max_iter': 1000, 'solver': 'lbfgs',
                   'penalty': 'l2', 'random_state': GLOBAL_RANDOM_STATE}
    },
    'LASSO': {
        'model': LogisticRegression,
        'params': {'C': 0.13292918943162169, 'max_iter': 10000, 'solver': 'saga',
                   'penalty': 'l1', 'random_state': GLOBAL_RANDOM_STATE}
    },
    'Ridge': {
        'model': LogisticRegression,
        'params': {'C': 0.6672367170464207, 'max_iter': 1000, 'solver': 'lbfgs',
                   'penalty': 'l2', 'random_state': GLOBAL_RANDOM_STATE}
    },
    'ElasticNet': {
        'model': LogisticRegression,
        'params': {'C': 0.13292918943162169, 'l1_ratio': 0.8605714451279329,
                   'max_iter': 10000, 'solver': 'saga', 'penalty': 'elasticnet',
                   'random_state': GLOBAL_RANDOM_STATE}
    },
    'RandomForest': {
        'model': RandomForestClassifier,
        'params': {'n_estimators': 99, 'max_depth': 5, 'min_samples_split': 2,
                   'min_samples_leaf': 5, 'max_features': 'sqrt',
                   'random_state': GLOBAL_RANDOM_STATE, 'n_jobs': 1}
    },
    'XGBoost': {
        'model': XGBClassifier,
        'params': {'n_estimators': 149, 'max_depth': 3, 'learning_rate': 0.012184186502221764,
                   'random_state': GLOBAL_RANDOM_STATE, 'n_jobs': 1,
                   'eval_metric': 'logloss', 'scale_pos_weight': spw}
    },
    'LightGBM': {
        'model': LGBMClassifier,
        'params': {'n_estimators': 199, 'max_depth': 5, 'learning_rate': 0.016256670876862694,
                   'num_leaves': 31, 'random_state': GLOBAL_RANDOM_STATE,
                   'verbosity': -1, 'scale_pos_weight': spw}
    },
    'SVM_RBF': {
        'model': SVC,
        'params': {'C': 0.04207988669606638, 'gamma': 0.002051110418843397,
                   'kernel': 'rbf', 'probability': True, 'random_state': GLOBAL_RANDOM_STATE}
    }
}

# 训练所有模型
print("\n[INFO] 使用最优超参数训练模型...")
trained_models = {}
for name, config in MODEL_CONFIGS.items():
    print(f"  Training {name}...")
    model = config['model'](**config['params'])
    model.fit(X_train, y_train)
    trained_models[name] = model

# 保存模型
with open(BASE_DIR / 'models' / 'trained_models_optimal.pkl', 'wb') as f:
    pickle.dump(trained_models, f)
print(f"\n[PASS] 最优参数模型已保存")

# ============ 评估指标计算 ============
COLOR_PALETTE = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#8491B4', '#F39B7F', '#91D1C2', '#7E6148']
MODEL_ORDER = ['LogisticRegression', 'LASSO', 'Ridge', 'ElasticNet', 'RandomForest', 'XGBoost', 'LightGBM', 'SVM_RBF']
MODEL_LABELS = {
    'LogisticRegression': 'Logistic Regression', 'LASSO': 'LASSO', 'Ridge': 'Ridge',
    'ElasticNet': 'Elastic Net', 'RandomForest': 'Random Forest',
    'XGBoost': 'XGBoost', 'LightGBM': 'LightGBM', 'SVM_RBF': 'SVM (RBF)'
}

def compute_auc_ci(y_true, y_prob, n_bootstrap=1000):
    rng = np.random.RandomState(GLOBAL_RANDOM_STATE)
    n = len(y_true)
    aucs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    aucs = np.array(aucs)
    return np.mean(aucs), np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)

def compute_all_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    auc, auc_ci_low, auc_ci_high = compute_auc_ci(y_true, y_prob)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    sensitivity = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    brier = brier_score_loss(y_true, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    return {
        'AUC': auc, 'AUC_95CI_Low': auc_ci_low, 'AUC_95CI_High': auc_ci_high,
        'Accuracy': accuracy, 'Precision': precision, 'Sensitivity': sensitivity,
        'Specificity': specificity, 'F1_Score': f1, 'Brier_Score': brier
    }

# 计算所有指标
print("\n[INFO] 计算完整评估指标...")
all_results = []
for model_name in MODEL_ORDER:
    model = trained_models[model_name]
    for dataset_name, (X, y) in datasets.items():
        y_prob = model.predict_proba(X)[:, 1]
        metrics = compute_all_metrics(y, y_prob)
        row = {'Model': MODEL_LABELS[model_name], 'Dataset': dataset_name, **metrics}
        all_results.append(row)

results_df = pd.DataFrame(all_results)
results_df.to_excel(OUTPUT_DIR / 'complete_evaluation_optimal.xlsx', index=False)

# 打印评估表
for dataset_name in ['Train', 'Validation', 'External']:
    df_subset = results_df[results_df['Dataset'] == dataset_name].copy()
    df_subset = df_subset.sort_values('AUC', ascending=False)
    print(f"\n[{dataset_name} Set]")
    for _, row in df_subset.iterrows():
        print(f"{row['Model']:20s} | AUC={row['AUC']:.3f} [{row['AUC_95CI_Low']:.3f}–{row['AUC_95CI_High']:.3f}] | "
              f"Acc={row['Accuracy']:.3f} | Prec={row['Precision']:.3f} | Sens={row['Sensitivity']:.3f} | "
              f"Spec={row['Specificity']:.3f} | F1={row['F1_Score']:.3f} | Brier={row['Brier_Score']:.3f}")

# ============ ROC曲线 ============
print("\n[INFO] 生成ROC曲线...")
for dataset_name, (X, y) in datasets.items():
    fig, ax = plt.subplots(figsize=(10, 10))
    for idx, model_name in enumerate(MODEL_ORDER):
        model = trained_models[model_name]
        y_prob = model.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, y_prob)
        auc = roc_auc_score(y, y_prob)
        ax.plot(fpr, tpr, color=COLOR_PALETTE[idx % len(COLOR_PALETTE)],
                linewidth=2.5, label=f"{MODEL_LABELS[model_name]} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax.set_xlabel('1 - Specificity', fontsize=12)
    ax.set_ylabel('Sensitivity', fontsize=12)
    ax.set_title(f'ROC Curves - {dataset_name} Set', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.tight_layout()
    out_path = FIGURES_DIR / dataset_name.lower() / 'roc_curves_optimal.png'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    fig.savefig(out_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [PASS] {dataset_name} ROC已保存")

# ============ 校准曲线 ============
print("\n[INFO] 生成校准曲线...")

def calibration_curve_manual(y_true, y_prob, n_bins=10):
    sorted_idx = np.argsort(y_prob)
    y_prob_sorted = y_prob[sorted_idx]
    y_true_sorted = y_true[sorted_idx]
    bin_size = len(y_prob) // n_bins
    prob_true = []
    prob_pred = []
    for i in range(n_bins):
        start = i * bin_size
        end = (i + 1) * bin_size if i < n_bins - 1 else len(y_prob)
        y_true_bin = y_true_sorted[start:end]
        y_prob_bin = y_prob_sorted[start:end]
        prob_true.append(np.mean(y_true_bin))
        prob_pred.append(np.mean(y_prob_bin))
    return np.array(prob_true), np.array(prob_pred)

for dataset_name, (X, y) in datasets.items():
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfectly calibrated')
    for idx, model_name in enumerate(MODEL_ORDER):
        model = trained_models[model_name]
        y_prob = model.predict_proba(X)[:, 1]
        prob_true, prob_pred = calibration_curve_manual(y, y_prob, n_bins=10)
        ax.plot(prob_pred, prob_true, 'o-', color=COLOR_PALETTE[idx % len(COLOR_PALETTE)],
                linewidth=2, markersize=6, label=MODEL_LABELS[model_name])
    ax.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax.set_ylabel('Fraction of Positives', fontsize=12)
    ax.set_title(f'Calibration Curves - {dataset_name} Set', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.tight_layout()
    out_path = FIGURES_DIR / dataset_name.lower() / 'calibration_curves_optimal.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    fig.savefig(out_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [PASS] {dataset_name} 校准曲线已保存")

# ============ DCA曲线 ============
print("\n[INFO] 生成DCA曲线...")

def decision_curve_analysis(y_true, y_prob, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.01, 1.00, 0.01)
    n = len(y_true)
    n_pos = y_true.sum()
    net_benefits = []
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        nb = (tp / n) - (fp / n) * (thresh / (1 - thresh))
        net_benefits.append(nb)
    treat_all = [(n_pos / n) - (1 - n_pos / n) * (t / (1 - t)) for t in thresholds]
    return thresholds, np.array(net_benefits), np.array(treat_all)

for dataset_name, (X, y) in datasets.items():
    fig, ax = plt.subplots(figsize=(10, 8))
    for idx, model_name in enumerate(MODEL_ORDER):
        model = trained_models[model_name]
        y_prob = model.predict_proba(X)[:, 1]
        thresholds, nb, treat_all = decision_curve_analysis(y, y_prob)
        ax.plot(thresholds, nb, color=COLOR_PALETTE[idx % len(COLOR_PALETTE)],
                linewidth=2, label=MODEL_LABELS[model_name])
    ax.plot(thresholds, treat_all, 'k--', linewidth=1, label='Treat all', alpha=0.5)
    ax.plot(thresholds, [0] * len(thresholds), 'k:', linewidth=1, label='Treat none', alpha=0.5)
    ax.set_xlabel('Threshold Probability', fontsize=12)
    ax.set_ylabel('Net Benefit', fontsize=12)
    ax.set_title(f'Decision Curve Analysis - {dataset_name} Set', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim([0, 1])
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linewidth=0.5)
    plt.tight_layout()
    out_path = FIGURES_DIR / dataset_name.lower() / 'dca_curves_optimal.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    fig.savefig(out_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [PASS] {dataset_name} DCA已保存")

print(f"\n{'='*60}")
print("[COMPLETE] 所有评估完成（最优超参数版）！")
print(f"{'='*60}")

def main(ctx):
    return {"status": "complete"}
