"""
完整评估脚本：生成8模型×3数据集的完整评估表、校准曲线、DCA图
使用已训练的模型（trained_models.pkl）进行评估
"""
import pandas as pd
import numpy as np
import json
import pickle
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import bootstrap
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, accuracy_score,
    precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, calibration_curve
)

warnings.filterwarnings('ignore')

# 设置随机种子和路径
GLOBAL_RANDOM_STATE = 42
np.random.seed(GLOBAL_RANDOM_STATE)

BASE_DIR = Path('output/pis/stage3_unified_eval')
MODELS_FILE = BASE_DIR / 'models' / 'trained_models.pkl'
FIGURES_DIR = BASE_DIR / 'figures'
OUTPUT_DIR = BASE_DIR / 'results'
OUTPUT_DIR.mkdir(exist_ok=True)

# 加载数据
STAGE1_DIR = Path('output/pis/stage1_optimized')
train_df = pd.read_csv(STAGE1_DIR / 'preprocessed_train.csv', index_col=0)
val_df = pd.read_csv(STAGE1_DIR / 'preprocessed_internal_val_fixed.csv', index_col=0)
ext_df = pd.read_csv(STAGE1_DIR / 'preprocessed_ext_val_v3.csv', index_col=0)

with open('output/pis/stage2_v3/selected_features.json', 'r', encoding='utf-8') as f:
    selected_features = json.load(f)['final_features']

# 准备数据集
datasets = {
    'Train': (train_df.drop(columns=['pis'])[selected_features].values, train_df['pis'].values),
    'Validation': (val_df.drop(columns=['pis'])[selected_features].values, val_df['pis'].values),
    'External': (ext_df.drop(columns=['pis'])[selected_features].values, ext_df['pis'].values)
}

# 加载已训练模型
with open(MODELS_FILE, 'rb') as f:
    trained_models = pickle.load(f)

print("[INFO] 已加载模型:", list(trained_models.keys()))

# 颜色配置
COLOR_PALETTE = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#8491B4', '#F39B7F', '#91D1C2', '#7E6148']
MODEL_ORDER = ['LogisticRegression', 'LASSO', 'Ridge', 'ElasticNet', 'RandomForest', 'XGBoost', 'LightGBM', 'SVM_RBF']
MODEL_LABELS = {
    'LogisticRegression': 'Logistic Regression',
    'LASSO': 'LASSO',
    'Ridge': 'Ridge',
    'ElasticNet': 'Elastic Net',
    'RandomForest': 'Random Forest',
    'XGBoost': 'XGBoost',
    'LightGBM': 'LightGBM',
    'SVM_RBF': 'SVM (RBF)'
}

# ============ 1. 完整评估指标计算 ============

def compute_auc_ci(y_true, y_prob, n_bootstrap=1000, confidence=0.95):
    """Bootstrap计算AUC的95%置信区间"""
    rng = np.random.RandomState(GLOBAL_RANDOM_STATE)
    n = len(y_true)
    aucs = []
    
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    
    aucs = np.array(aucs)
    alpha = 1 - confidence
    ci_lower = np.percentile(aucs, alpha/2 * 100)
    ci_upper = np.percentile(aucs, (1 - alpha/2) * 100)
    
    return np.mean(aucs), ci_lower, ci_upper

def compute_all_metrics(y_true, y_prob, threshold=0.5):
    """计算完整的评估指标"""
    y_pred = (y_prob >= threshold).astype(int)
    
    # AUC和95% CI
    auc, auc_ci_low, auc_ci_high = compute_auc_ci(y_true, y_prob)
    
    # 基础指标
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    sensitivity = recall_score(y_true, y_pred, zero_division=0)  # 敏感度 = 召回率
    f1 = f1_score(y_true, y_pred, zero_division=0)
    brier = brier_score_loss(y_true, y_prob)
    
    # 特异性
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        'AUC': auc,
        'AUC_95CI_Low': auc_ci_low,
        'AUC_95CI_High': auc_ci_high,
        'Accuracy': accuracy,
        'Precision': precision,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'F1_Score': f1,
        'Brier_Score': brier
    }

# 计算所有模型在所有数据集上的指标
print("\n[INFO] 开始计算完整评估指标...")
all_results = []

for model_name in MODEL_ORDER:
    if model_name not in trained_models:
        continue
    model = trained_models[model_name]
    
    for dataset_name, (X, y) in datasets.items():
        y_prob = model.predict_proba(X)[:, 1]
        metrics = compute_all_metrics(y, y_prob)
        
        row = {
            'Model': MODEL_LABELS[model_name],
            'Dataset': dataset_name,
            **metrics
        }
        all_results.append(row)
        print(f"  [{model_name}] {dataset_name}: AUC={metrics['AUC']:.3f} [{metrics['AUC_95CI_Low']:.3f}-{metrics['AUC_95CI_High']:.3f}]")

results_df = pd.DataFrame(all_results)

# 保存完整结果
results_df.to_excel(OUTPUT_DIR / 'complete_evaluation.xlsx', index=False)
print(f"\n[PASS] 完整评估表已保存: {OUTPUT_DIR / 'complete_evaluation.xlsx'}")

# ============ 2. 生成格式化评估表（按数据集分组） ============

def format_table(df_dataset):
    """格式化单个数据集的评估表"""
    rows = []
    for _, row in df_dataset.iterrows():
        rows.append({
            'Model': row['Model'],
            'AUC': f"{row['AUC']:.3f}",
            '95% CI': f"{row['AUC_95CI_Low']:.3f}–{row['AUC_95CI_High']:.3f}",
            'Accuracy': f"{row['Accuracy']:.3f}",
            'Precision': f"{row['Precision']:.3f}",
            'Sensitivity': f"{row['Sensitivity']:.3f}",
            'Specificity': f"{row['Specificity']:.3f}",
            'F1 Score': f"{row['F1_Score']:.3f}",
            'Brier Score': f"{row['Brier_Score']:.3f}"
        })
    return pd.DataFrame(rows)

# 按数据集输出
for dataset_name in ['Train', 'Validation', 'External']:
    df_subset = results_df[results_df['Dataset'] == dataset_name].copy()
    df_subset = df_subset.sort_values('AUC', ascending=False)
    
    formatted = format_table(df_subset)
    formatted.to_excel(OUTPUT_DIR / f'evaluation_table_{dataset_name.lower()}.xlsx', index=False)
    
    print(f"\n{'='*80}")
    print(f"[{dataset_name} Set] 评估表（按AUC排序）")
    print(f"{'='*80}")
    print(formatted.to_string(index=False))

# ============ 3. 校准曲线 ============

print("\n[INFO] 生成校准曲线...")

def plot_calibration_curves_for_dataset(dataset_name, X, y, output_path):
    """为单个数据集绘制所有模型的校准曲线"""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfectly calibrated')
    
    for idx, model_name in enumerate(MODEL_ORDER):
        if model_name not in trained_models:
            continue
        model = trained_models[model_name]
        y_prob = model.predict_proba(X)[:, 1]
        
        prob_true, prob_pred = calibration_curve(y, y_prob, n_bins=10, strategy='quantile')
        
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
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    fig.savefig(output_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [PASS] {dataset_name}校准曲线已保存")

for dataset_name, (X, y) in datasets.items():
    plot_calibration_curves_for_dataset(
        dataset_name, X, y,
        FIGURES_DIR / dataset_name.lower() / 'calibration_curves.png'
    )

# ============ 4. DCA曲线 ============

print("\n[INFO] 生成DCA曲线...")

def decision_curve_analysis(y_true, y_prob, thresholds=None):
    """计算决策曲线分析的净收益"""
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
    treat_none = [0] * len(thresholds)
    
    return thresholds, np.array(net_benefits), np.array(treat_all), np.array(treat_none)

def plot_dca_for_dataset(dataset_name, X, y, output_path):
    """为单个数据集绘制DCA曲线"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for idx, model_name in enumerate(MODEL_ORDER):
        if model_name not in trained_models:
            continue
        model = trained_models[model_name]
        y_prob = model.predict_proba(X)[:, 1]
        
        thresholds, nb, treat_all, treat_none = decision_curve_analysis(y, y_prob)
        
        ax.plot(thresholds, nb, color=COLOR_PALETTE[idx % len(COLOR_PALETTE)],
                linewidth=2, label=MODEL_LABELS[model_name])
    
    # 基线
    thresholds = np.arange(0.01, 1.00, 0.01)
    n_pos = y.sum()
    n = len(y)
    treat_all = [(n_pos / n) - (1 - n_pos / n) * (t / (1 - t)) for t in thresholds]
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
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    fig.savefig(output_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [PASS] {dataset_name} DCA曲线已保存")

for dataset_name, (X, y) in datasets.items():
    plot_dca_for_dataset(
        dataset_name, X, y,
        FIGURES_DIR / dataset_name.lower() / 'dca_curves.png'
    )

# ============ 5. 汇总输出 ============

print(f"\n{'='*60}")
print("[COMPLETE] 所有评估完成！")
print(f"{'='*60}")
print(f"\n输出文件:")
print(f"  - 完整评估表: {OUTPUT_DIR / 'complete_evaluation.xlsx'}")
print(f"  - 训练集评估表: {OUTPUT_DIR / 'evaluation_table_train.xlsx'}")
print(f"  - 验证集评估表: {OUTPUT_DIR / 'evaluation_table_validation.xlsx'}")
print(f"  - 外部验证集评估表: {OUTPUT_DIR / 'evaluation_table_external.xlsx'}")
print(f"\n图表文件:")
for dataset_name in ['Train', 'Validation', 'External']:
    d = dataset_name.lower()
    print(f"  [{dataset_name}] 校准曲线: {FIGURES_DIR / d / 'calibration_curves.png'}")
    print(f"  [{dataset_name}] DCA曲线: {FIGURES_DIR / d / 'dca_curves.png'}")
