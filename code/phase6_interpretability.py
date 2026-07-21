"""
Phase 6: Interpretability
可解释性分析：SHAP、列线图、DCA
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit

def calculate_shap_linear(model, X, feature_names):
    """
    线性模型的SHAP值计算
    SHAP = coef * (x - x_background)
    """
    coef = model.coef_.flatten()
    intercept = model.intercept_[0] if hasattr(model.intercept_, '__len__') else model.intercept_
    
    # 背景值（均值）
    background = X.mean(axis=0)
    
    # 计算SHAP值
    shap_values = np.zeros_like(X)
    for i in range(len(feature_names)):
        shap_values[:, i] = coef[i] * (X[:, i] - background[i])
    
    # 期望值
    expected_value = intercept + np.sum(coef * background)
    
    return shap_values, expected_value, background

def plot_shap_summary(shap_values, X, feature_names, output_path='shap_summary.png'):
    """SHAP重要性条形图"""
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    sorted_idx = np.argsort(mean_abs_shap)[::-1]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#8491B4', '#F39B7F', '#91D1C2', '#7E6148', '#B09C85']
    
    bars = ax.barh(range(len(feature_names)), mean_abs_shap[sorted_idx],
                   color=[colors[i % len(colors)] for i in range(len(feature_names))],
                   edgecolor='black', linewidth=0.5)
    
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels([feature_names[i] for i in sorted_idx], fontsize=11)
    ax.set_xlabel('Mean |SHAP Value|', fontsize=12)
    ax.set_title('SHAP Feature Importance', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                f'{width:.3f}', ha='left', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return mean_abs_shap

def decision_curve_analysis(y_true, y_prob, thresholds=None):
    """决策曲线分析"""
    if thresholds is None:
        thresholds = np.arange(0.01, 1.00, 0.01)
    
    n = len(y_true)
    n_pos = np.sum(y_true)
    prevalence = n_pos / n
    
    net_benefits = []
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        
        nb = (tp / n) - (fp / n) * (thresh / (1 - thresh))
        net_benefits.append(nb)
    
    # Treat all
    treat_all = [prevalence - (1 - prevalence) * (t / (1 - t)) for t in thresholds]
    
    # Treat none
    treat_none = [0] * len(thresholds)
    
    return thresholds, np.array(net_benefits), np.array(treat_all), np.array(treat_none)

def plot_dca(y_true, y_prob, model_name='Model', output_path='dca.png'):
    """绘制DCA曲线"""
    thresholds, nb, treat_all, treat_none = decision_curve_analysis(y_true, y_prob)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(thresholds, nb, color='#3C5488', linewidth=2.5, label=model_name)
    ax.plot(thresholds, treat_all, 'k--', linewidth=1, label='Treat all', alpha=0.5)
    ax.plot(thresholds, treat_none, 'k:', linewidth=1, label='Treat none', alpha=0.5)
    
    ax.set_xlabel('Threshold Probability', fontsize=12)
    ax.set_ylabel('Net Benefit', fontsize=12)
    ax.set_title(f'Decision Curve Analysis - {model_name}', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim([0, 1])
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    pass
