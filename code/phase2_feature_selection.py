"""
Phase 2: Feature Selection
特征选择阶段：四方法投票 + VIF筛选
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.inspection import permutation_importance
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42

def univariate_screening(X, y, alpha=0.10):
    """FDR校正的单因素逻辑回归筛选"""
    p_values = []
    feature_names = X.columns if hasattr(X, 'columns') else [f'f{i}' for i in range(X.shape[1])]
    
    for i in range(X.shape[1]):
        # 单因素逻辑回归
        model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        model.fit(X.iloc[:, i:i+1], y)
        # 使用Wald检验近似P值
        from sklearn.metrics import log_loss
        # 简化：使用相关性检验作为代理
        _, pval = stats.pearsonr(X.iloc[:, i], y)
        p_values.append(pval)
    
    # FDR校正 (Benjamini-Hochberg)
    from statsmodels.stats.multitest import multipletests
    reject, pvals_corrected, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
    
    selected = [feature_names[i] for i in range(len(feature_names)) if reject[i]]
    return selected, pvals_corrected

def lasso_selection(X, y, cv=5):
    """LASSO with 1-standard-error rule"""
    from sklearn.linear_model import LassoCV
    
    # 使用LassoCV找到最优alpha
    lasso_cv = LassoCV(cv=cv, random_state=RANDOM_STATE, max_iter=10000)
    lasso_cv.fit(X, y)
    
    # 1-standard-error rule: 选择lambda_1se
    # LassoCV不直接提供1se，我们用近似方法
    alphas = lasso_cv.alphas_
    mse_path = lasso_cv.mse_path_.mean(axis=1)
    min_mse_idx = np.argmin(mse_path)
    min_mse = mse_path[min_mse_idx]
    std_mse = lasso_cv.mse_path_.std(axis=1)[min_mse_idx] / np.sqrt(lasso_cv.mse_path_.shape[1])
    
    # 找到满足 MSE <= min_MSE + 1*SE 的最大alpha（最稀疏模型）
    threshold = min_mse + std_mse
    valid_idx = np.where(mse_path <= threshold)[0]
    selected_alpha_idx = valid_idx[-1] if len(valid_idx) > 0 else min_mse_idx
    selected_alpha = alphas[selected_alpha_idx]
    
    # 用选定的alpha重新拟合
    lasso_final = LogisticRegression(
        penalty='l1', solver='saga', C=1/selected_alpha if selected_alpha > 0 else 1.0,
        max_iter=10000, random_state=RANDOM_STATE
    )
    lasso_final.fit(X, y)
    
    # 选择非零系数特征
    coef = lasso_final.coef_.flatten()
    feature_names = X.columns if hasattr(X, 'columns') else [f'f{i}' for i in range(X.shape[1])]
    selected = [feature_names[i] for i in range(len(coef)) if abs(coef[i]) > 1e-6]
    
    return selected, lasso_final

def rf_permutation_importance(X, y, n_repeats=10, top_n=15):
    """随机森林排列重要性"""
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    rf.fit(X, y)
    
    result = permutation_importance(rf, X, y, n_repeats=n_repeats, 
                                    random_state=RANDOM_STATE, scoring='roc_auc')
    
    feature_names = X.columns if hasattr(X, 'columns') else [f'f{i}' for i in range(X.shape[1])]
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': result.importances_mean,
        'importance_std': result.importances_std
    }).sort_values('importance_mean', ascending=False)
    
    return importance_df.head(top_n)['feature'].tolist(), importance_df

def xgb_permutation_importance(X, y, n_repeats=10, top_n=20):
    """XGBoost排列重要性"""
    xgb = XGBClassifier(n_estimators=100, random_state=RANDOM_STATE, eval_metric='logloss')
    xgb.fit(X, y)
    
    result = permutation_importance(xgb, X, y, n_repeats=n_repeats,
                                    random_state=RANDOM_STATE, scoring='roc_auc')
    
    feature_names = X.columns if hasattr(X, 'columns') else [f'f{i}' for i in range(X.shape[1])]
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': result.importances_mean,
        'importance_std': result.importances_std
    }).sort_values('importance_mean', ascending=False)
    
    return importance_df.head(top_n)['feature'].tolist(), importance_df

def calculate_vif(X, threshold=5.0):
    """计算VIF并返回高VIF特征"""
    feature_names = X.columns if hasattr(X, 'columns') else [f'f{i}' for i in range(X.shape[1])]
    vif_data = pd.DataFrame()
    vif_data['feature'] = feature_names
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    
    high_vif = vif_data[vif_data['VIF'] > threshold]['feature'].tolist()
    return vif_data, high_vif

def four_method_voting(X, y, core_features=None, vote_threshold=2):
    """
    四方法投票特征选择
    1. FDR校正单因素分析
    2. LASSO 1se
    3. RF排列重要性
    4. XGBoost排列重要性
    
    然后VIF筛选，最终保留得票>=threshold的特征
    """
    feature_names = X.columns.tolist() if hasattr(X, 'columns') else [f'f{i}' for i in range(X.shape[1])]
    votes = {f: 0 for f in feature_names}
    
    # Method 1: 单因素分析
    uni_selected, _ = univariate_screening(X, y)
    for f in uni_selected:
        if f in votes:
            votes[f] += 1
    
    # Method 2: LASSO
    lasso_selected, _ = lasso_selection(X, y)
    for f in lasso_selected:
        if f in votes:
            votes[f] += 1
    
    # Method 3: RF排列重要性
    rf_selected, _ = rf_permutation_importance(X, y)
    for f in rf_selected:
        if f in votes:
            votes[f] += 1
    
    # Method 4: XGBoost排列重要性
    xgb_selected, _ = xgb_permutation_importance(X, y)
    for f in xgb_selected:
        if f in votes:
            votes[f] += 1
    
    # 保留得票>=threshold的特征
    final_selected = [f for f, v in votes.items() if v >= vote_threshold]
    
    # VIF筛选（迭代删除高VIF的非核心特征）
    X_selected = X[final_selected]
    vif_df, high_vif = calculate_vif(X_selected)
    
    # 优先删除非核心特征
    to_remove = []
    for f in high_vif:
        if core_features and f not in core_features:
            to_remove.append(f)
    
    final_features = [f for f in final_selected if f not in to_remove]
    
    return final_features, votes, vif_df

if __name__ == '__main__':
    pass
