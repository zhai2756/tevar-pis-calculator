"""
Phase 3: Model Building & Hyperparameter Optimization
模型训练与超参数优化：嵌套交叉验证
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42

def get_model_configs():
    """定义8种模型及其超参数搜索空间"""
    
    configs = {
        'LogisticRegression': {
            'model': LogisticRegression,
            'params': {
                'C': np.logspace(-3, 1, 50),
                'max_iter': [1000],
                'solver': ['lbfgs', 'saga'],
                'penalty': ['l2']
            }
        },
        'LASSO': {
            'model': LogisticRegression,
            'params': {
                'C': np.logspace(-3, 1, 50),
                'max_iter': [10000],
                'solver': ['saga'],
                'penalty': ['l1']
            }
        },
        'Ridge': {
            'model': LogisticRegression,
            'params': {
                'C': np.logspace(-3, 1, 50),
                'max_iter': [1000],
                'solver': ['lbfgs'],
                'penalty': ['l2']
            }
        },
        'ElasticNet': {
            'model': LogisticRegression,
            'params': {
                'C': np.logspace(-3, 1, 50),
                'l1_ratio': np.linspace(0.1, 0.9, 20),
                'max_iter': [10000],
                'solver': ['saga'],
                'penalty': ['elasticnet']
            }
        },
        'RandomForest': {
            'model': RandomForestClassifier,
            'params': {
                'n_estimators': [50, 100, 150, 200],
                'max_depth': [3, 5, 7, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 5],
                'max_features': ['sqrt', 'log2']
            }
        },
        'XGBoost': {
            'model': XGBClassifier,
            'params': {
                'n_estimators': [50, 100, 150, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': np.logspace(-3, -1, 20),
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
        },
        'LightGBM': {
            'model': LGBMClassifier,
            'params': {
                'n_estimators': [50, 100, 150, 200],
                'max_depth': [3, 5, 7, -1],
                'learning_rate': np.logspace(-3, -1, 20),
                'num_leaves': [20, 31, 50],
                'verbosity': [-1]
            }
        },
        'SVM_RBF': {
            'model': SVC,
            'params': {
                'C': np.logspace(-3, 3, 50),
                'gamma': np.logspace(-4, 1, 30),
                'kernel': ['rbf'],
                'probability': [True]
            }
        }
    }
    return configs

def nested_cv_evaluation(X, y, model_name, model_config, 
                         outer_cv=5, inner_cv=5, n_iter=100):
    """
    嵌套交叉验证评估
    外层CV: 模型评估
    内层CV: 超参数优化 (RandomizedSearchCV)
    """
    outer_skf = StratifiedKFold(n_splits=outer_cv, shuffle=True, random_state=RANDOM_STATE)
    
    outer_scores = []
    best_params_per_fold = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(outer_skf.split(X, y)):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # 内层CV超参数搜索
        model_class = model_config['model']
        search = RandomizedSearchCV(
            estimator=model_class(random_state=RANDOM_STATE) if 'random_state' in model_class.__init__.__code__.co_varnames else model_class(),
            param_distributions=model_config['params'],
            n_iter=n_iter,
            cv=StratifiedKFold(n_splits=inner_cv, shuffle=True, random_state=RANDOM_STATE),
            scoring='roc_auc',
            n_jobs=-1,
            random_state=RANDOM_STATE
        )
        
        # 对于需要random_state的模型
        if model_name in ['LogisticRegression', 'LASSO', 'Ridge', 'ElasticNet']:
            search = RandomizedSearchCV(
                estimator=LogisticRegression(random_state=RANDOM_STATE),
                param_distributions=model_config['params'],
                n_iter=n_iter, cv=inner_cv, scoring='roc_auc', n_jobs=-1, random_state=RANDOM_STATE
            )
        elif model_name == 'RandomForest':
            search = RandomizedSearchCV(
                estimator=RandomForestClassifier(random_state=RANDOM_STATE),
                param_distributions=model_config['params'],
                n_iter=n_iter, cv=inner_cv, scoring='roc_auc', n_jobs=-1, random_state=RANDOM_STATE
            )
        elif model_name == 'XGBoost':
            search = RandomizedSearchCV(
                estimator=XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss'),
                param_distributions=model_config['params'],
                n_iter=n_iter, cv=inner_cv, scoring='roc_auc', n_jobs=-1, random_state=RANDOM_STATE
            )
        elif model_name == 'LightGBM':
            search = RandomizedSearchCV(
                estimator=LGBMClassifier(random_state=RANDOM_STATE, verbosity=-1),
                param_distributions=model_config['params'],
                n_iter=n_iter, cv=inner_cv, scoring='roc_auc', n_jobs=-1, random_state=RANDOM_STATE
            )
        elif model_name == 'SVM_RBF':
            search = RandomizedSearchCV(
                estimator=SVC(probability=True, random_state=RANDOM_STATE),
                param_distributions=model_config['params'],
                n_iter=n_iter, cv=inner_cv, scoring='roc_auc', n_jobs=-1, random_state=RANDOM_STATE
            )
        
        search.fit(X_train_fold, y_train_fold)
        
        # 评估
        best_model = search.best_estimator_
        y_pred_proba = best_model.predict_proba(X_val_fold)[:, 1]
        auc = roc_auc_score(y_val_fold, y_pred_proba)
        
        outer_scores.append(auc)
        best_params_per_fold.append(search.best_params_)
        
        print(f"  Fold {fold_idx+1}: AUC={auc:.3f}, Best params={search.best_params_}")
    
    mean_auc = np.mean(outer_scores)
    std_auc = np.std(outer_scores)
    
    return {
        'mean_auc': mean_auc,
        'std_auc': std_auc,
        'fold_scores': outer_scores,
        'best_params_per_fold': best_params_per_fold
    }

def compare_all_models(X, y, selected_features):
    """比较所有8种模型"""
    configs = get_model_configs()
    results = {}
    
    for model_name, config in configs.items():
        print(f"\nEvaluating {model_name}...")
        result = nested_cv_evaluation(X, y, model_name, config)
        results[model_name] = result
        print(f"{model_name}: Mean AUC={result['mean_auc']:.3f} ± {result['std_auc']:.3f}")
    
    return results

if __name__ == '__main__':
    pass
