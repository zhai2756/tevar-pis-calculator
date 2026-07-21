"""
Phase 1: Data Preprocessing
数据预处理阶段：缺失值处理、特征编码、标准化
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42

def load_data(train_path, ext_path):
    """加载训练和外部验证数据"""
    train_df = pd.read_csv(train_path)
    ext_df = pd.read_csv(ext_path)
    return train_df, ext_df

def identify_variable_types(df, target_col='pis'):
    """识别变量类型"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != target_col]
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    return numeric_cols, categorical_cols

def create_preprocessing_pipeline(numeric_cols, categorical_cols, ordinal_cols=None):
    """创建预处理pipeline"""
    # 数值型变量：中位数填充 + 标准化
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # 分类型变量：众数填充 + OneHot
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
    ])
    
    # 有序分类变量：OrdinalEncoder
    ordinal_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ordinal', OrdinalEncoder())
    ])
    
    transformers = [
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
    
    if ordinal_cols:
        transformers.append(('ord', ordinal_transformer, ordinal_cols))
    
    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
    return preprocessor

def preprocess_data(train_df, ext_df, target_col='pis', test_size=0.2):
    """
    完整预处理流程
    1. 划分训练/内部验证集 (80:20 stratified)
    2. 在训练集上fit预处理pipeline
    3. 用transform()应用到验证集和外部验证集（防止数据泄漏）
    """
    # 分离特征和标签
    X = train_df.drop(columns=[target_col])
    y = train_df[target_col]
    
    # 划分训练/内部验证
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )
    
    # 识别变量类型
    numeric_cols, categorical_cols = identify_variable_types(train_df, target_col)
    
    # 创建并fit预处理pipeline（仅在训练集上fit）
    preprocessor = create_preprocessing_pipeline(numeric_cols, categorical_cols)
    preprocessor.fit(X_train)
    
    # 转换所有数据集
    X_train_processed = preprocessor.transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_ext_processed = preprocessor.transform(ext_df.drop(columns=[target_col]))
    y_ext = ext_df[target_col]
    
    # 获取特征名称
    feature_names = []
    for name, transformer, cols in preprocessor.transformers_:
        if name == 'num':
            feature_names.extend([f'num__{c}' for c in cols])
        elif name == 'cat':
            ohe = transformer.named_steps['onehot']
            cat_features = ohe.get_feature_names_out(cols)
            feature_names.extend(cat_features.tolist())
    
    # 保存
    results = {
        'X_train': X_train_processed, 'y_train': y_train,
        'X_val': X_val_processed, 'y_val': y_val,
        'X_ext': X_ext_processed, 'y_ext': y_ext,
        'feature_names': feature_names,
        'preprocessor': preprocessor
    }
    
    return results

if __name__ == '__main__':
    # 使用示例
    # train_df = pd.read_csv('data/train.csv')
    # ext_df = pd.read_csv('data/external.csv')
    # results = preprocess_data(train_df, ext_df)
    pass
