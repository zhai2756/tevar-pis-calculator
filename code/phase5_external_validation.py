"""
Phase 5: External Validation
外部验证：在独立队列上评估模型性能
"""
import pandas as pd
import numpy as np
from phase4_final_model import evaluate_model

def external_validation(model, X_ext, y_ext, preprocessor=None):
    """
    外部验证
    注意：外部验证数据必须经过与训练集相同的预处理流程
    """
    # 如果提供了preprocessor，先转换数据
    if preprocessor is not None:
        X_ext_processed = preprocessor.transform(X_ext)
    else:
        X_ext_processed = X_ext
    
    # 评估
    metrics = evaluate_model(model, X_ext_processed, y_ext, dataset_name='External Validation')
    
    # 计算泛化差距
    # (内部验证AUC - 外部验证AUC)
    
    return metrics

if __name__ == '__main__':
    pass
