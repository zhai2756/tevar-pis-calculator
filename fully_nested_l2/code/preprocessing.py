# -*- coding: utf-8 -*-
"""
完全嵌套 L2 敏感性分析 —— 模块 preprocessing.py
==================================================
在每个训练子集内: 中位数/众数插补 -> 临床范围截断 -> 确定性特征工程 -> RobustScaler
(中心=子集中位数, 尺度=子集IQR; IQR=0 时尺度=1.0) -> 固定编码。
验证折/测试折一律用训练子集参数转换。
所有定义与 stage1_optimized 归档产物逐字节核验一致 (见 config.json 中的来源说明)。
"""
import json
import numpy as np
import pandas as pd

BASE = __file__.rsplit("\\", 2)[0] if "\\" in __file__ else __file__.rsplit("/", 2)[0]
CONFIG = json.load(open(__file__.replace("preprocessing.py", "config.json"), encoding="utf-8"))
BOUNDS = {k: tuple(v) for k, v in CONFIG["clinical_bounds_applied"].items()}

CONTINUOUS_BASE = [
    "age", "bmi", "weight_kg", "height_cm", "hgb_pre", "plt_pre", "mono_pre",
    "neut_pre", "lym_pre", "fib_pre", "siri", "sii", "far", "nlr", "pnr",
    "alb_pre", "aptt_pre", "d_dimer_pre", "cre_pre", "bnp_pre", "crp_pre",
    "operation_time_min", "blood_loss_ml", "contrast_ml", "stent_count",
]
ORDINAL = ["tbad_stage", "tbad_stratum"]
BINARY = [
    "gender", "smoking", "alcohol", "cad", "hypertension", "dm",
    "pre_rupture_hypoperfusion", "refractory_pain_hypertension",
    "prior_aortic_surgery",
]
# 插补所需、参与工程但不进入 53 候选池的辅助变量
AUX = ["wbc_pre"]

def _engineered_cont_names():
    direct = [f"num__{c}" for c in CONTINUOUS_BASE if c not in ("siri", "sii", "far", "nlr", "pnr")]
    direct += ["num__siri", "num__sii", "num__far", "num__nlr", "num__pnr"]
    derived = ["num__age_sq", "num__bmi_sq", "num__neut_pre_sq", "num__nlr_sq",
               "num__siri_sq", "num__log_crp_pre", "num__log_bnp_pre",
               "num__log_cre_pre", "num__log_d_dimer_pre", "num__age_x_neut",
               "num__bmi_x_age", "num__tbad_x_age", "num__siri_x_nlr",
               "num__comorbidity_count", "num__inflammation_score", "num__pni",
               "num__surgery_complexity"]
    return direct + derived


CONT_FEATURE_NAMES = _engineered_cont_names()
ORD_FEATURE_NAMES = ["cat_ordinal__tbad_stage", "cat_ordinal__tbad_stratum"]
BIN_FEATURE_NAMES = [f"binary__{c}" for c in BINARY]
ALL_53 = CONT_FEATURE_NAMES + ORD_FEATURE_NAMES + BIN_FEATURE_NAMES
assert len(ALL_53) == 53, len(ALL_53)

SQUARE_OF = {"num__age_sq": "num__age", "num__bmi_sq": "num__bmi",
             "num__neut_pre_sq": "num__neut_pre", "num__nlr_sq": "num__nlr",
             "num__siri_sq": "num__siri"}
DERIVED_FEATURES = set(ALL_53) - set(
    [f"num__{c}" for c in CONTINUOUS_BASE] + ORD_FEATURE_NAMES + BIN_FEATURE_NAMES)


def _clip(s, name):
    if name in BOUNDS:
        lo, hi = BOUNDS[name]
        return s.clip(lo, hi)
    return s


def fit_preprocessor(raw_train: pd.DataFrame):
    """仅用训练子集拟合插补统计量。返回 params dict。"""
    params = {"medians": {}, "modes": {}}
    for c in CONTINUOUS_BASE + AUX:
        params["medians"][c] = float(raw_train[c].median())
    for c in ORDINAL + BINARY:
        params["modes"][c] = float(raw_train[c].mode().iloc[0])
    return params


def impute(raw: pd.DataFrame, params) -> pd.DataFrame:
    df = raw.copy()
    for c in CONTINUOUS_BASE + AUX:
        df[c] = df[c].fillna(params["medians"][c])
    for c in ORDINAL + BINARY:
        df[c] = df[c].fillna(params["modes"][c])
    return df


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """截断 + 确定性特征工程（在插补后的数据上）。"""
    g = {}
    for c in CONTINUOUS_BASE + AUX:
        g[c] = _clip(df[c].astype(float), c)
    lym0 = g["lym_pre"].replace(0, 0.01)
    neut0 = g["neut_pre"].replace(0, 0.01)
    alb0 = g["alb_pre"].replace(0, 0.01)
    nlr_r = g["neut_pre"] / lym0
    pnr_r = g["plt_pre"] / neut0
    far_r = g["fib_pre"] / alb0
    siri_r = g["neut_pre"] * g["mono_pre"].replace(0, 0.01) / lym0
    sii_r = g["plt_pre"] * g["neut_pre"] / lym0

    out = pd.DataFrame(index=df.index)
    for c in CONTINUOUS_BASE:
        if c in ("siri", "sii", "far", "nlr", "pnr"):
            continue
        out[f"num__{c}"] = g[c]
    out["num__siri"] = siri_r
    out["num__sii"] = sii_r
    out["num__far"] = far_r
    out["num__nlr"] = nlr_r
    out["num__pnr"] = pnr_r
    out["num__age_sq"] = g["age"] ** 2
    out["num__bmi_sq"] = g["bmi"] ** 2
    out["num__neut_pre_sq"] = g["neut_pre"] ** 2
    out["num__nlr_sq"] = nlr_r ** 2
    out["num__siri_sq"] = siri_r ** 2
    out["num__log_crp_pre"] = np.log1p(g["crp_pre"])
    out["num__log_bnp_pre"] = np.log1p(g["bnp_pre"])
    out["num__log_cre_pre"] = np.log1p(g["cre_pre"])
    out["num__log_d_dimer_pre"] = np.log1p(g["d_dimer_pre"])
    out["num__age_x_neut"] = g["age"] * g["neut_pre"]
    out["num__bmi_x_age"] = g["bmi"] * g["age"]
    out["num__tbad_x_age"] = df["tbad_stratum"].astype(float) * g["age"]  # 临床编码 1/2/3
    out["num__siri_x_nlr"] = siri_r * nlr_r
    out["num__comorbidity_count"] = (df["hypertension"] + df["cad"] + df["dm"] +
                                     df["smoking"] + df["alcohol"]).astype(float)
    out["num__inflammation_score"] = g["crp_pre"] + g["wbc_pre"]
    out["num__surgery_complexity"] = g["operation_time_min"] / 60.0 + g["stent_count"]
    out["num__pni"] = g["alb_pre"] + 5.0 * g["lym_pre"]
    # 编码
    out["cat_ordinal__tbad_stage"] = df["tbad_stage"].astype(float) - 1.0
    out["cat_ordinal__tbad_stratum"] = df["tbad_stratum"].astype(float) - 1.0
    for c in BINARY:
        out[f"binary__{c}"] = df[c].astype(float)
    return out[ALL_53]


def fit_scaler(X_train_eng: pd.DataFrame):
    """RobustScaler 参数：中心=训练子集中位数，尺度=IQR（0 时取 1.0）。"""
    center = X_train_eng[CONT_FEATURE_NAMES].median()
    q = X_train_eng[CONT_FEATURE_NAMES].quantile([0.25, 0.75])
    scale = (q.iloc[1] - q.iloc[0]).replace(0, 1.0)
    return {"center": center.to_dict(), "scale": scale.to_dict()}


def apply_scaler(X_eng: pd.DataFrame, scaler) -> pd.DataFrame:
    X = X_eng.copy()
    for c in CONT_FEATURE_NAMES:
        X[c] = (X[c] - scaler["center"][c]) / scaler["scale"][c]
    return X


def fit_transform(raw_train: pd.DataFrame):
    """训练子集：拟合插补+scaler，返回 (X_processed, prep_params, scaler_params)。"""
    params = fit_preprocessor(raw_train)
    eng = engineer(impute(raw_train, params))
    scaler = fit_scaler(eng)
    X = apply_scaler(eng, scaler)
    return X, {"impute": params, "scaler": scaler}


def transform(raw_other: pd.DataFrame, prep) -> pd.DataFrame:
    eng = engineer(impute(raw_other, prep["impute"]))
    return apply_scaler(eng, prep["scaler"])
