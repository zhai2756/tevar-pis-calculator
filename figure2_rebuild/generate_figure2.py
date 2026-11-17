# -*- coding: utf-8 -*-
"""
Figure 2. Feature selection process (TEVAR PIS prediction model)
================================================================
A: LASSO regularization path vs log10(C); C_min (max 10-fold CV AUC) and
   C_1se (1-SE rule, selected); features non-zero at C_1se highlighted.
   Inset: 10-fold CV AUC vs log10(C).
B: Random-forest permutation importance (top 15), mean AUC decrease +/- SD
   (20 repeats). RF = tuned stage-3 config (n=99, depth=5, min_leaf=5,
   class_weight=balanced, seed 42).
C: XGBoost permutation importance (top 20) from the locked stage2_v3 artifact
   permutation_importance.xlsx (the importance table used in the stage2 audit).
D: Four-method voting Venn: FDR (BH q<0.05), LASSO (non-zero at C_1se),
   RF (importance > 0), XGB (importance > 0; audit convention). The final
   9 features each receive >= 2 votes.
Outputs: Figure2_feature_selection.png (600 dpi) / .pdf (vector), vote_table.csv
"""
import json
import os

import numpy as np
import pandas as pd

REBUILD = os.path.dirname(os.path.abspath(__file__))

FINAL9 = ["num__age", "num__operation_time_min", "num__contrast_ml",
          "cat_ordinal__tbad_stratum", "num__neut_pre", "num__aptt_pre",
          "num__fib_pre", "num__neut_pre_sq", "num__plt_pre"]

DISPLAY = {
    "num__age": "Age", "num__age_sq": "Age$^2$", "num__age_x_neut": "Age$\\times$Neut",
    "num__alb_pre": "Albumin", "num__aptt_pre": "APTT", "num__blood_loss_ml": "Blood loss",
    "num__bmi": "BMI", "num__bmi_sq": "BMI$^2$", "num__bmi_x_age": "BMI$\\times$Age",
    "num__bnp_pre": "BNP", "num__comorbidity_count": "Comorbidity count",
    "num__contrast_ml": "Contrast volume", "num__cre_pre": "Creatinine",
    "num__crp_pre": "CRP", "num__d_dimer_pre": "D-dimer", "num__far": "FAR",
    "num__fib_pre": "Fibrinogen", "num__height_cm": "Height", "num__hgb_pre": "Hemoglobin",
    "num__inflammation_score": "Inflammation score", "num__log_bnp_pre": "log(BNP)",
    "num__log_cre_pre": "log(Creatinine)", "num__log_crp_pre": "log(CRP)",
    "num__log_d_dimer_pre": "log(D-dimer)", "num__lym_pre": "Lymphocytes",
    "num__mono_pre": "Monocytes", "num__neut_pre": "Neutrophils",
    "num__neut_pre_sq": "Neutrophils$^2$", "num__nlr": "NLR", "num__nlr_sq": "NLR$^2$",
    "num__operation_time_min": "Operation time", "num__plt_pre": "Platelets",
    "num__pni": "PNI", "num__pnr": "PNR", "num__sii": "SII", "num__siri": "SIRI",
    "num__siri_sq": "SIRI$^2$", "num__siri_x_nlr": "SIRI$\\times$NLR",
    "num__stent_count": "Stent count", "num__surgery_complexity": "Surgical complexity",
    "num__tbad_x_age": "TBAD$\\times$Age", "num__weight_kg": "Weight",
    "cat_ordinal__tbad_stage": "TBAD stage", "cat_ordinal__tbad_stratum": "TBAD risk stratum",
    "binary__alcohol": "Alcohol use", "binary__cad": "CAD", "binary__dm": "Diabetes",
    "binary__gender": "Male sex", "binary__hypertension": "Hypertension",
    "binary__pre_rupture_hypoperfusion": "Rupture/hypoperfusion",
    "binary__prior_aortic_surgery": "Prior aortic surgery",
    "binary__refractory_pain_hypertension": "Refractory pain/HTN",
    "binary__smoking": "Smoking",
}

METHOD_COLORS = {"FDR": "#0072B2", "LASSO": "#E69F00", "RF": "#009E73", "XGB": "#CC79A7"}
HIGHLIGHT = "#D55E00"
OTHER = "#9E9E9E"

# 4-ellipse Venn geometry (verified: all 8 observed regions have area)
ELL = {
    "FDR":   (-0.50, 0.0, 1.30, 2.05),
    "RF":    (0.50, 0.0, 1.30, 2.05),
    "LASSO": (0.0, 0.50, 2.05, 0.95),
    "XGB":   (0.0, -0.50, 2.05, 0.95),
}


def load_inputs():
    uni = pd.read_csv(os.path.join(REBUILD, "univariate_fdr.csv"))
    rf = pd.read_csv(os.path.join(REBUILD, "rf_perm_importance.csv"))
    xgb = pd.read_csv(os.path.join(REBUILD, "xgb_perm_importance.csv")) \
             .sort_values("mean", ascending=False).reset_index(drop=True)
    with open(os.path.join(REBUILD, "selection_sets.json"), encoding="utf-8") as f:
        meta = json.load(f)
    npz = np.load(os.path.join(REBUILD, "lasso_path.npz"))
    return uni, rf, xgb, meta, npz


def build_sets(uni, rf, xgb, meta):
    S = {
        "FDR": set(uni.loc[uni["q_value"] < 0.05, "feature"]),
        "LASSO": set(meta["nz_at_1se"].keys()),
        "RF": set(rf.loc[rf["mean"] > 0, "feature"]),
        "XGB": set(xgb.loc[xgb["mean"] > 0, "feature"]),
    }
    votes = {}
    for f in sorted(set().union(*S.values())):
        votes[f] = {"votes": sum(f in s for s in S.values()),
                    "methods": [m for m in ("FDR", "LASSO", "RF", "XGB") if f in S[m]]}
    return S, votes


def style_ax(ax, xgrid=False, ygrid=False):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_facecolor("white")
    if xgrid:
        ax.xaxis.grid(True, color="0.90", lw=0.7)
        ax.set_axisbelow(True)
    if ygrid:
        ax.yaxis.grid(True, color="0.90", lw=0.7)
        ax.set_axisbelow(True)
    ax.tick_params(labelsize=8)


def panel_a(fig, gs, npz, meta):
    """Panel A as a stacked duo: top = 10-fold CV AUC, bottom = coefficient path,
    shared flipped x-axis (strong penalty on the right), classic glmnet style."""
    Cs, coefs = npz["Cs"], npz["coefs"]
    auc_mean, auc_se = npz["auc_mean"], npz["auc_se"]
    C_min, C_1se = float(npz["C_min"]), float(npz["C_1se"])
    logC = np.log10(Cs)
    nz = meta["nz_at_1se"]
    order = meta["feature_order"]
    idx = {f: order.index(f) for f in nz}

    sub = gs.subgridspec(2, 1, height_ratios=[0.34, 0.66], hspace=0.06)
    ax1 = fig.add_subplot(sub[0])
    ax2 = fig.add_subplot(sub[1], sharex=ax1)

    # ---- top: CV AUC ----
    ax1.plot(logC, auc_mean, color="#333333", lw=1.3)
    ax1.fill_between(logC, auc_mean - auc_se, auc_mean + auc_se,
                     color="#999999", alpha=0.35, lw=0)
    ax1.axvline(np.log10(C_min), color="black", ls="--", lw=1.2)
    ax1.axvline(np.log10(C_1se), color="#CC0000", ls="-.", lw=1.2)
    ax1.set_ylim(max(0.55, (auc_mean - auc_se).min() - 0.02),
                 (auc_mean + auc_se).max() + 0.03)
    ax1.set_ylabel("10-fold\nCV AUC", fontsize=8)
    y1top = ax1.get_ylim()[1]
    ax1.annotate("$C_{min}$ = 0.059\n(max CV AUC)", xy=(np.log10(C_min), y1top),
                 xytext=(-4, -2), textcoords="offset points", fontsize=7.5,
                 ha="right", va="top", color="black")
    import matplotlib.pyplot as _plt
    _plt.setp(ax1.get_xticklabels(), visible=False)
    style_ax(ax1, ygrid=True)
    ax1.tick_params(labelsize=7.5)

    # ---- bottom: coefficient path ----
    for j in range(coefs.shape[1]):
        ax2.plot(logC, coefs[:, j], color="0.80", lw=0.8, alpha=0.6, zorder=1)
    hl_colors = {"cat_ordinal__tbad_stratum": "#0072B2",
                 "num__contrast_ml": "#D55E00",
                 "num__siri_sq": "#009E73"}
    for f in nz:
        j = idx[f]
        ax2.plot(logC, coefs[:, j], color=hl_colors[f], lw=2.0, zorder=3)
    ax2.invert_xaxis()   # shared x: both panels flip (strong penalty -> right)
    label_dy = {"cat_ordinal__tbad_stratum": 0.0, "num__contrast_ml": -0.10,
                "num__siri_sq": 0.07}
    for f in nz:
        j = idx[f]
        ax2.annotate(DISPLAY[f], xy=(logC[-1], coefs[-1, j] + label_dy[f]),
                     xytext=(-8, 0), textcoords="offset points",
                     fontsize=8, color=hl_colors[f], va="center", ha="right",
                     fontweight="bold")
    ymin, ymax = ax2.get_ylim()
    ax2.set_ylim(ymin - 0.05, ymax + 0.08)
    ax2.set_xlim(logC[-1] + 1.30, logC[0] - 0.10)
    ax2.axvline(np.log10(C_min), color="black", ls="--", lw=1.2, zorder=2)
    ax2.axvline(np.log10(C_1se), color="#CC0000", ls="-.", lw=1.2, zorder=2)
    ax2.annotate("$C_{1se}$ = 0.0070\n(1-SE rule, selected)",
                 xy=(np.log10(C_1se), ax2.get_ylim()[1]),
                 xytext=(-5, -3), textcoords="offset points", fontsize=7.5,
                 ha="right", va="top", color="#CC0000", zorder=4)
    ax2.set_xlabel("log$_{10}$(C)   (stronger penalty $\\rightarrow$)", fontsize=9)
    ax2.set_ylabel("Standardized coefficient", fontsize=9)
    style_ax(ax2, ygrid=True)
    return ax1


def panel_importance(ax, df, top_n, title):
    d = df.head(top_n).iloc[::-1]
    colors = [HIGHLIGHT if f in FINAL9 else OTHER for f in d["feature"]]
    labels = [DISPLAY.get(f, f) for f in d["feature"]]
    ypos = np.arange(len(d))
    ax.barh(ypos, d["mean"], xerr=d["std"], color=colors, height=0.68,
            error_kw=dict(ecolor="black", lw=1.0, capsize=2.2), zorder=3)
    ax.set_yticks(ypos, labels, fontsize=8)
    span = max((df.head(top_n)["mean"] + df.head(top_n)["std"]).max(), 1e-6)
    for i, (m, s) in enumerate(zip(d["mean"], d["std"])):
        xt = m + s + span * 0.015 if m >= 0 else m + s + span * 0.015
        ax.text(xt, i, f"{m:.3f}", va="center", fontsize=6.8, color="#333333")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Mean decrease in AUC $\\pm$ SD", fontsize=9)
    ax.set_title(title, fontsize=9.5, pad=6)
    xmax = (d["mean"] + d["std"]).max()
    xmin = min(0.0, (d["mean"] - d["std"]).min())
    ax.set_xlim(xmin - span * 0.06, xmax + span * 0.17)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=HIGHLIGHT, label="Final model feature"),
                       Patch(color=OTHER, label="Not selected")],
              fontsize=7.5, loc="lower right", frameon=False)
    style_ax(ax, xgrid=True)


def region_anchors():
    gx, gy = np.meshgrid(np.linspace(-3.35, 3.35, 480),
                         np.linspace(-3.35, 3.35, 480))
    masks, depth = {}, {}
    for m in ("FDR", "LASSO", "RF", "XGB"):
        cx, cy, a, b = ELL[m]
        v = ((gx - cx) / a) ** 2 + ((gy - cy) / b) ** 2
        masks[m] = v <= 1.0
        depth[m] = np.abs(v - 1.0)
    sig_map = np.zeros(gx.shape, dtype=int)
    for i, m in enumerate(("FDR", "LASSO", "RF", "XGB")):
        sig_map += masks[m].astype(int) << i
    min_depth = np.minimum.reduce(list(depth.values()))
    anchors = {}
    for sig in range(1, 16):
        sel = sig_map == sig
        if sel.any():
            idx = np.argmax(np.where(sel, min_depth, -1.0))
            iy, ix = np.unravel_index(idx, gx.shape)
            anchors[sig] = (float(gx[iy, ix]), float(gy[iy, ix]))
    return anchors


def panel_d(fig, gs, S, votes, plt):
    from matplotlib.patches import Ellipse
    sub = gs.subgridspec(1, 2, width_ratios=[1.12, 1.0], wspace=0.02)
    axv = fig.add_subplot(sub[0])
    axt = fig.add_subplot(sub[1])

    for m, (cx, cy, a, b) in ELL.items():
        axv.add_patch(Ellipse((cx, cy), 2 * a, 2 * b, facecolor=METHOD_COLORS[m],
                              edgecolor="none", alpha=0.20))
        axv.add_patch(Ellipse((cx, cy), 2 * a, 2 * b, facecolor="none",
                              edgecolor=METHOD_COLORS[m], lw=1.6))
    union = sorted(set().union(*S.values()))
    sig_count = {}
    for f in union:
        sig = sum((f in S[m]) << i for i, m in enumerate(("FDR", "LASSO", "RF", "XGB")))
        sig_count[sig] = sig_count.get(sig, 0) + 1
    anchors = region_anchors()
    for sig, cnt in sig_count.items():
        if sig in anchors:
            x, y = anchors[sig]
            axv.text(x, y, str(cnt), ha="center", va="center",
                     fontsize=9, color="#222222")
    axv.set_xlim(-3.55, 3.55)
    axv.set_ylim(-2.72, 2.72)
    axv.set_aspect("equal")
    axv.axis("off")
    handles = [plt.Line2D([0], [0], marker="o", color="none",
                          markerfacecolor=METHOD_COLORS[m], markeredgecolor=METHOD_COLORS[m],
                          alpha=0.7, markersize=8,
                          label=f"{m} (n={len(S[m])})") for m in ("FDR", "LASSO", "RF", "XGB")]
    axv.legend(handles=handles, fontsize=7.5, loc="lower center",
               bbox_to_anchor=(0.5, -0.10), ncol=4, frameon=False,
               handletextpad=0.15, columnspacing=0.9)

    axt.axis("off")
    axt.set_xlim(0, 1)
    axt.set_ylim(0, 1)
    axt.text(0.0, 1.00, "Final 9 features (all $\\geq$2 votes)",
             fontsize=9.5, fontweight="bold", va="top")
    cols = {"FDR": 0.60, "LASSO": 0.725, "RF": 0.85, "XGB": 0.975}
    hdr = {"FDR": "F", "LASSO": "L", "RF": "R", "XGB": "X"}
    for m, x in cols.items():
        axt.text(x, 0.915, hdr[m], fontsize=8.5, fontweight="bold",
                 ha="center", color=METHOD_COLORS[m])
    axt.plot([0.0, 1.0], [0.888, 0.888], color="#444444", lw=0.8)
    order = sorted(FINAL9, key=lambda f: (-votes[f]["votes"], DISPLAY[f]))
    y = 0.832
    for f in order:
        v = votes[f]
        axt.text(0.0, y, f"{DISPLAY[f]} ({v['votes']})", fontsize=8.4, va="center")
        for m, x in cols.items():
            on = f in S[m]
            axt.scatter([x], [y], s=54,
                        facecolor=METHOD_COLORS[m] if on else "white",
                        edgecolor=METHOD_COLORS[m] if on else "#BBBBBB",
                        linewidth=1.2, zorder=3)
        y -= 0.0815
    axt.text(0.0, y - 0.015,
             "F = FDR-corrected univariate ($q$ < 0.05)\n"
             "L = LASSO non-zero at $C_{1se}$\n"
             "R = RF importance > 0\n"
             "X = XGBoost importance > 0",
             fontsize=7.2, va="top", color="#444444", linespacing=1.6)
    return axv


def main(ctx=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    if ctx is not None:
        from daimon_runtime import setup_plot
        setup_plot(ctx)

    for fp in (r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\arialbd.ttf",
               r"C:\Windows\Fonts\ariali.ttf", r"C:\Windows\Fonts\arialbi.ttf"):
        if os.path.exists(fp):
            try:
                font_manager.fontManager.addfont(fp)
            except Exception:
                pass
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 8.5,
        "axes.linewidth": 0.9,
        "axes.facecolor": "white",
        "axes.grid": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    uni, rf, xgb, meta, npz = load_inputs()
    S, votes = build_sets(uni, rf, xgb, meta)
    check = {f: votes[f]["votes"] for f in FINAL9}
    all_ok = all(v >= 2 for v in check.values())

    fig = plt.figure(figsize=(13.2, 10.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.12],
                          hspace=0.32, wspace=0.20,
                          left=0.115, right=0.985, top=0.94, bottom=0.065)
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axA = panel_a(fig, gs[0, 0], npz, meta)
    panel_importance(axB, rf, 15, "Random forest permutation importance (top 15)")
    panel_importance(axC, xgb, 20, "XGBoost permutation importance (top 20)")
    axDv = panel_d(fig, gs[1, 1], S, votes, plt)

    for ax, letter in [(axA, "A"), (axB, "B"), (axC, "C"), (axDv, "D")]:
        ax.text(-0.13, 1.05, letter, transform=ax.transAxes,
                fontsize=16, fontweight="bold", va="top", ha="left")

    png = os.path.join(REBUILD, "Figure2_feature_selection.png")
    pdf = os.path.join(REBUILD, "Figure2_feature_selection.pdf")
    fig.savefig(png, dpi=600, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)

    rows = [{"feature": f, "display": DISPLAY[f], "votes": votes[f]["votes"],
             **{m: (f in S[m]) for m in ("FDR", "LASSO", "RF", "XGB")}} for f in FINAL9]
    pd.DataFrame(rows).to_csv(os.path.join(REBUILD, "vote_table.csv"), index=False)
    report = {
        "all9_ge2_votes": all_ok,
        "votes": {DISPLAY[f]: check[f] for f in FINAL9},
        "set_sizes": {m: len(S[m]) for m in S},
        "outputs": {"png": "Figure2_feature_selection.png", "pdf": "Figure2_feature_selection.pdf"},
    }
    with open(os.path.join(REBUILD, "figure2_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    return report


if __name__ == "__main__":
    main()
