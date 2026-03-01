"""
Approach C evaluation + model export.

Runs Personalized Outlier Detection (Approach C) in LOPO protocol,
compares against saved Approach A/B results, generates all figures,
and exports models + metrics.
"""

import sys, os, time, json, warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from pathlib import Path
from sklearn.metrics import f1_score, roc_auc_score
import joblib

from src.data.fog_star_loader import FoGStarDataset
from src.data.windowing import create_windowed_dataset
from src.data.features import extract_batch_features
from src.models.outlier_detector import PersonalizedOutlierEnsemble
from src.utils.metrics import compute_all_metrics, event_level_metrics

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("results/figures")
MODELS_DIR  = Path("results/models")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── Plotting style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 11,
    "axes.titlesize": 13, "axes.labelsize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 10, "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
})

COL_A = "#3498db"
COL_B = "#e74c3c"
COL_C = "#27ae60"


# ════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("LOADING DATA")
print("=" * 70)
ds       = FoGStarDataset("datasets/fog").load()
windowed = create_windowed_dataset(ds, window_seconds=2.0, overlap=0.5)
print(f"  Windows: {len(windowed.labels):,}  FoG: {windowed.labels.sum():,} ({windowed.labels.mean()*100:.1f}%)")

print("\nExtracting features (one-time)...")
t0 = time.time()
all_feat = extract_batch_features(windowed.windows)
all_feat = np.nan_to_num(all_feat, nan=0.0, posinf=0.0, neginf=0.0)
print(f"  Done in {time.time()-t0:.1f}s  shape={all_feat.shape}")


# ════════════════════════════════════════════════════════════════════════════
#  APPROACH C: PERSONALIZED OUTLIER DETECTION (LOPO)
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("APPROACH C: PERSONALIZED OUTLIER DETECTION (LOPO)")
print("=" * 70)

# n_seed = number of confirmed NON-FoG windows from the test patient
# used to personalize the normal gait model.
# 0  = pure population baseline (no patient data needed)
# 10 = 10 normal windows (~20 sec walking)
# 20 = ~40 sec, 50 = ~100 sec
n_seed_levels = [0, 10, 20, 50]

results_c = {}

for fold_idx, test_sid in enumerate(ds.subject_ids):
    test_mask  = windowed.subject_ids == test_sid
    train_mask = ~test_mask

    test_labels  = windowed.labels[test_mask]
    train_labels = windowed.labels[train_mask]

    if test_labels.sum() == 0:
        print(f"  [{fold_idx+1}/22] Patient {test_sid}: no FoG, skipping")
        continue

    print(f"\n[{fold_idx+1}/22] Patient {test_sid}  "
          f"({test_labels.sum()} FoG / {len(test_labels)} windows)")

    test_feat    = all_feat[test_mask]
    train_feat   = all_feat[train_mask]
    test_windows = windowed.windows[test_mask]
    train_windows= windowed.windows[train_mask]

    # Fit population-level normal model on ALL training non-FoG windows
    ensemble = PersonalizedOutlierEnsemble(fi_weight=0.5, n_estimators=200)
    ensemble.fit_population(train_feat, train_windows, train_labels)

    patient_results = {}
    for n_seed in n_seed_levels:
        preds, scores, threshold, eval_lbl = ensemble.score_and_predict(
            test_feat, test_windows, test_labels,
            n_seed_nonfog=n_seed, fs=60.0,
        )

        if len(eval_lbl) == 0 or eval_lbl.sum() == 0:
            continue

        metrics    = compute_all_metrics(eval_lbl, preds, scores)
        ev_metrics = event_level_metrics(eval_lbl, preds)
        patient_results[n_seed] = {**metrics, **ev_metrics, "threshold": threshold}

    results_c[test_sid] = patient_results
    best_f1 = max((v.get("f1", 0) for v in patient_results.values()), default=0)
    print(f"  Best F1={best_f1:.3f}  "
          f"(n=0: {patient_results.get(0,{}).get('f1',0):.3f}, "
          f"n=50: {patient_results.get(50,{}).get('f1',0):.3f})")

# Save Approach C results
def _np_convert(obj):
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    return obj

with open(RESULTS_DIR / "approach_c_results.json", "w") as f:
    json.dump(results_c, default=_np_convert, fp=f, indent=2)
print(f"\nSaved approach_c_results.json")


# ════════════════════════════════════════════════════════════════════════════
#  EXPORT POPULATION MODEL (re-train on full dataset)
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("EXPORTING MODELS")
print("=" * 70)

# Export Approach C population model trained on ALL data
print("  Training final Approach C model on full dataset...")
final_ensemble = PersonalizedOutlierEnsemble(fi_weight=0.5, n_estimators=200)
final_ensemble.fit_population(all_feat, windowed.windows, windowed.labels)
joblib.dump(final_ensemble, MODELS_DIR / "approach_c_outlier_detector.pkl")
print("  Saved approach_c_outlier_detector.pkl")

# Also save the scaler and feature names for inference
from src.data.features import get_feature_names
feat_names = get_feature_names(n_channels=24)
joblib.dump({"scaler": final_ensemble.pop_detector_.scaler_,
             "feature_names": feat_names,
             "fi_weight": 0.5,
             "fs": 60.0,
             "window_size": 120},
            MODELS_DIR / "approach_c_inference_bundle.pkl")
print("  Saved approach_c_inference_bundle.pkl")


# ════════════════════════════════════════════════════════════════════════════
#  COMBINED FIGURES
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("GENERATING FIGURES")
print("=" * 70)

# Load A and B results
with open(RESULTS_DIR / "approach_a_results.json") as f:
    results_a = json.load(f)
with open(RESULTS_DIR / "approach_b_results.json") as f:
    results_b = json.load(f)

def collect(results, n_adapt, metric):
    vals, sids = [], []
    for sid, res in results.items():
        key = str(n_adapt) if isinstance(n_adapt, int) else n_adapt
        r = res.get(key, {})
        if metric in r:
            vals.append(r[metric]); sids.append(sid)
    return np.array(vals), sids


# ── Figure 1: Per-patient F1 at n_adapt=0 ──────────────────────────────────
def fig_per_patient_f1():
    n = 0
    f1_a, sa = collect(results_a, n, "f1")
    f1_b, sb = collect(results_b, n, "f1")
    f1_c, sc = collect(results_c, n, "f1")
    common = sorted(set(sa) & set(sb) & set(sc), key=int)
    if not common: return

    fa = [f1_a[sa.index(s)] for s in common]
    fb = [f1_b[sb.index(s)] for s in common]
    fc = [f1_c[sc.index(s)] for s in common]

    x  = np.arange(len(common))
    w  = 0.26
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - w, fa, w, color=COL_A, label=f"A: Bayesian Ensemble  (mean={np.mean(fa):.3f})", alpha=0.9)
    ax.bar(x,     fb, w, color=COL_B, label=f"B: SSL+LoRA           (mean={np.mean(fb):.3f})", alpha=0.9)
    ax.bar(x + w, fc, w, color=COL_C, label=f"C: Outlier Detection  (mean={np.mean(fc):.3f})", alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels([str(s) for s in common], rotation=45, ha="right")
    ax.set_xlabel("Patient ID"); ax.set_ylabel("F1 Score")
    ax.set_title("Per-Patient F1 — No Adaptation Data (n=0)", fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right"); ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig1_per_patient_f1.png"); plt.close(fig)
    print("  fig1_per_patient_f1.png")

fig_per_patient_f1()


# ── Figure 2: Adaptation / personalisation curve ───────────────────────────
def fig_adaptation_curve():
    n_vals_a = [0, 10, 20, 50]
    n_vals_c = [0, 10, 20, 50]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics_info = [("f1","F1 Score"), ("sensitivity","Sensitivity"), ("specificity","Specificity")]

    for ax, (metric, title) in zip(axes, metrics_info):
        for label, results, color, marker, n_vals in [
            ("A: Bayesian Ensemble", results_a, COL_A, "o", n_vals_a),
            ("B: SSL+LoRA",          results_b, COL_B, "s", n_vals_a),
            ("C: Outlier Detection", results_c, COL_C, "^", n_vals_c),
        ]:
            means, stds, xs = [], [], []
            for n in n_vals:
                vals, _ = collect(results, n, metric)
                if len(vals):
                    means.append(np.mean(vals)); stds.append(np.std(vals)); xs.append(n)
            if means:
                means, stds = np.array(means), np.array(stds)
                ax.plot(xs, means, f"-{marker}", color=color, label=label, lw=2, ms=8)
                ax.fill_between(xs, means-stds, means+stds, alpha=0.12, color=color)

        # x-axis label differs: A/B = #adapt labelled, C = #normal windows
        ax.set_xlabel("Adaptation windows (A/B: labelled mix) | Normal seed (C: non-FoG)")
        ax.set_ylabel(title); ax.set_title(title)
        ax.legend(fontsize=9); ax.grid(True, linestyle="--", alpha=0.3)
        ax.set_ylim(0, 1.05)

    fig.suptitle("Performance vs. Patient-Specific Data Available", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig2_adaptation_curve.png"); plt.close(fig)
    print("  fig2_adaptation_curve.png")

fig_adaptation_curve()


# ── Figure 3: Box plots ─────────────────────────────────────────────────────
def fig_boxplots():
    metrics = [("f1","F1 Score"), ("sensitivity","Sensitivity"), ("specificity","Specificity")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (metric, title) in zip(axes, metrics):
        data, labels, colors = [], [], []
        for approach, results, color, n_key in [
            ("A (n=0)",  results_a, COL_A, 0),
            ("A (n=50)", results_a, "#1a5276", 50),
            ("B (n=0)",  results_b, COL_B, 0),
            ("B (n=50)", results_b, "#7b241c", 50),
            ("C (n=0)",  results_c, COL_C, 0),
            ("C (n=50)", results_c, "#1e8449", 50),
        ]:
            vals, _ = collect(results, n_key, metric)
            if len(vals): data.append(vals); labels.append(approach); colors.append(color)

        bp = ax.boxplot(data, patch_artist=True, notch=False,
                        medianprops=dict(color="black", linewidth=2))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color); patch.set_alpha(0.7)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel(title); ax.set_title(title)
        ax.set_ylim(-0.05, 1.1); ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig.suptitle("Performance Distribution Across 16 Patients", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig3_boxplots.png"); plt.close(fig)
    print("  fig3_boxplots.png")

fig_boxplots()


# ── Figure 4: Head-to-head scatter: A vs C ─────────────────────────────────
def fig_scatter_a_vs_c():
    f1_a, sa = collect(results_a, 0, "f1")
    f1_c, sc = collect(results_c, 0, "f1")
    common = sorted(set(sa) & set(sc), key=int)
    if len(common) < 3: return
    fa = np.array([f1_a[sa.index(s)] for s in common])
    fc = np.array([f1_c[sc.index(s)] for s in common])

    fig, ax = plt.subplots(figsize=(6, 6))
    colors = [COL_C if c > a else COL_A for a, c in zip(fa, fc)]
    ax.scatter(fa, fc, s=90, c=colors, edgecolors="black", lw=0.5, zorder=5)
    for i, sid in enumerate(common):
        ax.annotate(str(sid), (fa[i], fc[i]), textcoords="offset points",
                    xytext=(5, 4), fontsize=8)
    ax.plot([0,1],[0,1], "k--", alpha=0.3, label="Equal performance")
    ax.set_xlabel("Approach A (Bayesian Ensemble) F1")
    ax.set_ylabel("Approach C (Outlier Detection) F1")
    ax.set_title("Head-to-Head: Outlier vs Ensemble (n=0)")
    ax.set_xlim(0, 1.05); ax.set_ylim(0, 1.05); ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.3)
    n_c_wins = (fc > fa).sum()
    ax.text(0.05, 0.95, f"C wins: {n_c_wins}/{len(common)} patients",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax.legend(handles=[Patch(facecolor=COL_C, label="C wins"),
                       Patch(facecolor=COL_A, label="A wins")])
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig4_scatter_a_vs_c.png"); plt.close(fig)
    print("  fig4_scatter_a_vs_c.png")

fig_scatter_a_vs_c()


# ── Figure 5: Anomaly score distributions (FoG vs non-FoG) ─────────────────
def fig_score_distributions():
    """Show that FoG windows produce higher outlier scores than non-FoG."""
    import matplotlib.gridspec as gridspec
    # Pick 4 representative patients
    example_sids = [sid for sid in ds.subject_ids
                    if windowed.labels[windowed.subject_ids == sid].sum() > 20][:4]

    fig = plt.figure(figsize=(14, 8))
    gs  = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)

    for i, sid in enumerate(example_sids):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        mask = windowed.subject_ids == sid
        feat = all_feat[mask]
        wins = windowed.windows[mask]
        lbls = windowed.labels[mask]

        train_mask_i = ~mask
        train_feat_i = all_feat[train_mask_i]
        train_wins_i = windowed.windows[train_mask_i]
        train_lbls_i = windowed.labels[train_mask_i]

        ens = PersonalizedOutlierEnsemble(fi_weight=0.5)
        ens.fit_population(train_feat_i, train_wins_i, train_lbls_i)
        scores = ens.pop_detector_.score(feat, wins)

        fog_s  = scores[lbls == 1]
        nfog_s = scores[lbls == 0]

        bins = np.linspace(0, 1, 30)
        ax.hist(nfog_s, bins=bins, alpha=0.6, color=COL_A, density=True, label="Non-FoG (normal)")
        ax.hist(fog_s,  bins=bins, alpha=0.6, color=COL_B, density=True, label="FoG (outlier)")
        auc = roc_auc_score(lbls, scores) if lbls.sum() > 0 else 0
        ax.set_title(f"Patient {sid}  (AUC={auc:.2f})")
        ax.set_xlabel("Outlier Score"); ax.set_ylabel("Density")
        ax.legend(fontsize=9)

    fig.suptitle("Outlier Score Distributions: FoG vs Normal Gait", fontsize=14)
    fig.savefig(FIGURES_DIR / "fig5_score_distributions.png"); plt.close(fig)
    print("  fig5_score_distributions.png")

fig_score_distributions()


# ── Figure 6: Summary table ─────────────────────────────────────────────────
def fig_summary_table():
    rows = []
    for approach, results, n_vals in [
        ("A: Bayesian Ensemble", results_a, [0, 10, 20, 50]),
        ("B: SSL+LoRA",          results_b, [0, 10, 20, 50]),
        ("C: Outlier Detection", results_c, [0, 10, 20, 50]),
    ]:
        for n in n_vals:
            f1s,  _ = collect(results, n, "f1")
            sens, _ = collect(results, n, "sensitivity")
            spec, _ = collect(results, n, "specificity")
            auc,  _ = collect(results, n, "auroc")
            edr,  _ = collect(results, n, "event_detection_rate")
            if len(f1s) == 0: continue
            rows.append({
                "Approach": approach,
                "n_adapt / n_seed": n,
                "F1": f"{np.mean(f1s):.3f} +/- {np.std(f1s):.3f}",
                "Sensitivity": f"{np.mean(sens):.3f}",
                "Specificity": f"{np.mean(spec):.3f}",
                "AUC-ROC": f"{np.mean(auc):.3f}" if len(auc) else "N/A",
                "Event DR": f"{np.mean(edr):.3f}" if len(edr) else "N/A",
            })

    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.axis("off")
    tbl = ax.table(cellText=df.values, colLabels=df.columns,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.5)
    tbl.auto_set_column_width(col=list(range(len(df.columns))))

    # Colour rows by approach
    colors_map = {"A": COL_A, "B": COL_B, "C": COL_C}
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2c3e50"); cell.set_text_props(color="white", fontweight="bold")
        elif row > 0:
            approach_char = df.iloc[row-1]["Approach"][0]
            base = colors_map.get(approach_char, "#ecf0f1")
            cell.set_facecolor(base + "33")  # 20% alpha hex

    fig.suptitle("Summary: All Approaches & Adaptation Levels", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig6_summary_table.png", bbox_inches="tight"); plt.close(fig)
    print("  fig6_summary_table.png")

    df.to_csv(RESULTS_DIR / "summary_metrics.csv", index=False)
    print("  summary_metrics.csv")

fig_summary_table()


# ── Figure 7: Personalization gain curve (C only) ──────────────────────────
def fig_personalization_gain():
    """Show F1 improvement as more patient-specific normal windows are added."""
    n_vals = [0, 10, 20, 50]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: mean + std curve
    ax = axes[0]
    means, stds = [], []
    for n in n_vals:
        vals, _ = collect(results_c, n, "f1")
        means.append(np.mean(vals) if len(vals) else 0)
        stds.append(np.std(vals) if len(vals) else 0)
    means, stds = np.array(means), np.array(stds)
    ax.plot(n_vals, means, "-o", color=COL_C, lw=2.5, ms=10, label="Approach C")
    ax.fill_between(n_vals, means-stds, means+stds, alpha=0.2, color=COL_C)

    # Baseline: A at n=0
    a_f1, _ = collect(results_a, 0, "f1")
    ax.axhline(np.mean(a_f1), color=COL_A, linestyle="--", lw=1.5, label=f"A baseline (n=0) = {np.mean(a_f1):.3f}")
    ax.set_xlabel("Normal gait seed windows from patient (non-FoG only)")
    ax.set_ylabel("Mean F1 Score"); ax.set_title("Personalization Gain vs. Patient Data")
    ax.legend(); ax.grid(True, linestyle="--", alpha=0.3); ax.set_ylim(0, 0.8)
    ax.set_xticks(n_vals)

    # Right: individual patient curves
    ax2 = axes[1]
    for sid in results_c:
        ys = [results_c[sid].get(n, {}).get("f1", np.nan) for n in n_vals]
        ax2.plot(n_vals, ys, "-o", alpha=0.5, ms=5)
    ax2.set_xlabel("Normal gait seed windows from patient")
    ax2.set_ylabel("F1 Score"); ax2.set_title("Per-Patient Personalization Curves")
    ax2.grid(True, linestyle="--", alpha=0.3); ax2.set_ylim(0, 1.05)
    ax2.set_xticks(n_vals)

    fig.suptitle("Approach C: Personalization via Normal Gait Calibration", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig7_personalization_gain.png"); plt.close(fig)
    print("  fig7_personalization_gain.png")

fig_personalization_gain()


# ════════════════════════════════════════════════════════════════════════════
#  SUMMARY PRINT
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)
header = f"{'Approach':<28} {'n':>5}  {'F1':>14}  {'Sens':>7}  {'Spec':>7}  {'AUC':>7}"
print(header)
print("-" * 70)

for approach, results in [("A: Bayesian Ensemble", results_a),
                           ("B: SSL+LoRA",          results_b),
                           ("C: Outlier Detection", results_c)]:
    for n in [0, 10, 20, 50]:
        f1s,  _ = collect(results, n, "f1")
        sens, _ = collect(results, n, "sensitivity")
        spec, _ = collect(results, n, "specificity")
        auc,  _ = collect(results, n, "auroc")
        if not len(f1s): continue
        a_str = f"{approach:<28}" if n == 0 else f"{'':28}"
        print(f"{a_str} {n:>5}  "
              f"{np.mean(f1s):>6.3f}+/-{np.std(f1s):.3f}  "
              f"{np.mean(sens):>7.3f}  "
              f"{np.mean(spec):>7.3f}  "
              f"{np.mean(auc) if len(auc) else 0:>7.3f}")
    print()

print(f"\nModels: {MODELS_DIR}")
print(f"Figures: {FIGURES_DIR}")
print(f"Results: {RESULTS_DIR}")
