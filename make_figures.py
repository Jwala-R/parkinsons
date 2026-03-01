"""
Generate publication-quality figures comparing all 3 FoG detection approaches.
Includes: performance curves, per-patient bars, cluster/outlier visualisation (Approach C).
"""
import json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path

FIGURES_DIR = Path("results/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

with open("results/approach_a_results.json") as f: ra = json.load(f)
with open("results/approach_b_results.json") as f: rb = json.load(f)
with open("results/approach_c_results.json") as f: rc = json.load(f)

N_VALS  = [0, 10, 20, 50]

# ── Palette ────────────────────────────────────────────────────────────────
COL_A  = "#3B82F6"   # blue
COL_B  = "#EF4444"   # red
COL_C  = "#10B981"   # emerald
BG     = "#F8FAFC"   # near-white panel background

# ── Global style ───────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          11,
    "axes.titlesize":     13,
    "axes.titleweight":   "bold",
    "axes.labelsize":     11,
    "axes.labelweight":   "semibold",
    "axes.facecolor":     BG,
    "axes.edgecolor":     "#CBD5E1",
    "axes.linewidth":     1.0,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.color":         "#E2E8F0",
    "grid.linewidth":     0.8,
    "grid.linestyle":     "-",
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "xtick.color":        "#64748B",
    "ytick.color":        "#64748B",
    "xtick.direction":    "out",
    "ytick.direction":    "out",
    "legend.fontsize":    9.5,
    "legend.framealpha":  0.95,
    "legend.edgecolor":   "#CBD5E1",
    "legend.borderpad":   0.6,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "figure.facecolor":   "white",
    "savefig.facecolor":  "white",
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.15,
})

APPROACHES = [
    ("A: Bayesian Ensemble", ra, COL_A, "o", "-"),
    ("B: SSL + LoRA",        rb, COL_B, "s", "--"),
    ("C: Outlier Detection", rc, COL_C, "^", "-."),
]


# ── helpers ────────────────────────────────────────────────────────────────
def acc_from(r):
    ns  = r.get("n_samples", 0)
    np_ = r.get("n_positive", 0)
    nn  = ns - np_
    s   = r.get("sensitivity", 0)
    sp  = r.get("specificity", 0)
    return (s * np_ + sp * nn) / ns if ns > 0 else 0.0

def collect(results, n, metric):
    vals = []
    for sid, res in results.items():
        r = res.get(str(n), {})
        vals.append(acc_from(r) if metric == "accuracy" else r.get(metric, np.nan))
    return np.array([v for v in vals if not np.isnan(v)])

def curve(results, metric):
    means, stds = [], []
    for n in N_VALS:
        v = collect(results, n, metric)
        means.append(np.mean(v) if len(v) else 0)
        stds.append(np.std(v)  if len(v) else 0)
    return np.array(means), np.array(stds)

def patient_series(results, sid, metric):
    out = []
    for n in N_VALS:
        r = results.get(str(sid), {}).get(str(n), {})
        out.append(acc_from(r) if metric == "accuracy" else r.get(metric, np.nan))
    return np.array(out)

def style_ax(ax, ylim=(0, 1.05)):
    ax.set_ylim(*ylim)
    ax.set_xticks(N_VALS)
    ax.tick_params(length=3, width=0.8)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")

def add_end_label(ax, x, y, text, color, offset_y=0):
    ax.annotate(
        text, xy=(x, y), xytext=(8, offset_y),
        textcoords="offset points",
        fontsize=8.5, fontweight="bold", color=color, va="center",
    )

def shared_legend(fig, y=-0.04):
    handles = [
        Line2D([0], [0], color=col, lw=2.2, ls=ls, marker=mk, ms=8, label=lbl)
        for lbl, _, col, mk, ls in APPROACHES
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=3, fontsize=10,
        bbox_to_anchor=(0.5, y), framealpha=0.97, edgecolor="#CBD5E1", handlelength=2.5,
    )

X_AXIS_LABEL = "Patient-specific windows available (labelled for A/B, non-FoG only for C)"


# ══════════════════════════════════════════════════════════════════════════
# FIG 1 — Accuracy / F1 / AUC  (mean +/- 1 SD)
# ══════════════════════════════════════════════════════════════════════════
METRIC_DEFS = [
    ("accuracy", "Window Accuracy",
     "Fraction of 2-second windows correctly classified (balanced)"),
    ("f1",       "F1 Score",
     "Harmonic mean of precision and recall — primary metric"),
    ("auroc",    "AUC-ROC",
     "Area under the ROC curve — threshold-independent discrimination"),
]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
fig.suptitle(
    "FoG Detection Performance  \u2014  Mean \u00b1 1 SD across Patients",
    fontsize=14, fontweight="bold", y=1.02, color="#1E293B",
)

for ax, (metric, title, subtitle) in zip(axes, METRIC_DEFS):
    finals = []
    for lbl, res, col, mk, ls in APPROACHES:
        m, s = curve(res, metric)
        ax.plot(N_VALS, m, ls + mk, color=col, lw=2.4, ms=8,
                markerfacecolor="white", markeredgewidth=2.0, zorder=3)
        ax.fill_between(N_VALS,
                        np.clip(m - s, 0, 1), np.clip(m + s, 0, 1),
                        alpha=0.18, color=col, zorder=2)
        finals.append((m[-1], col, lbl.split(":")[0]))

    # Stagger end-labels by rank to avoid overlap
    for rank, (orig_i, (val, col, short)) in enumerate(
            sorted(enumerate(finals), key=lambda x: x[1][0])):
        add_end_label(ax, N_VALS[-1], val, f"{short}: {val:.3f}", col,
                      offset_y=(rank - 1) * 9)

    style_ax(ax)
    ax.set_xlabel(X_AXIS_LABEL, fontsize=8.5, color="#475569")
    ax.set_ylabel(title, fontsize=11)
    ax.set_title(title, fontsize=13, pad=8)
    ax.text(0.5, -0.26, subtitle, transform=ax.transAxes,
            ha="center", fontsize=8, color="#64748B", style="italic")

shared_legend(fig, y=-0.10)
fig.tight_layout(rect=[0, 0.05, 1, 1])
fig.savefig(FIGURES_DIR / "combined_acc_f1_roc_mean.png")
plt.close(fig)
print("Saved combined_acc_f1_roc_mean.png")


# ══════════════════════════════════════════════════════════════════════════
# FIG 2 — 6-panel: all key metrics
# ══════════════════════════════════════════════════════════════════════════
ALL_METRICS = [
    ("accuracy",             "Window Accuracy",
     "Balanced accuracy across 2-second IMU windows"),
    ("f1",                   "F1 Score",
     "Harmonic mean of precision & recall (FoG = positive class)"),
    ("auroc",                "AUC-ROC",
     "Threshold-independent discriminative ability"),
    ("sensitivity",          "Sensitivity (Recall)",
     "Fraction of true FoG windows correctly detected"),
    ("specificity",          "Specificity",
     "Fraction of normal-gait windows correctly rejected"),
    ("event_detection_rate", "FoG Episode Detection Rate",
     "Fraction of clinically-defined FoG episodes caught (event-level)"),
]

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle(
    "All Performance Metrics  \u2014  Three Approaches vs. Adaptation Data",
    fontsize=14, fontweight="bold", y=1.01, color="#1E293B",
)

for ax, (metric, title, subtitle) in zip(axes.flat, ALL_METRICS):
    for lbl, res, col, mk, ls in APPROACHES:
        m, s = curve(res, metric)
        ax.plot(N_VALS, m, ls + mk, color=col, lw=2.2, ms=7.5,
                markerfacecolor="white", markeredgewidth=1.8, zorder=3, label=lbl)
        ax.fill_between(N_VALS,
                        np.clip(m - s, 0, 1), np.clip(m + s, 0, 1),
                        alpha=0.15, color=col, zorder=2)
    style_ax(ax)
    ax.set_xlabel(X_AXIS_LABEL, fontsize=8.5, color="#475569")
    ax.set_ylabel(title, fontsize=11)
    ax.set_title(title, fontsize=12, pad=6)
    ax.text(0.5, -0.28, subtitle, transform=ax.transAxes,
            ha="center", fontsize=7.5, color="#64748B", style="italic")

shared_legend(fig, y=-0.04)
fig.tight_layout(rect=[0, 0.03, 1, 1])
fig.savefig(FIGURES_DIR / "combined_all_metrics.png")
plt.close(fig)
print("Saved combined_all_metrics.png")


# ══════════════════════════════════════════════════════════════════════════
# FIG 3 — Per-patient F1 curves
# ══════════════════════════════════════════════════════════════════════════
all_sids = sorted(set(ra) | set(rb) | set(rc), key=int)
ncols    = 4
nrows    = int(np.ceil(len(all_sids) / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.8, nrows * 3.0),
                         sharex=True, sharey=True)
fig.suptitle(
    "Per-Patient F1 Score  \u2014  All Three Approaches",
    fontsize=14, fontweight="bold", y=1.01, color="#1E293B",
)

for i, sid in enumerate(all_sids):
    ax = axes.flat[i]
    ax.set_facecolor(BG)
    for lbl, res, col, mk, ls in APPROACHES:
        ys = patient_series(res, sid, "f1")
        if not np.all(np.isnan(ys)):
            ax.plot(N_VALS, ys, ls + mk, color=col, lw=1.8, ms=6.5,
                    markerfacecolor="white", markeredgewidth=1.6, alpha=0.9)
    ax.set_title(f"Patient {sid}", fontsize=10, fontweight="bold",
                 color="#1E293B", pad=4)
    ax.set_ylim(-0.05, 1.09)
    ax.set_xticks(N_VALS)
    ax.tick_params(labelsize=8.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")
    if i % ncols == 0:
        ax.set_ylabel("F1 Score", fontsize=9, fontweight="semibold")
    if i >= (nrows - 1) * ncols:
        ax.set_xlabel("Patient-specific\nwindows available", fontsize=8.5, color="#475569")

for j in range(i + 1, nrows * ncols):
    axes.flat[j].set_visible(False)

shared_legend(fig, y=-0.03)
fig.tight_layout(rect=[0, 0.03, 1, 1])
fig.savefig(FIGURES_DIR / "per_patient_f1_all.png", dpi=300)
plt.close(fig)
print("Saved per_patient_f1_all.png")


# ══════════════════════════════════════════════════════════════════════════
# FIG 4 — Per-patient grouped bars at n=0 and n=50
# ══════════════════════════════════════════════════════════════════════════
common_sids = sorted(set(ra) & set(rb) & set(rc), key=int)
x = np.arange(len(common_sids))
w = 0.25

BAR_METRICS = [
    ("accuracy", "Window Accuracy",
     "Fraction of 2-second windows correctly classified (balanced)"),
    ("f1",       "F1 Score",
     "Harmonic mean of precision and recall"),
    ("auroc",    "AUC-ROC",
     "Area under the ROC curve"),
]

for n_level in [0, 50]:
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    title_str = ("Zero patient-specific data (cold-start)" if n_level == 0
                 else f"After {n_level} patient-specific windows provided")
    fig.suptitle(
        f"Per-Patient Comparison  \u2014  {title_str}  (n={n_level})",
        fontsize=14, fontweight="bold", y=1.01, color="#1E293B",
    )

    for row_i, (ax, (metric, ylabel, subtitle)) in enumerate(zip(axes, BAR_METRICS)):
        da = [acc_from(ra.get(s, {}).get(str(n_level), {})) if metric == "accuracy"
              else ra.get(s, {}).get(str(n_level), {}).get(metric, 0)
              for s in common_sids]
        db = [acc_from(rb.get(s, {}).get(str(n_level), {})) if metric == "accuracy"
              else rb.get(s, {}).get(str(n_level), {}).get(metric, 0)
              for s in common_sids]
        dc = [acc_from(rc.get(s, {}).get(str(n_level), {})) if metric == "accuracy"
              else rc.get(s, {}).get(str(n_level), {}).get(metric, 0)
              for s in common_sids]

        ax.bar(x - w, da, w, color=COL_A, alpha=0.88, zorder=3, linewidth=0,
               label=f"A: Bayesian Ensemble  (mean={np.nanmean(da):.3f})")
        ax.bar(x,     db, w, color=COL_B, alpha=0.88, zorder=3, linewidth=0,
               label=f"B: SSL + LoRA         (mean={np.nanmean(db):.3f})")
        ax.bar(x + w, dc, w, color=COL_C, alpha=0.88, zorder=3, linewidth=0,
               label=f"C: Outlier Detection  (mean={np.nanmean(dc):.3f})")

        ax.axhline(np.nanmean(da), color=COL_A, lw=1.5, ls="--", alpha=0.7, zorder=2)
        ax.axhline(np.nanmean(db), color=COL_B, lw=1.5, ls="--", alpha=0.7, zorder=2)
        ax.axhline(np.nanmean(dc), color=COL_C, lw=1.5, ls="--", alpha=0.7, zorder=2)

        ax.set_facecolor(BG)
        ax.set_ylabel(ylabel, fontsize=11, fontweight="semibold")
        ax.set_ylim(0, 1.18)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#CBD5E1")
        ax.spines["bottom"].set_color("#CBD5E1")
        ax.tick_params(length=3, width=0.8)
        ax.text(0.5, 1.04, subtitle, transform=ax.transAxes,
                ha="center", fontsize=8, color="#64748B", style="italic")

        if row_i == 0:
            ax.legend(loc="upper right", fontsize=8.5, framealpha=0.95,
                      edgecolor="#CBD5E1", ncol=3, bbox_to_anchor=(1, 1.30))

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([f"P{s}" for s in common_sids],
                             rotation=35, ha="right", fontsize=9.5)
    axes[-1].set_xlabel("Patient ID", fontsize=11, fontweight="semibold")

    fig.tight_layout(rect=[0, 0, 1, 1])
    fname = f"per_patient_bars_n{n_level}.png"
    fig.savefig(FIGURES_DIR / fname, dpi=300)
    plt.close(fig)
    print(f"Saved {fname}")


# ══════════════════════════════════════════════════════════════════════════
# FIG 5 — Approach C: Normal-gait cluster + FoG outliers (PCA)
# ══════════════════════════════════════════════════════════════════════════
print("Building cluster/outlier figure (loading data)...")

import sys
sys.path.insert(0, ".")
from src.data.fog_star_loader import FoGStarDataset
from src.data.windowing import create_windowed_dataset
from src.data.features import extract_batch_features
from src.models.outlier_detector import PersonalizedOutlierEnsemble, compute_freeze_index_batch
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
import time

ds       = FoGStarDataset("datasets/fog").load()
windowed = create_windowed_dataset(ds, window_seconds=2.0, overlap=0.5)

print("  Extracting features...")
t0 = time.time()
all_feat = extract_batch_features(windowed.windows)
all_feat = np.nan_to_num(all_feat, nan=0.0, posinf=0.0, neginf=0.0)
print(f"  Done in {time.time()-t0:.1f}s")

# Pick 4 patients that have FoG and enough windows for a clear plot
candidate_sids = [
    sid for sid in ds.subject_ids
    if (windowed.labels[windowed.subject_ids == sid].sum() >= 20
        and (windowed.labels[windowed.subject_ids == sid] == 0).sum() >= 30)
][:4]

fig, axes = plt.subplots(2, 2, figsize=(13, 11))
fig.suptitle(
    "Approach C: Normal-Gait Cluster vs. FoG Outliers\n"
    "(PCA projection of 339 handcrafted IMU features, population model)",
    fontsize=14, fontweight="bold", y=1.01, color="#1E293B",
)

for ax, sid in zip(axes.flat, candidate_sids):
    mask       = windowed.subject_ids == sid
    train_mask = ~mask

    feat  = all_feat[mask]
    wins  = windowed.windows[mask]
    lbls  = windowed.labels[mask]

    train_feat = all_feat[train_mask]
    train_wins = windowed.windows[train_mask]
    train_lbls = windowed.labels[train_mask]

    # Fit population model on training subjects
    ens = PersonalizedOutlierEnsemble(fi_weight=0.5, n_estimators=100)
    ens.fit_population(train_feat, train_wins, train_lbls)

    # Get anomaly scores
    scores = ens.pop_detector_.score(feat, wins, fs=60.0)

    # PCA on the scaled features
    scaler   = ens.pop_detector_.scaler_
    feat_sc  = scaler.transform(feat)
    pca      = PCA(n_components=2, random_state=42)
    feat_2d  = pca.fit_transform(feat_sc)

    nfog_mask = lbls == 0
    fog_mask  = lbls == 1

    # --- draw normal cluster as a filled ellipse-like scatter ---
    ax.scatter(
        feat_2d[nfog_mask, 0], feat_2d[nfog_mask, 1],
        c=scores[nfog_mask], cmap="Blues",
        vmin=scores.min(), vmax=scores.max(),
        s=18, alpha=0.55, linewidths=0,
        label=f"Normal gait  (n={nfog_mask.sum()})",
        zorder=2,
    )
    sc = ax.scatter(
        feat_2d[fog_mask, 0], feat_2d[fog_mask, 1],
        c=scores[fog_mask], cmap="Reds",
        vmin=scores.min(), vmax=scores.max(),
        s=38, alpha=0.85, linewidths=0.5, edgecolors="#7F1D1D",
        marker="^",
        label=f"FoG (outlier)  (n={fog_mask.sum()})",
        zorder=4,
    )

    # Mark the highest-scored window (most FoG-like)
    top_idx = np.argmax(scores)
    ax.scatter(
        feat_2d[top_idx, 0], feat_2d[top_idx, 1],
        s=160, facecolors="none", edgecolors="#B91C1C",
        linewidths=2.2, zorder=5,
        label="Most anomalous window",
    )

    # Compute variance explained
    var_exp = pca.explained_variance_ratio_
    ax.set_xlabel(
        f"PC 1  ({var_exp[0]*100:.1f}% variance explained)",
        fontsize=9.5, color="#475569",
    )
    ax.set_ylabel(
        f"PC 2  ({var_exp[1]*100:.1f}% variance explained)",
        fontsize=9.5, color="#475569",
    )

    # Mean outlier scores in title
    mean_nfog = scores[nfog_mask].mean()
    mean_fog  = scores[fog_mask].mean()
    ax.set_title(
        f"Patient {sid}  \u2014  "
        f"Outlier score: normal={mean_nfog:.2f}  FoG={mean_fog:.2f}",
        fontsize=11, fontweight="bold", color="#1E293B", pad=6,
    )
    ax.set_facecolor(BG)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")
    ax.tick_params(labelsize=8.5)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.92,
              edgecolor="#CBD5E1", markerscale=1.2)

# Shared colourbar annotation
fig.text(
    0.5, -0.02,
    "Point colour encodes anomaly score (0 = normal, 1 = maximally anomalous).  "
    "Triangles = labelled FoG windows.  Circles = normal gait.",
    ha="center", fontsize=9, color="#475569", style="italic",
)

fig.tight_layout(rect=[0, 0.02, 1, 1])
fig.savefig(FIGURES_DIR / "approach_c_cluster_outliers.png", dpi=300)
plt.close(fig)
print("Saved approach_c_cluster_outliers.png")


print("\nAll figures saved ->", FIGURES_DIR)
