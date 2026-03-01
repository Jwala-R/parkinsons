"""
Generate publication-quality figures comparing all 3 approaches on the
Parkinson Speech Dataset (acoustic features).
"""
import json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

FIGURES_DIR = Path("results/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

with open("results/speech_approach_a_results.json") as f: ra = json.load(f)
with open("results/speech_approach_b_results.json") as f: rb = json.load(f)
with open("results/speech_approach_c_results.json") as f: rc = json.load(f)

N_VALS  = [0, 5, 10, 20]
X_LABEL = "Patient-specific recordings available (labelled for A/B, healthy-only for C)"

COL_A = "#3B82F6"
COL_B = "#EF4444"
COL_C = "#10B981"
BG    = "#F8FAFC"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.labelsize": 11, "axes.labelweight": "semibold",
    "axes.facecolor": BG, "axes.edgecolor": "#CBD5E1", "axes.linewidth": 1.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#E2E8F0", "grid.linewidth": 0.8,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "xtick.color": "#64748B", "ytick.color": "#64748B",
    "legend.fontsize": 9.5, "legend.framealpha": 0.95, "legend.edgecolor": "#CBD5E1",
    "figure.dpi": 150, "savefig.dpi": 300,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "savefig.bbox": "tight", "savefig.pad_inches": 0.15,
})

APPROACHES = [
    ("A: Bayesian Ensemble", ra, COL_A, "o", "-"),
    ("B: MLP Fine-tuning",   rb, COL_B, "s", "--"),
    ("C: Outlier Detection", rc, COL_C, "^", "-."),
]


# ── helpers ────────────────────────────────────────────────────────────────
def collect(results, n, metric):
    vals = []
    for sid, res in results.items():
        r = res.get(str(n), {})
        v = r.get(metric, np.nan)
        if not np.isnan(v):
            vals.append(v)
    return np.array(vals)

def curve(results, metric):
    means, stds = [], []
    for n in N_VALS:
        v = collect(results, n, metric)
        means.append(np.mean(v) if len(v) else 0)
        stds.append(np.std(v)  if len(v) else 0)
    return np.array(means), np.array(stds)

def patient_series(results, sid, metric):
    return np.array([results.get(str(sid), {}).get(str(n), {}).get(metric, np.nan)
                     for n in N_VALS])

def style_ax(ax, ylim=(0, 1.05)):
    ax.set_ylim(*ylim)
    ax.set_xticks(N_VALS)
    ax.tick_params(length=3, width=0.8)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")

def stagger_end_labels(ax, finals):
    for rank, (orig_i, (val, col, short)) in enumerate(
            sorted(enumerate(finals), key=lambda x: x[1][0])):
        ax.annotate(f"{short}: {val:.3f}", xy=(N_VALS[-1], val),
                    xytext=(8, (rank - 1) * 9), textcoords="offset points",
                    fontsize=8.5, fontweight="bold", color=col, va="center")

def shared_legend(fig, y=-0.05):
    handles = [Line2D([0], [0], color=col, lw=2.2, ls=ls, marker=mk, ms=8, label=lbl)
               for lbl, _, col, mk, ls in APPROACHES]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, y), framealpha=0.97, edgecolor="#CBD5E1",
               handlelength=2.5)


# ══════════════════════════════════════════════════════════════════════════
# FIG 1 — Accuracy / F1 / AUC (mean +/- 1 SD)
# ══════════════════════════════════════════════════════════════════════════
METRIC_DEFS = [
    ("f1",          "F1 Score",
     "Harmonic mean of precision and recall — primary metric"),
    ("sensitivity", "Sensitivity (Recall)",
     "Fraction of true Parkinson's recordings correctly detected"),
    ("specificity", "Specificity",
     "Fraction of healthy recordings correctly identified"),
    ("auroc",       "AUC-ROC",
     "Threshold-independent discriminative ability"),
]

fig, axes = plt.subplots(1, 4, figsize=(19, 4.6))
fig.suptitle(
    "Parkinson Speech Detection  \u2014  Mean \u00b1 1 SD across Patients",
    fontsize=14, fontweight="bold", y=1.02, color="#1E293B",
)

for ax, (metric, title, subtitle) in zip(axes, METRIC_DEFS):
    finals = []
    for lbl, res, col, mk, ls in APPROACHES:
        m, s = curve(res, metric)
        ax.plot(N_VALS, m, ls + mk, color=col, lw=2.4, ms=8,
                markerfacecolor="white", markeredgewidth=2.0, zorder=3)
        ax.fill_between(N_VALS, np.clip(m - s, 0, 1), np.clip(m + s, 0, 1),
                        alpha=0.18, color=col, zorder=2)
        finals.append((m[-1], col, lbl.split(":")[0]))
    stagger_end_labels(ax, finals)
    style_ax(ax)
    ax.set_xlabel(X_LABEL, fontsize=8.5, color="#475569")
    ax.set_ylabel(title, fontsize=11)
    ax.set_title(title, fontsize=12, pad=8)
    ax.text(0.5, -0.30, subtitle, transform=ax.transAxes,
            ha="center", fontsize=7.5, color="#64748B", style="italic")

shared_legend(fig, y=-0.12)
fig.tight_layout(rect=[0, 0.06, 1, 1])
fig.savefig(FIGURES_DIR / "speech_combined_metrics_mean.png")
plt.close(fig)
print("Saved speech_combined_metrics_mean.png")


# ══════════════════════════════════════════════════════════════════════════
# FIG 2 — Per-patient F1 curves (small multiples)
# ══════════════════════════════════════════════════════════════════════════
all_sids = sorted(set(ra) | set(rb) | set(rc), key=int)
ncols    = 5
nrows    = int(np.ceil(len(all_sids) / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.6, nrows * 2.9),
                         sharex=True, sharey=True)
fig.suptitle(
    "Per-Patient F1 Score  \u2014  All Three Approaches  (Parkinson Speech)",
    fontsize=13, fontweight="bold", y=1.01, color="#1E293B",
)

for i, sid in enumerate(all_sids):
    ax = axes.flat[i]
    ax.set_facecolor(BG)
    for lbl, res, col, mk, ls in APPROACHES:
        ys = patient_series(res, sid, "f1")
        if not np.all(np.isnan(ys)):
            ax.plot(N_VALS, ys, ls + mk, color=col, lw=1.8, ms=6,
                    markerfacecolor="white", markeredgewidth=1.6, alpha=0.9)
    ax.set_title(f"Patient {sid}", fontsize=9.5, fontweight="bold", color="#1E293B", pad=3)
    ax.set_ylim(-0.05, 1.09)
    ax.set_xticks(N_VALS)
    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")
    if i % ncols == 0:
        ax.set_ylabel("F1 Score", fontsize=9, fontweight="semibold")
    if i >= (nrows - 1) * ncols:
        ax.set_xlabel("n recordings", fontsize=8.5, color="#475569")

for j in range(i + 1, nrows * ncols):
    axes.flat[j].set_visible(False)

shared_legend(fig, y=-0.03)
fig.tight_layout(rect=[0, 0.03, 1, 1])
fig.savefig(FIGURES_DIR / "speech_per_patient_f1.png", dpi=300)
plt.close(fig)
print("Saved speech_per_patient_f1.png")


# ══════════════════════════════════════════════════════════════════════════
# FIG 3 — Per-patient grouped bars at n=0 and n=20
# ══════════════════════════════════════════════════════════════════════════
common_sids = sorted(set(ra) & set(rb) & set(rc), key=int)
x  = np.arange(len(common_sids))
w  = 0.25

BAR_METRICS = [
    ("f1",          "F1 Score",    "Harmonic mean of precision and recall"),
    ("sensitivity", "Sensitivity", "Fraction of Parkinson's recordings detected"),
    ("specificity", "Specificity", "Fraction of healthy recordings correctly rejected"),
]

for n_level in [0, 20]:
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    title_str = ("Zero patient recordings (cold-start)" if n_level == 0
                 else f"After {n_level} patient recordings provided")
    fig.suptitle(
        f"Per-Patient Results  \u2014  {title_str}  (n={n_level})",
        fontsize=14, fontweight="bold", y=1.01, color="#1E293B",
    )

    for row_i, (ax, (metric, ylabel, subtitle)) in enumerate(zip(axes, BAR_METRICS)):
        da = [ra.get(s, {}).get(str(n_level), {}).get(metric, 0) for s in common_sids]
        db = [rb.get(s, {}).get(str(n_level), {}).get(metric, 0) for s in common_sids]
        dc = [rc.get(s, {}).get(str(n_level), {}).get(metric, 0) for s in common_sids]

        ax.bar(x - w, da, w, color=COL_A, alpha=0.88, zorder=3, linewidth=0,
               label=f"A: Bayesian Ensemble  (mean={np.nanmean(da):.3f})")
        ax.bar(x,     db, w, color=COL_B, alpha=0.88, zorder=3, linewidth=0,
               label=f"B: MLP Fine-tuning    (mean={np.nanmean(db):.3f})")
        ax.bar(x + w, dc, w, color=COL_C, alpha=0.88, zorder=3, linewidth=0,
               label=f"C: Outlier Detection  (mean={np.nanmean(dc):.3f})")

        ax.axhline(np.nanmean(da), color=COL_A, lw=1.5, ls="--", alpha=0.7, zorder=2)
        ax.axhline(np.nanmean(db), color=COL_B, lw=1.5, ls="--", alpha=0.7, zorder=2)
        ax.axhline(np.nanmean(dc), color=COL_C, lw=1.5, ls="--", alpha=0.7, zorder=2)

        ax.set_facecolor(BG)
        ax.set_ylabel(ylabel, fontsize=11, fontweight="semibold")
        ax.set_ylim(0, 1.18)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#CBD5E1"); ax.spines["bottom"].set_color("#CBD5E1")
        ax.tick_params(length=3, width=0.8)
        ax.text(0.5, 1.04, subtitle, transform=ax.transAxes,
                ha="center", fontsize=8, color="#64748B", style="italic")
        if row_i == 0:
            ax.legend(loc="upper right", fontsize=8.5, framealpha=0.95,
                      edgecolor="#CBD5E1", ncol=3, bbox_to_anchor=(1, 1.32))

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([f"P{s}" for s in common_sids],
                             rotation=35, ha="right", fontsize=9)
    axes[-1].set_xlabel("Patient ID", fontsize=11, fontweight="semibold")
    fig.tight_layout(rect=[0, 0, 1, 1])
    fname = f"speech_per_patient_bars_n{n_level}.png"
    fig.savefig(FIGURES_DIR / fname, dpi=300)
    plt.close(fig)
    print(f"Saved {fname}")


print("\nAll speech figures saved ->", FIGURES_DIR)
