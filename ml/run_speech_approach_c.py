"""
Approach C: Personalized Outlier Detection applied to the Parkinson Speech Dataset.

Each patient's healthy (label=0) voice recordings define their "normal" acoustic profile.
Parkinson's recordings (label=1) are detected as anomalies from that normal profile.

n_seed = number of healthy recordings per patient used to personalise the model.
At n_seed=0 the model uses the population distribution of healthy voice (other patients).
"""

import sys, os, time, json, warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
import joblib

from src.data.acoustic_loader import AcousticDataset
from src.models.outlier_detector import PersonalizedOutlierEnsemble
from src.utils.metrics import compute_all_metrics

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("results/figures")
MODELS_DIR  = Path("results/models")
for d in [RESULTS_DIR, FIGURES_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DATA_DIR = (
    "datasets/parkinson+speech+dataset+with+multiple+types+of+sound+recordings"
    "/Parkinson_Multiple_Sound_Recording"
)

# ════════════════════════════════════════════════════════════════════════════
# DATA
# ════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("LOADING PARKINSON SPEECH DATASET")
print("=" * 70)
ds   = AcousticDataset(DATA_DIR).load(use_test=True)
feat = ds.features          # (n, 26)
lbls = ds.labels            # (n,)
sids = ds.subject_ids       # (n,)
print(f"  Shape: {feat.shape}  subjects: {ds.n_subjects}")

# ════════════════════════════════════════════════════════════════════════════
# APPROACH C — LOPO
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("APPROACH C: PERSONALIZED OUTLIER DETECTION (LOPO)")
print("=" * 70)

# Seed levels: how many healthy recordings from the test patient are used
# (fewer than FoG-STAR because each patient has only ~26 recordings total)
n_seed_levels = [0, 5, 10, 20]

results_c = {}

for fold_idx, test_sid in enumerate(ds.subject_ids_unique):
    test_mask  = sids == test_sid
    train_mask = ~test_mask

    test_feat   = feat[test_mask]
    train_feat  = feat[train_mask]
    test_lbls   = lbls[test_mask]
    train_lbls  = lbls[train_mask]

    # Skip patients with only one class in test set (can't compute all metrics)
    if len(np.unique(test_lbls)) < 2:
        print(f"  [{fold_idx+1:02d}/{ds.n_subjects}] Patient {test_sid}: "
              f"only class {test_lbls[0]} present, skipping")
        continue

    print(f"\n[{fold_idx+1:02d}/{ds.n_subjects}] Patient {test_sid}  "
          f"(PD={test_lbls.sum()} / healthy={(test_lbls==0).sum()} / total={len(test_lbls)})")

    # Fit population model on all OTHER patients' healthy recordings
    ensemble = PersonalizedOutlierEnsemble(fi_weight=0.0, n_estimators=200)
    ensemble.fit_population(train_feat, None, train_lbls)

    patient_results = {}
    for n_seed in n_seed_levels:
        preds, scores, threshold, eval_lbl = ensemble.score_and_predict(
            test_feat, None, test_lbls, n_seed_nonfog=n_seed,
        )

        if len(eval_lbl) == 0 or len(np.unique(eval_lbl)) < 2:
            continue

        metrics = compute_all_metrics(eval_lbl, preds, scores)
        patient_results[n_seed] = {**metrics, "threshold": float(threshold)}
        print(f"  n_seed={n_seed:2d}  F1={metrics['f1']:.3f}  "
              f"Sens={metrics['sensitivity']:.3f}  Spec={metrics['specificity']:.3f}  "
              f"AUC={metrics.get('auroc', 0):.3f}")

    results_c[str(test_sid)] = {str(k): v for k, v in patient_results.items()}


# ── Save results ─────────────────────────────────────────────────────────
def _np_convert(obj):
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    return obj

with open(RESULTS_DIR / "speech_approach_c_results.json", "w") as f:
    json.dump(results_c, f, default=_np_convert, indent=2)
print(f"\nSaved speech_approach_c_results.json")


# ── Summary ──────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
for n_seed in n_seed_levels:
    f1s   = [v[str(n_seed)]["f1"]          for v in results_c.values() if str(n_seed) in v]
    senss = [v[str(n_seed)]["sensitivity"] for v in results_c.values() if str(n_seed) in v]
    specs = [v[str(n_seed)]["specificity"] for v in results_c.values() if str(n_seed) in v]
    aucs  = [v[str(n_seed)].get("auroc",0) for v in results_c.values() if str(n_seed) in v]
    if f1s:
        print(f"  n_seed={n_seed:2d}  F1={np.mean(f1s):.3f}+/-{np.std(f1s):.3f}  "
              f"Sens={np.mean(senss):.3f}  Spec={np.mean(specs):.3f}  AUC={np.mean(aucs):.3f}  "
              f"(n={len(f1s)} patients)")


# ── Export final model ────────────────────────────────────────────────────
print("\nExporting final model (trained on full dataset)...")
final_ensemble = PersonalizedOutlierEnsemble(fi_weight=0.0, n_estimators=200)
final_ensemble.fit_population(feat, None, lbls)
joblib.dump(final_ensemble, MODELS_DIR / "speech_approach_c_detector.pkl")
print("  Saved speech_approach_c_detector.pkl")


# ════════════════════════════════════════════════════════════════════════════
# CLUSTER / OUTLIER VISUALISATION
# ════════════════════════════════════════════════════════════════════════════
print("\nGenerating cluster/outlier figure...")

BG    = "#F8FAFC"
COL_H = "#3B82F6"   # healthy = blue
COL_P = "#EF4444"   # Parkinson's = red

# Pick 4 patients with both classes in test fold
vis_sids = [
    sid for sid in ds.subject_ids_unique
    if (lbls[sids == sid] == 0).sum() >= 3
    and (lbls[sids == sid] == 1).sum() >= 3
][:4]

fig, axes = plt.subplots(2, 2, figsize=(13, 11))
fig.patch.set_facecolor("white")
fig.suptitle(
    "Approach C: Normal Voice Cluster vs. Parkinson's Outliers\n"
    "(PCA of 26 acoustic features — population-level outlier model)",
    fontsize=14, fontweight="bold", color="#1E293B", y=1.01,
)

for ax, sid in zip(axes.flat, vis_sids):
    mask       = sids == sid
    train_mask = ~mask

    p_feat = feat[mask]
    p_lbls = lbls[mask]
    tf     = feat[train_mask]
    tl     = lbls[train_mask]

    ens = PersonalizedOutlierEnsemble(fi_weight=0.0, n_estimators=100)
    ens.fit_population(tf, None, tl)
    scores = ens.pop_detector_.score(p_feat)

    scaler  = ens.pop_detector_.scaler_
    feat_sc = scaler.transform(p_feat)
    pca     = PCA(n_components=2, random_state=42)
    pts     = pca.fit_transform(feat_sc)
    var     = pca.explained_variance_ratio_

    h_mask = p_lbls == 0
    p_mask = p_lbls == 1

    ax.scatter(pts[h_mask, 0], pts[h_mask, 1],
               c=scores[h_mask], cmap="Blues",
               vmin=scores.min(), vmax=scores.max(),
               s=55, alpha=0.65, linewidths=0, zorder=2,
               label=f"Healthy  (n={h_mask.sum()})")
    ax.scatter(pts[p_mask, 0], pts[p_mask, 1],
               c=scores[p_mask], cmap="Reds",
               vmin=scores.min(), vmax=scores.max(),
               s=75, alpha=0.85, linewidths=0.5, edgecolors="#7F1D1D",
               marker="^", zorder=4,
               label=f"Parkinson's (n={p_mask.sum()})")

    # Circle the most anomalous point
    top = np.argmax(scores)
    ax.scatter(pts[top, 0], pts[top, 1],
               s=200, facecolors="none", edgecolors="#B91C1C",
               linewidths=2.2, zorder=5, label="Most anomalous")

    mn_h = scores[h_mask].mean()
    mn_p = scores[p_mask].mean()
    ax.set_title(
        f"Patient {sid}  —  Outlier score: healthy={mn_h:.2f}  PD={mn_p:.2f}",
        fontsize=11, fontweight="bold", color="#1E293B", pad=5,
    )
    ax.set_xlabel(f"PC 1  ({var[0]*100:.1f}% variance)", fontsize=9.5, color="#475569")
    ax.set_ylabel(f"PC 2  ({var[1]*100:.1f}% variance)", fontsize=9.5, color="#475569")
    ax.set_facecolor(BG)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")
    ax.legend(fontsize=8.5, loc="upper right", framealpha=0.92, edgecolor="#CBD5E1")

fig.text(
    0.5, -0.02,
    "Point colour = anomaly score (darker = more anomalous).  "
    "Triangles = Parkinson's recordings.  Circles = healthy.",
    ha="center", fontsize=9, color="#475569", style="italic",
)
fig.tight_layout(rect=[0, 0.02, 1, 1])
fig.savefig(FIGURES_DIR / "speech_approach_c_cluster_outliers.png", dpi=300,
            bbox_inches="tight")
plt.close(fig)
print("  Saved speech_approach_c_cluster_outliers.png")

print(f"\nDone. Results -> {RESULTS_DIR}")
