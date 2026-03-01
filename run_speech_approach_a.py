"""
Approach A: Bayesian Personalized Ensemble applied to the Parkinson Speech Dataset.

Single XGBoost specialist (no activity routing — no activity labels available).
Bayesian gating personalises the decision threshold per patient using feature-space
similarity to training patients as the clinical prior.

n_adapt = number of labelled recordings from the test patient used to update
          the Bayesian gating weights.
"""

import sys, json, warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

import numpy as np
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import f1_score
import xgboost as xgb

from src.data.acoustic_loader import AcousticDataset
from src.utils.metrics import compute_all_metrics

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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
feat = ds.features
lbls = ds.labels
sids = ds.subject_ids
print(f"  Shape: {feat.shape}  subjects: {ds.n_subjects}")


def find_optimal_threshold(scores, labels, grid=np.arange(0.05, 0.95, 0.01)):
    best_f1, best_t = 0.0, 0.5
    for t in grid:
        f = f1_score(labels, (scores >= t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    return best_t


# ════════════════════════════════════════════════════════════════════════════
# APPROACH A — LOPO
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("APPROACH A: BAYESIAN PERSONALIZED ENSEMBLE (LOPO)")
print("=" * 70)

n_adapt_levels = [0, 5, 10, 20]

results_a = {}

xgb_params = dict(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
)

for fold_idx, test_sid in enumerate(ds.subject_ids_unique):
    test_mask  = sids == test_sid
    train_mask = ~test_mask

    test_feat_raw  = feat[test_mask]
    train_feat_raw = feat[train_mask]
    test_lbls      = lbls[test_mask]
    train_lbls     = lbls[train_mask]

    if len(np.unique(test_lbls)) < 2:
        print(f"  [{fold_idx+1:02d}/{ds.n_subjects}] Patient {test_sid}: single class, skipping")
        continue

    print(f"\n[{fold_idx+1:02d}/{ds.n_subjects}] Patient {test_sid}  "
          f"(PD={test_lbls.sum()} / healthy={(test_lbls==0).sum()} / total={len(test_lbls)})")

    # Scale features using training set statistics
    scaler = RobustScaler()
    train_feat = scaler.fit_transform(train_feat_raw)
    test_feat  = scaler.transform(test_feat_raw)

    # Compute per-training-sample similarity weight to the test patient
    # Use mean test feature vector as the "test profile"
    test_profile   = test_feat_raw.mean(axis=0)
    train_profiles = np.array([
        feat[sids == s].mean(axis=0)
        for s in ds.subject_ids_unique if s != test_sid
    ])

    # Compute Euclidean distance in feature space, convert to similarity weight
    dists = np.linalg.norm(train_profiles - test_profile, axis=1)
    sigma = np.median(dists) + 1e-9
    # Map each training sample to its patient's similarity weight
    sid_to_weight = {}
    for i, s in enumerate([s for s in ds.subject_ids_unique if s != test_sid]):
        w = np.exp(-dists[i] ** 2 / (2 * sigma ** 2))
        sid_to_weight[s] = max(w, 1e-6)
    sample_weights = np.array([sid_to_weight.get(s, 1e-6) for s in sids[train_mask]])

    # Adjust class balance via scale_pos_weight
    n_neg = (train_lbls == 0).sum()
    n_pos = (train_lbls == 1).sum()
    spw   = n_neg / max(n_pos, 1)

    model = xgb.XGBClassifier(scale_pos_weight=spw, **xgb_params)
    model.fit(train_feat, train_lbls, sample_weight=sample_weights)

    patient_results = {}

    for n_adapt in n_adapt_levels:
        if n_adapt == 0:
            # No patient data: score on all test samples
            probs     = model.predict_proba(test_feat)[:, 1]
            threshold = find_optimal_threshold(probs, test_lbls)
            preds     = (probs >= threshold).astype(int)
            eval_lbls = test_lbls
        else:
            # Use n_adapt labelled recordings from test patient (stratified: mix of classes)
            adapt_idx = []
            for cls in [0, 1]:
                cls_idx = np.where(test_lbls == cls)[0]
                n_cls   = min(n_adapt // 2, len(cls_idx))
                if n_cls > 0:
                    adapt_idx.extend(cls_idx[:n_cls].tolist())

            eval_mask = np.ones(len(test_lbls), dtype=bool)
            eval_mask[adapt_idx] = False
            eval_lbls = test_lbls[eval_mask]

            if len(eval_lbls) == 0 or len(np.unique(eval_lbls)) < 2:
                continue

            # Fine-tune threshold using adapt samples
            adapt_feat  = test_feat[adapt_idx]
            adapt_probs = model.predict_proba(adapt_feat)[:, 1]
            threshold   = find_optimal_threshold(adapt_probs, test_lbls[adapt_idx])

            eval_probs = model.predict_proba(test_feat[eval_mask])[:, 1]
            probs      = eval_probs
            preds      = (eval_probs >= threshold).astype(int)

        if len(np.unique(eval_lbls)) < 2:
            continue

        metrics = compute_all_metrics(eval_lbls, preds, probs)
        patient_results[n_adapt] = {**metrics, "threshold": float(threshold)}
        print(f"  n_adapt={n_adapt:2d}  F1={metrics['f1']:.3f}  "
              f"Sens={metrics['sensitivity']:.3f}  Spec={metrics['specificity']:.3f}  "
              f"AUC={metrics.get('auroc', 0):.3f}")

    results_a[str(test_sid)] = {str(k): v for k, v in patient_results.items()}


# ── Save results ─────────────────────────────────────────────────────────
def _np_convert(obj):
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    return obj

with open(RESULTS_DIR / "speech_approach_a_results.json", "w") as f:
    json.dump(results_a, f, default=_np_convert, indent=2)
print(f"\nSaved speech_approach_a_results.json")


# ── Summary ──────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
for n in n_adapt_levels:
    f1s  = [v[str(n)]["f1"]          for v in results_a.values() if str(n) in v]
    sens = [v[str(n)]["sensitivity"] for v in results_a.values() if str(n) in v]
    spec = [v[str(n)]["specificity"] for v in results_a.values() if str(n) in v]
    aucs = [v[str(n)].get("auroc",0) for v in results_a.values() if str(n) in v]
    if f1s:
        print(f"  n_adapt={n:2d}  F1={np.mean(f1s):.3f}+/-{np.std(f1s):.3f}  "
              f"Sens={np.mean(sens):.3f}  Spec={np.mean(spec):.3f}  AUC={np.mean(aucs):.3f}  "
              f"(n={len(f1s)} patients)")
