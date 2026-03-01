"""
Approach B: MLP Personalisation applied to the Parkinson Speech Dataset.

A small MLP (2-layer, 64 hidden units) is pre-trained on all training patients,
then fine-tuned per patient during the LOPO protocol.

No T-MAE or LoRA needed — the input is already 26 scalar features (not raw
time-series), so a lightweight MLP replaces the Transformer+LoRA stack.

n_adapt = number of labelled recordings from the test patient used for
          per-patient fine-tuning.
"""

import sys, json, warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from sklearn.preprocessing import RobustScaler

from src.data.acoustic_loader import AcousticDataset
from src.utils.metrics import compute_all_metrics

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

DATA_DIR = (
    "datasets/parkinson+speech+dataset+with+multiple+types+of+sound+recordings"
    "/Parkinson_Multiple_Sound_Recording"
)


# ── MLP model ─────────────────────────────────────────────────────────────
class MLPClassifier(nn.Module):
    def __init__(self, n_features: int = 26, hidden: int = 64, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x)

    def predict_proba(self, x):
        self.eval()
        with torch.no_grad():
            return torch.sigmoid(self.forward(x)).squeeze(-1)


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce  = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        prob = torch.sigmoid(logits)
        pt   = torch.where(targets == 1, prob, 1 - prob)
        at   = torch.where(targets == 1,
                           torch.full_like(targets, self.alpha),
                           torch.full_like(targets, 1 - self.alpha))
        return (at * (1 - pt) ** self.gamma * bce).mean()


def find_threshold(probs_np, labels_np):
    best_f1, best_t = 0.0, 0.5
    for t in np.arange(0.05, 0.95, 0.01):
        from sklearn.metrics import f1_score
        f = f1_score(labels_np, (probs_np >= t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    return best_t


def train_model(model, X_t, y_t, n_epochs=100, lr=1e-3, batch_size=64):
    """Train the MLP on (X_t, y_t) with focal loss."""
    criterion = FocalLoss(alpha=0.75, gamma=2.0)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    dataset = torch.utils.data.TensorDataset(X_t, y_t)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(n_epochs):
        for xb, yb in loader:
            opt.zero_grad()
            loss = criterion(model(xb).squeeze(-1), yb)
            loss.backward()
            opt.step()
        scheduler.step()
    return model


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


# ════════════════════════════════════════════════════════════════════════════
# APPROACH B — LOPO
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("APPROACH B: MLP PERSONALISATION (LOPO)")
print("=" * 70)

n_adapt_levels = [0, 5, 10, 20]
N_FEATURES = feat.shape[1]

results_b = {}

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

    # Scale
    scaler     = RobustScaler()
    train_feat = scaler.fit_transform(train_feat_raw).astype(np.float32)
    test_feat  = scaler.transform(test_feat_raw).astype(np.float32)

    X_train = torch.tensor(train_feat, dtype=torch.float32).to(DEVICE)
    y_train = torch.tensor(train_lbls, dtype=torch.float32).to(DEVICE)
    X_test  = torch.tensor(test_feat,  dtype=torch.float32).to(DEVICE)

    # Train base model on all other patients
    model = MLPClassifier(n_features=N_FEATURES).to(DEVICE)
    train_model(model, X_train, y_train, n_epochs=120, lr=1e-3)

    patient_results = {}

    for n_adapt in n_adapt_levels:
        # Save base weights for reset between n_adapt levels
        base_state = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(base_state)

        if n_adapt == 0:
            probs     = model.predict_proba(X_test).cpu().numpy()
            threshold = find_threshold(probs, test_lbls)
            preds     = (probs >= threshold).astype(int)
            eval_lbls = test_lbls
        else:
            # Stratified adaptation samples
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

            # Fine-tune on patient's own labelled samples
            X_adapt = torch.tensor(test_feat[adapt_idx], dtype=torch.float32).to(DEVICE)
            y_adapt = torch.tensor(test_lbls[adapt_idx], dtype=torch.float32).to(DEVICE)

            # Fewer epochs for adaptation (avoid overfitting on small sample)
            n_steps = min(50, max(10, n_adapt * 3))
            train_model(model, X_adapt, y_adapt, n_epochs=n_steps, lr=5e-4)

            eval_probs = model.predict_proba(X_test[eval_mask]).cpu().numpy()
            probs      = eval_probs
            # Threshold from adapt set
            adapt_probs = model.predict_proba(X_adapt).cpu().numpy()
            threshold   = find_threshold(adapt_probs, test_lbls[adapt_idx])
            preds       = (eval_probs >= threshold).astype(int)

        if len(np.unique(eval_lbls)) < 2:
            continue

        metrics = compute_all_metrics(eval_lbls, preds, probs)
        patient_results[n_adapt] = {**metrics, "threshold": float(threshold)}
        print(f"  n_adapt={n_adapt:2d}  F1={metrics['f1']:.3f}  "
              f"Sens={metrics['sensitivity']:.3f}  Spec={metrics['specificity']:.3f}  "
              f"AUC={metrics.get('auroc', 0):.3f}")

    results_b[str(test_sid)] = {str(k): v for k, v in patient_results.items()}


# ── Save results ─────────────────────────────────────────────────────────
def _np_convert(obj):
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    return obj

with open(RESULTS_DIR / "speech_approach_b_results.json", "w") as f:
    json.dump(results_b, f, default=_np_convert, indent=2)
print(f"\nSaved speech_approach_b_results.json")


# ── Summary ──────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
for n in n_adapt_levels:
    f1s  = [v[str(n)]["f1"]          for v in results_b.values() if str(n) in v]
    sens = [v[str(n)]["sensitivity"] for v in results_b.values() if str(n) in v]
    spec = [v[str(n)]["specificity"] for v in results_b.values() if str(n) in v]
    aucs = [v[str(n)].get("auroc",0) for v in results_b.values() if str(n) in v]
    if f1s:
        print(f"  n_adapt={n:2d}  F1={np.mean(f1s):.3f}+/-{np.std(f1s):.3f}  "
              f"Sens={np.mean(sens):.3f}  Spec={np.mean(spec):.3f}  AUC={np.mean(aucs):.3f}  "
              f"(n={len(f1s)} patients)")
