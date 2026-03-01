"""
Full evaluation pipeline: runs both Approach A and Approach B,
generates publication-ready figures and comparison charts.
"""

import sys, os, time, json, warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from pathlib import Path
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from src.data.fog_star_loader import FoGStarDataset, IMU_COLUMNS, SAMPLING_RATE
from src.data.windowing import create_windowed_dataset, lopo_split
from src.data.features import extract_batch_features, get_feature_names
from src.models.base_ensemble import SpecialistEnsemble
from src.models.bayesian_gating import BayesianGating, PersonalizedEnsemblePredictor
from src.models.transformer_mae import TemporalMAE
from src.models.personalized_detector import PersonalizedFoGDetector
from src.models.lora_adapter import get_lora_params, reset_all_lora, save_lora_state
from src.utils.clinical import ClinicalProfiler
from src.utils.metrics import compute_all_metrics, event_level_metrics
from src.utils.label_cleaning import clean_labels


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance — down-weights easy examples."""
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1 - targets) * (1 - probs)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal = alpha_t * (1 - pt) ** self.gamma * bce
        return focal.mean()


def find_optimal_threshold(y_true, y_prob, thresholds=None):
    """Find threshold that maximizes F1 score."""
    if thresholds is None:
        thresholds = np.arange(0.05, 0.95, 0.01)
    best_f1, best_t = 0, 0.5
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        f = f1_score(y_true, preds, zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_t = t
    return best_t, best_f1


def temperature_scale(logits_np: np.ndarray, labels_np: np.ndarray,
                      n_iter: int = 50) -> float:
    """
    Find a scalar temperature T that minimises NLL on calibration data.
    Calibrated probability = sigmoid(logit / T).
    Returns T (>1 = soften, <1 = sharpen).
    """
    T = torch.nn.Parameter(torch.ones(1))
    logits = torch.FloatTensor(logits_np).unsqueeze(1)
    labels = torch.FloatTensor(labels_np).unsqueeze(1)
    opt = torch.optim.LBFGS([T], lr=0.1, max_iter=n_iter)

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(logits / T.clamp(min=0.1), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(T.detach().clamp(min=0.1))

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("results/figures")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ─── Plotting style ────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLORS = {
    "baseline": "#7f8c8d",
    "approach_a_0": "#3498db",
    "approach_a_adapt": "#2980b9",
    "approach_b_0": "#e74c3c",
    "approach_b_adapt": "#c0392b",
    "fog": "#e74c3c",
    "nonfog": "#2ecc71",
}


# ═══════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════════

def load_data():
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    ds = FoGStarDataset("datasets/fog").load()
    print(ds.summary())

    print("\nExtracting windows (2s, 50% overlap)...")
    windowed = create_windowed_dataset(ds, window_seconds=2.0, overlap=0.5)
    print(f"  Total windows: {len(windowed.labels):,}")
    print(f"  FoG windows: {windowed.labels.sum():,} ({windowed.labels.mean()*100:.1f}%)")

    clinical_features, clinical_ids = ds.get_all_clinical_features()
    return ds, windowed, clinical_features, clinical_ids


# ═══════════════════════════════════════════════════════════════════
#  APPROACH A: BAYESIAN PERSONALIZED ENSEMBLE
# ═══════════════════════════════════════════════════════════════════

def run_approach_a(ds, windowed, clinical_features, clinical_ids, all_features=None):
    print("\n" + "=" * 70)
    print("APPROACH A: BAYESIAN PERSONALIZED ENSEMBLE (LOPO)")
    print("=" * 70)

    n_adaptation = [0, 10, 20, 50]
    results = {}

    # Extract features ONCE for all windows (the big optimization)
    if all_features is None:
        print("Extracting features for all windows (one-time cost)...")
        t0 = time.time()
        all_features = extract_batch_features(windowed.windows)
        all_features = np.nan_to_num(all_features, nan=0.0, posinf=0.0, neginf=0.0)
        print(f"  Done: {all_features.shape} in {time.time()-t0:.1f}s")

    for fold_idx, test_sid in enumerate(ds.subject_ids):
        print(f"\n[{fold_idx+1}/{ds.n_subjects}] Patient {test_sid} held out...")

        # Split using indices
        test_mask = windowed.subject_ids == test_sid
        train_mask = ~test_mask

        train_labels = windowed.labels[train_mask]
        test_labels = windowed.labels[test_mask]

        if len(test_labels) == 0 or test_labels.sum() == 0:
            print("  Skipping (no FoG in test)")
            continue

        train_feat = all_features[train_mask]
        test_feat = all_features[test_mask]
        train_activities = windowed.activities[train_mask] if len(windowed.activities) > 0 else None

        # Normalize features per fold — RobustScaler is less sensitive to outlier patients
        scaler = RobustScaler()
        train_feat = scaler.fit_transform(train_feat)
        test_feat = scaler.transform(test_feat)

        # Clinical profiler for similarity-based instance weighting
        train_sids = np.unique(windowed.subject_ids[train_mask])
        train_clin_mask = np.isin(clinical_ids, train_sids)
        profiler = ClinicalProfiler().fit(clinical_features[train_clin_mask],
                                          [c for c in clinical_ids if c in train_sids])

        # Clinical similarity weights: weight each training window by how similar
        # its patient is to the test patient
        try:
            test_clin_feat = ds.get_clinical_features(test_sid)
            sim_weights = profiler.get_similarity_weights(test_clin_feat, bandwidth=1.5)
            # Map per-patient weights to per-window weights
            train_sids_per_window = windowed.subject_ids[train_mask]
            train_instance_weights = np.ones(len(train_labels), dtype=np.float32)
            for i, sid in enumerate(profiler.subject_ids):
                sid_mask = train_sids_per_window == sid
                train_instance_weights[sid_mask] = float(sim_weights[i])
            # Normalize to mean=1 so overall scale doesn't change
            mean_w = train_instance_weights.mean()
            if mean_w > 0:
                train_instance_weights /= mean_w
        except Exception:
            train_instance_weights = np.ones(len(train_labels), dtype=np.float32)

        # Label cleaning on training fold
        train_windows_raw = windowed.windows[train_mask]
        train_labels_clean, ambiguous_mask, lc_weights = clean_labels(
            train_labels, train_windows_raw, fs=60.0,
            bridge_gaps=True, max_gap=3,
            remove_isolated=True, min_fog_duration=2,
            flag_ambiguous=True, fi_threshold=2.5, context_radius=5,
        )

        # Combine similarity weights with label-cleaning weights
        combined_weights = train_instance_weights * lc_weights
        # Ensure min weight of 0 (ambiguous) and sensible max
        combined_weights = np.clip(combined_weights, 0.0, 10.0)

        # Dynamic scale_pos_weight per fold (neg/pos ratio)
        n_pos_fold = float(train_labels_clean.sum())
        n_neg_fold = float(len(train_labels_clean) - n_pos_fold)
        spw = max(1.0, n_neg_fold / n_pos_fold) if n_pos_fold > 0 else 4.0

        # Train specialists with combined sample weights + dynamic pos weighting
        ensemble = SpecialistEnsemble(xgb_params={
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.08,
            "scale_pos_weight": spw,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "eval_metric": "logloss",
            "tree_method": "hist",
            "random_state": 42,
        })
        ensemble.fit(train_feat, train_labels_clean, train_activities,
                     sample_weights=combined_weights)

        # Find optimal threshold on training data
        _, train_probs = ensemble.predict_uniform(train_feat)
        opt_threshold, _ = find_optimal_threshold(train_labels_clean, train_probs)
        print(f"  Optimal threshold: {opt_threshold:.2f}")

        # Compute clinical prior once (reused across n_adapt levels)
        try:
            _test_clin = ds.get_clinical_features(test_sid)
            dirichlet_prior = profiler.get_dirichlet_prior(_test_clin, {}, ensemble.n_specialists)
        except Exception:
            dirichlet_prior = None

        patient_results = {}
        for n_adapt in n_adaptation:
            prior = dirichlet_prior

            gating = BayesianGating(ensemble.n_specialists, prior)
            predictor = PersonalizedEnsemblePredictor(ensemble, gating)

            if n_adapt > 0 and n_adapt <= len(test_feat):
                predictor.adapt(test_feat[:n_adapt], test_labels[:n_adapt])
                eval_feat = test_feat[n_adapt:]
                eval_labels = test_labels[n_adapt:]
            else:
                eval_feat = test_feat
                eval_labels = test_labels

            if len(eval_labels) == 0:
                continue

            _, probs = predictor.predict(eval_feat)
            # Use optimized threshold instead of fixed 0.5
            preds = (probs >= opt_threshold).astype(np.int64)
            metrics = compute_all_metrics(eval_labels, preds, probs)
            ev_metrics = event_level_metrics(eval_labels, preds)
            patient_results[n_adapt] = {**metrics, **ev_metrics,
                                         "weights": gating.mixture_weights.tolist()}

        results[test_sid] = patient_results
        best_f1 = max((v.get("f1", 0) for v in patient_results.values()), default=0)
        print(f"  Best F1: {best_f1:.3f}")

    return results, n_adaptation, all_features


# ═══════════════════════════════════════════════════════════════════
#  APPROACH B: SSL PRE-TRAINING + LoRA PERSONALIZATION
# ═══════════════════════════════════════════════════════════════════

def run_approach_b(ds, windowed, clinical_features, clinical_ids):
    print("\n" + "=" * 70)
    print("APPROACH B: SSL + LoRA PERSONALIZATION (LOPO)")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    n_adaptation = [0, 10, 20, 50]
    results = {}

    # Global normalization
    all_windows = windowed.windows
    global_mean = all_windows.mean(axis=(0, 1), keepdims=True).astype(np.float32)
    global_std = all_windows.std(axis=(0, 1), keepdims=True).astype(np.float32)
    global_std[global_std == 0] = 1.0
    all_w_norm = ((all_windows - global_mean) / global_std).astype(np.float32)

    # ── Step 1: Pre-train MAE ONCE on all data (SSL - no labels used) ──
    print("\n  Pre-training T-MAE on all windows (SSL)...")
    t0 = time.time()
    mae = TemporalMAE(n_channels=24, patch_size=4, d_model=128,
                      n_heads=4, n_encoder_layers=4, n_decoder_layers=2,
                      d_ff=256, dropout=0.1, mask_ratio=0.5).to(device)

    pretrain_ds = TensorDataset(torch.FloatTensor(all_w_norm))
    pretrain_loader = DataLoader(pretrain_ds, batch_size=256, shuffle=True,
                                 drop_last=True, num_workers=0, pin_memory=True)

    optimizer = torch.optim.AdamW(mae.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    mae.train()
    for epoch in range(50):
        epoch_loss = 0
        n_batches = 0
        for (batch,) in pretrain_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss, _, _ = mae(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mae.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/50 | Loss: {epoch_loss/n_batches:.5f}")

    mae_state = mae.state_dict()
    print(f"  Pre-training done in {time.time()-t0:.1f}s")

    # Focal loss for better class imbalance handling
    focal_loss = FocalLoss(alpha=0.75, gamma=2.0)

    # ── Step 2: LOPO evaluation ──
    for fold_idx, test_sid in enumerate(ds.subject_ids):
        print(f"\n[{fold_idx+1}/{ds.n_subjects}] Patient {test_sid} held out...")

        test_mask = windowed.subject_ids == test_sid
        train_mask = ~test_mask
        test_labels = windowed.labels[test_mask]

        if len(test_labels) == 0 or test_labels.sum() == 0:
            print("  Skipping (no FoG in test)")
            continue

        t0 = time.time()
        train_w = all_w_norm[train_mask]
        test_w = all_w_norm[test_mask]
        test_l = test_labels

        # Label cleaning on training fold (uses raw windows for FI computation)
        train_labels_raw = windowed.labels[train_mask]
        train_windows_raw = windowed.windows[train_mask]
        train_labels_clean, ambiguous_b, lc_weights_b = clean_labels(
            train_labels_raw, train_windows_raw, fs=60.0,
            bridge_gaps=True, max_gap=3,
            remove_isolated=True, min_fog_duration=2,
            flag_ambiguous=True, fi_threshold=2.5, context_radius=5,
        )
        train_l = train_labels_clean.astype(np.float32)

        # Create detector from pre-trained MAE — larger LoRA rank for more capacity
        detector = PersonalizedFoGDetector(
            n_channels=24, patch_size=4, d_model=128, n_heads=4,
            n_encoder_layers=4, d_ff=256, dropout=0.1,
            n_clinical=9, lora_rank=8, lora_alpha=16.0,
        ).to(device)
        detector.load_pretrained_encoder(mae)
        detector.inject_lora_adapters()

        # Also unfreeze the last encoder layer for more capacity
        encoder_layers = list(detector.encoder.encoder.layers)
        for param in encoder_layers[-1].parameters():
            param.requires_grad = True

        # Clinical features
        try:
            clin = torch.FloatTensor(ds.get_clinical_features(test_sid)).to(device)
            clin = torch.nan_to_num(clin, nan=0.0)
        except Exception:
            clin = torch.zeros(9).to(device)

        # ── Step 3: Supervised fine-tune on training data (once per fold) ──
        # Gather ALL trainable params (LoRA + classifier + conditioner + last encoder layer)
        trainable = list(detector.get_trainable_params())
        trainable_ids = {id(p) for p in trainable}
        for param in encoder_layers[-1].parameters():
            if param.requires_grad and id(param) not in trainable_ids:
                trainable.append(param)
                trainable_ids.add(id(param))

        ft_optimizer = torch.optim.AdamW(trainable, lr=1e-3, weight_decay=1e-4)
        ft_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(ft_optimizer, T_max=30)

        # Balanced sampling: oversample minority class, also zero-weight ambiguous windows
        sample_weights_b = lc_weights_b.copy()  # start with label-cleaning weights (0 for ambiguous)
        n_pos = float(train_l.sum())
        n_neg = float(len(train_l) - n_pos)
        if n_pos > 0 and n_neg > 0:
            sample_weights_b[train_l == 1] *= (n_neg / n_pos)
        # Add tiny epsilon so sampler doesn't crash if all-zero
        sample_weights_b = np.clip(sample_weights_b, 1e-6, None)
        sampler = WeightedRandomSampler(sample_weights_b, num_samples=len(train_l), replacement=True)

        ft_dataset = TensorDataset(torch.FloatTensor(train_w), torch.FloatTensor(train_l))
        ft_loader = DataLoader(ft_dataset, batch_size=128, sampler=sampler,
                               drop_last=True, num_workers=0, pin_memory=True)

        detector.train()
        for epoch in range(30):
            for bx, by in ft_loader:
                bx, by = bx.to(device), by.to(device)
                bc = clin.unsqueeze(0).expand(len(bx), -1)
                ft_optimizer.zero_grad()
                logits = detector(bx, bc).squeeze(-1)
                loss = focal_loss(logits, by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                ft_optimizer.step()
            ft_scheduler.step()

        # ── Temperature scaling calibration on training predictions ──
        # Collects raw logits, fits scalar T minimising NLL → better calibrated probs
        detector.eval()
        with torch.no_grad():
            train_logits_list = []
            train_probs_list = []
            for i in range(0, len(train_w), 256):
                bw = torch.FloatTensor(train_w[i:i+256]).to(device)
                bc = clin.unsqueeze(0).expand(len(bw), -1)
                raw_logits = detector(bw, bc).squeeze(-1).cpu().numpy()
                train_logits_list.append(raw_logits)
                train_probs_list.append(torch.sigmoid(torch.FloatTensor(raw_logits)).numpy())
            train_logits_all = np.concatenate(train_logits_list)
            train_probs = np.concatenate(train_probs_list)

        # Fit temperature on training data
        temp_T = temperature_scale(train_logits_all, train_l.astype(np.float32))
        # Calibrated probs
        train_probs_cal = torch.sigmoid(
            torch.FloatTensor(train_logits_all) / temp_T
        ).numpy()
        opt_threshold, _ = find_optimal_threshold(train_l, train_probs_cal)
        print(f"  Temperature: {temp_T:.3f} | Optimal threshold: {opt_threshold:.2f}")

        # Save base fine-tuned state (for resetting between adaptation experiments)
        base_state = {k: v.clone() for k, v in detector.state_dict().items()}

        # EWC: save reference parameters + compute diagonal Fisher importance
        # Fisher diagonal estimated from training data gradient variance
        ewc_anchors = {k: v.clone().detach() for k, v in detector.state_dict().items()
                       if v.dtype == torch.float32}

        # Diagonal Fisher (importance weights per parameter)
        # Use a subsample of training data for efficiency
        fisher_diag = {}
        detector.eval()
        adapt_params_ref = [p for p in detector.parameters() if p.requires_grad]
        for p in adapt_params_ref:
            fisher_diag[id(p)] = torch.zeros_like(p.data)

        n_fisher_samples = min(200, len(train_w))
        fisher_indices = np.random.choice(len(train_w), n_fisher_samples, replace=False)
        for fi in fisher_indices:
            bw = torch.FloatTensor(train_w[fi:fi+1]).to(device)
            bc = clin.unsqueeze(0).to(device)
            logits = detector(bw, bc).squeeze(-1)
            prob = torch.sigmoid(logits)
            # Log-likelihood gradient squared = Fisher diagonal estimate
            ll = prob * float(train_l[fi]) + (1 - prob) * float(1 - train_l[fi])
            ll = torch.log(ll.clamp(min=1e-7))
            detector.zero_grad()
            ll.backward()
            for p in adapt_params_ref:
                if p.grad is not None:
                    fisher_diag[id(p)] += p.grad.data.pow(2)
        for p in adapt_params_ref:
            fisher_diag[id(p)] /= n_fisher_samples
            fisher_diag[id(p)] = fisher_diag[id(p)].detach()

        patient_results = {}
        for n_adapt in n_adaptation:
            # Restore to base fine-tuned state
            detector.load_state_dict(base_state)

            if n_adapt > 0 and n_adapt <= len(test_w):
                adapt_w = torch.FloatTensor(test_w[:n_adapt]).to(device)
                adapt_l = torch.FloatTensor(test_l[:n_adapt].astype(np.float32)).to(device)
                eval_w = test_w[n_adapt:]
                eval_l = test_l[n_adapt:]

                # ── Step 4: Patient-specific adaptation with Fisher EWC ──
                # Only adapt LoRA params + classifier head
                adapt_params = list(detector.get_trainable_params())
                adapt_param_ids = {id(p) for p in adapt_params}
                adapt_optimizer = torch.optim.Adam(adapt_params, lr=3e-4)
                detector.train()
                adapt_focal = FocalLoss(alpha=0.75, gamma=2.0)

                # Class-balanced mini-batches during adaptation
                n_fog_adapt = int(adapt_l.sum().item())
                n_nfog_adapt = len(adapt_l) - n_fog_adapt

                # Adaptive step count
                n_steps = min(20, max(5, n_adapt))
                ewc_lambda = 10.0  # stronger EWC with Fisher weighting

                for step in range(n_steps):
                    # Build balanced mini-batch from adapt data
                    if n_fog_adapt > 0 and n_nfog_adapt > 0:
                        fog_idx = (adapt_l == 1).nonzero(as_tuple=True)[0]
                        nfog_idx = (adapt_l == 0).nonzero(as_tuple=True)[0]
                        n_take = min(len(fog_idx), len(nfog_idx), 8)
                        perm_fog = torch.randperm(len(fog_idx))[:n_take]
                        perm_nfog = torch.randperm(len(nfog_idx))[:n_take]
                        batch_idx = torch.cat([fog_idx[perm_fog], nfog_idx[perm_nfog]])
                        bw_a = adapt_w[batch_idx]
                        bl_a = adapt_l[batch_idx]
                    else:
                        bw_a, bl_a = adapt_w, adapt_l

                    bc = clin.unsqueeze(0).expand(len(bw_a), -1)
                    adapt_optimizer.zero_grad()
                    logits = detector(bw_a, bc).squeeze(-1)
                    task_loss = adapt_focal(logits, bl_a)

                    # Fisher-weighted EWC penalty
                    ewc_loss = torch.tensor(0.0, device=device)
                    for name, param in detector.named_parameters():
                        if param.requires_grad and name in ewc_anchors and id(param) in fisher_diag:
                            anchor = ewc_anchors[name].to(device)
                            fim = fisher_diag[id(param)].to(device)
                            ewc_loss = ewc_loss + (fim * (param - anchor) ** 2).sum()

                    loss = task_loss + ewc_lambda * ewc_loss
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(adapt_params, 0.5)
                    adapt_optimizer.step()
            else:
                eval_w = test_w
                eval_l = test_l

            # ── Step 5: Evaluate with temperature-calibrated probabilities ──
            detector.eval()
            with torch.no_grad():
                all_logits = []
                for i in range(0, len(eval_w), 256):
                    batch_w = torch.FloatTensor(eval_w[i:i+256]).to(device)
                    bc = clin.unsqueeze(0).expand(len(batch_w), -1)
                    raw = detector(batch_w, bc).squeeze(-1).cpu().numpy()
                    all_logits.append(raw)
                logits_eval = np.concatenate(all_logits)
                # Apply temperature calibration
                probs = torch.sigmoid(
                    torch.FloatTensor(logits_eval) / temp_T
                ).numpy()
                preds = (probs >= opt_threshold).astype(np.int64)

            if len(eval_l) == 0:
                continue

            metrics = compute_all_metrics(eval_l, preds, probs)
            ev_metrics = event_level_metrics(eval_l, preds)
            patient_results[n_adapt] = {**metrics, **ev_metrics}

        results[test_sid] = patient_results
        elapsed = time.time() - t0
        best_f1 = max((v.get("f1", 0) for v in patient_results.values()), default=0)
        print(f"  Time: {elapsed:.1f}s | Best F1: {best_f1:.3f}")

        del detector, base_state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    del mae
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results, n_adaptation


# ═══════════════════════════════════════════════════════════════════
#  VISUALIZATION
# ═══════════════════════════════════════════════════════════════════

def _collect_metric(results, n_adapt, metric_name):
    """Collect a metric across patients for a given n_adapt."""
    vals = []
    sids = []
    for sid, patient_res in results.items():
        if n_adapt in patient_res and metric_name in patient_res[n_adapt]:
            vals.append(patient_res[n_adapt][metric_name])
            sids.append(sid)
    return np.array(vals), sids


def fig1_per_patient_f1(results_a, results_b, n_adapt_list):
    """Fig 1: Per-patient F1 comparison across approaches."""
    fig, axes = plt.subplots(1, len(n_adapt_list), figsize=(5 * len(n_adapt_list), 5),
                             sharey=True)
    if len(n_adapt_list) == 1:
        axes = [axes]

    for ax_idx, n_adapt in enumerate(n_adapt_list):
        ax = axes[ax_idx]
        f1_a, sids_a = _collect_metric(results_a, n_adapt, "f1")
        f1_b, sids_b = _collect_metric(results_b, n_adapt, "f1")

        # Align patients present in both
        common = sorted(set(sids_a) & set(sids_b))
        if not common:
            continue

        fa = [f1_a[sids_a.index(s)] for s in common]
        fb = [f1_b[sids_b.index(s)] for s in common]

        x = np.arange(len(common))
        width = 0.35
        ax.bar(x - width/2, fa, width, color="#3498db", label="Approach A (Ensemble)")
        ax.bar(x + width/2, fb, width, color="#e74c3c", label="Approach B (SSL+LoRA)")

        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in common], rotation=45, ha="right")
        ax.set_xlabel("Patient ID")
        ax.set_title(f"n_adapt = {n_adapt}")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        if ax_idx == 0:
            ax.set_ylabel("F1 Score")
            ax.legend(loc="upper right")

    fig.suptitle("Per-Patient F1 Score Comparison (LOPO)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig1_per_patient_f1.png")
    plt.close(fig)
    print(f"  Saved fig1_per_patient_f1.png")


def fig2_adaptation_curve(results_a, results_b, n_adapt_list):
    """Fig 2: Adaptation curve — F1 vs number of adaptation samples."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for metric_idx, (metric, title) in enumerate([
        ("f1", "F1 Score"), ("sensitivity", "Sensitivity"), ("specificity", "Specificity")
    ]):
        ax = axes[metric_idx]

        for label, results, color, marker in [
            ("Approach A", results_a, "#3498db", "o"),
            ("Approach B", results_b, "#e74c3c", "s"),
        ]:
            means = []
            stds = []
            valid_n = []
            for n in n_adapt_list:
                vals, _ = _collect_metric(results, n, metric)
                if len(vals) > 0:
                    means.append(np.mean(vals))
                    stds.append(np.std(vals))
                    valid_n.append(n)

            if means:
                means = np.array(means)
                stds = np.array(stds)
                ax.plot(valid_n, means, f"-{marker}", color=color, label=label,
                        linewidth=2, markersize=8)
                ax.fill_between(valid_n, means - stds, means + stds,
                                alpha=0.15, color=color)

        ax.set_xlabel("Adaptation Samples (n)")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.set_ylim(0, 1.05)

    fig.suptitle("Adaptation Curve: Performance vs. Personalization Data", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig2_adaptation_curve.png")
    plt.close(fig)
    print(f"  Saved fig2_adaptation_curve.png")


def fig3_boxplot_comparison(results_a, results_b, n_adapt_list):
    """Fig 3: Box plots of F1, Sensitivity, Specificity across patients."""
    metrics = ["f1", "sensitivity", "specificity"]
    titles = ["F1 Score", "Sensitivity", "Specificity"]

    # Pick two interesting n_adapt values: 0 (no adapt) and max
    n_vals = [0, max(n_adapt_list)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for m_idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[m_idx]
        data_list = []
        labels_list = []
        colors_list = []

        for n_adapt in n_vals:
            va, _ = _collect_metric(results_a, n_adapt, metric)
            vb, _ = _collect_metric(results_b, n_adapt, metric)
            if len(va) > 0:
                data_list.append(va)
                labels_list.append(f"A (n={n_adapt})")
                colors_list.append("#3498db")
            if len(vb) > 0:
                data_list.append(vb)
                labels_list.append(f"B (n={n_adapt})")
                colors_list.append("#e74c3c")

        if data_list:
            bp = ax.boxplot(data_list, patch_artist=True, labels=labels_list,
                           medianprops=dict(color="black", linewidth=2))
            for patch, color in zip(bp["boxes"], colors_list):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)

        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        if m_idx == 0:
            ax.set_ylabel("Score")

    fig.suptitle("Distribution of Metrics Across Patients (LOPO)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig3_boxplot_comparison.png")
    plt.close(fig)
    print(f"  Saved fig3_boxplot_comparison.png")


def fig4_event_detection(results_a, results_b, best_n):
    """Fig 4: Event-level detection rates."""
    fig, ax = plt.subplots(figsize=(8, 5))

    edr_a, sids_a = _collect_metric(results_a, best_n, "event_detection_rate")
    edr_b, sids_b = _collect_metric(results_b, best_n, "event_detection_rate")

    common = sorted(set(sids_a) & set(sids_b))
    if not common:
        plt.close(fig)
        return

    ea = [edr_a[sids_a.index(s)] for s in common]
    eb = [edr_b[sids_b.index(s)] for s in common]

    x = np.arange(len(common))
    width = 0.35
    ax.bar(x - width/2, ea, width, color="#3498db", label="Approach A (Ensemble)")
    ax.bar(x + width/2, eb, width, color="#e74c3c", label="Approach B (SSL+LoRA)")

    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in common], rotation=45, ha="right")
    ax.set_xlabel("Patient ID")
    ax.set_ylabel("Event Detection Rate")
    ax.set_title(f"FoG Event Detection Rate per Patient (n_adapt={best_n})")
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig4_event_detection.png")
    plt.close(fig)
    print(f"  Saved fig4_event_detection.png")


def fig5_specialist_weights(results_a, best_n):
    """Fig 5: Specialist weight distribution across patients (Approach A)."""
    specialist_names = ["Walk", "Turn", "Transition", "Dual-task", "General"]

    weights_data = []
    sids = []
    for sid, patient_res in results_a.items():
        if best_n in patient_res and "weights" in patient_res[best_n]:
            weights_data.append(patient_res[best_n]["weights"])
            sids.append(sid)

    if not weights_data:
        return

    weights_arr = np.array(weights_data)  # (n_patients, n_specialists)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Stacked bar chart
    bottom = np.zeros(len(sids))
    colors_spec = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]
    for i, name in enumerate(specialist_names):
        ax1.bar([str(s) for s in sids], weights_arr[:, i], bottom=bottom,
                color=colors_spec[i], label=name)
        bottom += weights_arr[:, i]

    ax1.set_xlabel("Patient ID")
    ax1.set_ylabel("Mixture Weight")
    ax1.set_title("Per-Patient Specialist Weights")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.tick_params(axis="x", rotation=45)

    # Heatmap
    im = ax2.imshow(weights_arr.T, aspect="auto", cmap="YlOrRd")
    ax2.set_xticks(range(len(sids)))
    ax2.set_xticklabels([str(s) for s in sids], rotation=45, ha="right")
    ax2.set_yticks(range(len(specialist_names)))
    ax2.set_yticklabels(specialist_names)
    ax2.set_xlabel("Patient ID")
    ax2.set_title("Specialist Weight Heatmap")
    plt.colorbar(im, ax=ax2, label="Weight")

    fig.suptitle("Bayesian Gating: Personalized Specialist Routing", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig5_specialist_weights.png")
    plt.close(fig)
    print(f"  Saved fig5_specialist_weights.png")


def fig6_summary_table(results_a, results_b, n_adapt_list):
    """Fig 6: Summary table as a figure."""
    metrics = ["f1", "sensitivity", "specificity", "auroc"]
    metric_labels = ["F1", "Sensitivity", "Specificity", "AUROC"]

    rows = []
    for n_adapt in n_adapt_list:
        for label, results, in [("Ensemble (A)", results_a), ("SSL+LoRA (B)", results_b)]:
            row = {"Approach": label, "n_adapt": n_adapt}
            for metric, mlabel in zip(metrics, metric_labels):
                vals, _ = _collect_metric(results, n_adapt, metric)
                if len(vals) > 0:
                    row[f"{mlabel} Mean"] = f"{np.mean(vals):.3f}"
                    row[f"{mlabel} Std"] = f"{np.std(vals):.3f}"
                else:
                    row[f"{mlabel} Mean"] = "N/A"
                    row[f"{mlabel} Std"] = "N/A"
            rows.append(row)

    df = pd.DataFrame(rows)

    # Create table figure
    fig, ax = plt.subplots(figsize=(16, 0.5 + 0.4 * len(rows)))
    ax.axis("off")

    # Build cell text
    col_labels = ["Approach", "n_adapt"]
    for ml in metric_labels:
        col_labels.extend([f"{ml} Mean", f"{ml} Std"])

    cell_text = []
    for _, row in df.iterrows():
        cell_text.append([row.get(c, "") for c in col_labels])

    table = ax.table(cellText=cell_text, colLabels=col_labels,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)

    # Color header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#2c3e50")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Alternate row colors
    for i in range(1, len(cell_text) + 1):
        for j in range(len(col_labels)):
            if (i - 1) % 4 < 2:
                table[i, j].set_facecolor("#ebf5fb")  # light blue for A
            else:
                table[i, j].set_facecolor("#fdedec")  # light red for B

    fig.suptitle("LOPO Evaluation Summary (Mean +/- Std across 22 patients)",
                 fontsize=13, y=0.98)
    fig.savefig(FIGURES_DIR / "fig6_summary_table.png")
    plt.close(fig)
    print(f"  Saved fig6_summary_table.png")

    return df


def fig7_improvement_scatter(results_a, results_b, n_adapt):
    """Fig 7: Scatter — Approach A F1 vs Approach B F1 per patient."""
    f1_a, sids_a = _collect_metric(results_a, n_adapt, "f1")
    f1_b, sids_b = _collect_metric(results_b, n_adapt, "f1")

    common = sorted(set(sids_a) & set(sids_b))
    if len(common) < 3:
        return

    fa = np.array([f1_a[sids_a.index(s)] for s in common])
    fb = np.array([f1_b[sids_b.index(s)] for s in common])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(fa, fb, s=80, c=["#2ecc71" if b > a else "#e74c3c" for a, b in zip(fa, fb)],
               edgecolors="black", linewidth=0.5, zorder=5)

    for i, sid in enumerate(common):
        ax.annotate(str(sid), (fa[i], fb[i]), textcoords="offset points",
                    xytext=(5, 5), fontsize=8)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Equal performance")
    ax.set_xlabel("Approach A (Ensemble) F1")
    ax.set_ylabel("Approach B (SSL+LoRA) F1")
    ax.set_title(f"Head-to-Head F1 Comparison (n_adapt={n_adapt})")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(handles=[
        Patch(facecolor="#2ecc71", label="B wins"),
        Patch(facecolor="#e74c3c", label="A wins"),
    ])

    n_b_wins = (fb > fa).sum()
    ax.text(0.05, 0.95, f"B wins: {n_b_wins}/{len(common)} patients",
            transform=ax.transAxes, fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig7_improvement_scatter.png")
    plt.close(fig)
    print(f"  Saved fig7_improvement_scatter.png")


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    overall_start = time.time()

    # Load data
    ds, windowed, clinical_features, clinical_ids = load_data()

    # Run Approach A
    results_a, n_adapt_list, all_features = run_approach_a(ds, windowed, clinical_features, clinical_ids)

    # Run Approach B
    results_b, _ = run_approach_b(ds, windowed, clinical_features, clinical_ids)

    # Save raw results
    def _np_convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    for name, res in [("approach_a", results_a), ("approach_b", results_b)]:
        with open(RESULTS_DIR / f"{name}_results.json", "w") as f:
            json.dump(res, default=_np_convert, fp=f, indent=2)

    # Generate figures
    print("\n" + "=" * 70)
    print("GENERATING FIGURES")
    print("=" * 70)

    fig1_per_patient_f1(results_a, results_b, n_adapt_list)
    fig2_adaptation_curve(results_a, results_b, n_adapt_list)
    fig3_boxplot_comparison(results_a, results_b, n_adapt_list)
    best_n = max(n_adapt_list)
    fig4_event_detection(results_a, results_b, best_n)
    fig5_specialist_weights(results_a, best_n)
    summary_df = fig6_summary_table(results_a, results_b, n_adapt_list)
    fig7_improvement_scatter(results_a, results_b, best_n)

    total_time = time.time() - overall_start
    print(f"\nTotal evaluation time: {total_time/60:.1f} minutes")
    print(f"Results saved to: {RESULTS_DIR}")
    print(f"Figures saved to: {FIGURES_DIR}")

    # Print quick summary
    print("\n" + "=" * 70)
    print("QUICK SUMMARY")
    print("=" * 70)
    for n_adapt in n_adapt_list:
        f1_a, _ = _collect_metric(results_a, n_adapt, "f1")
        f1_b, _ = _collect_metric(results_b, n_adapt, "f1")
        a_str = f"A: {np.mean(f1_a):.3f}+/-{np.std(f1_a):.3f}" if len(f1_a) > 0 else "A: N/A"
        b_str = f"B: {np.mean(f1_b):.3f}+/-{np.std(f1_b):.3f}" if len(f1_b) > 0 else "B: N/A"
        print(f"  n_adapt={n_adapt:3d} | {a_str} | {b_str}")
