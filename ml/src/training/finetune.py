"""
Per-patient fine-tuning with LoRA adapters.

Fine-tunes the pre-trained T-MAE encoder with LoRA for each patient,
implementing the personalization stage of Approach B.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.transformer_mae import TemporalMAE
from src.models.personalized_detector import PersonalizedFoGDetector
from src.models.lora_adapter import save_lora_state, load_lora_state, reset_all_lora


def create_detector_from_pretrained(
    checkpoint_path: str,
    n_channels: int = 24,
    patch_size: int = 4,
    d_model: int = 128,
    n_heads: int = 4,
    n_encoder_layers: int = 4,
    d_ff: int = 256,
    dropout: float = 0.1,
    n_clinical: int = 9,
    lora_rank: int = 4,
    lora_alpha: float = 8.0,
    device: str = "auto",
) -> tuple[PersonalizedFoGDetector, dict]:
    """
    Create a PersonalizedFoGDetector from a pre-trained T-MAE checkpoint.

    Returns:
        detector: initialized detector with LoRA adapters
        normalization: dict with mean/std for input normalization
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load pre-trained MAE
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    mae = TemporalMAE(
        n_channels=n_channels, patch_size=patch_size, d_model=d_model,
        n_heads=n_heads, n_encoder_layers=n_encoder_layers,
        d_ff=d_ff, dropout=dropout,
    )
    mae.load_state_dict(checkpoint["model_state"])

    # Create detector and load encoder
    detector = PersonalizedFoGDetector(
        n_channels=n_channels, patch_size=patch_size, d_model=d_model,
        n_heads=n_heads, n_encoder_layers=n_encoder_layers,
        d_ff=d_ff, dropout=dropout, n_clinical=n_clinical,
        lora_rank=lora_rank, lora_alpha=lora_alpha,
    ).to(device)

    detector.load_pretrained_encoder(mae)
    detector.inject_lora_adapters()

    normalization = checkpoint.get("normalization", {"mean": 0, "std": 1})
    return detector, normalization


def finetune_patient(
    detector: PersonalizedFoGDetector,
    train_windows: np.ndarray,
    train_labels: np.ndarray,
    val_windows: np.ndarray | None = None,
    val_labels: np.ndarray | None = None,
    clinical_features: np.ndarray | None = None,
    normalization: dict | None = None,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    device: str = "auto",
) -> dict:
    """
    Fine-tune detector for a specific patient.

    Args:
        detector: PersonalizedFoGDetector with LoRA injected
        train_windows: (n_train, window_size, n_channels)
        train_labels: (n_train,)
        val_windows/labels: optional validation data
        clinical_features: (n_clinical,) patient's clinical features
        normalization: dict with 'mean' and 'std' for input normalization
        ...training hyperparameters...

    Returns:
        dict with training history and best validation metrics
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Reset LoRA for this patient
    reset_all_lora(detector)

    # Normalize inputs
    if normalization is not None:
        mean = normalization["mean"]
        std_val = normalization["std"]
        train_windows = (train_windows - mean) / std_val
        if val_windows is not None:
            val_windows = (val_windows - mean) / std_val

    # Prepare clinical features
    clinical_tensor = None
    if clinical_features is not None:
        clinical_tensor = torch.FloatTensor(clinical_features).unsqueeze(0).to(device)

    # Create dataloaders
    train_dataset = TensorDataset(
        torch.FloatTensor(train_windows),
        torch.FloatTensor(train_labels),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    # Optimizer for trainable params only (LoRA + classifier + clinical conditioner)
    trainable_params = detector.get_trainable_params()
    optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)

    # Class weight for imbalanced data
    n_pos = train_labels.sum()
    n_neg = len(train_labels) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)]).to(device)

    history = {"train_loss": [], "val_loss": [], "val_f1": []}
    best_val_f1 = 0.0
    best_lora_state = None

    for epoch in range(epochs):
        # Training
        detector.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # Expand clinical features to batch size
            batch_clinical = None
            if clinical_tensor is not None:
                batch_clinical = clinical_tensor.expand(len(batch_x), -1)

            optimizer.zero_grad()
            logits = detector(batch_x, batch_clinical).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, batch_y, pos_weight=pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / n_batches
        history["train_loss"].append(avg_train_loss)

        # Validation
        if val_windows is not None and len(val_windows) > 0:
            detector.eval()
            with torch.no_grad():
                val_x = torch.FloatTensor(val_windows).to(device)
                val_clinical = None
                if clinical_tensor is not None:
                    val_clinical = clinical_tensor.expand(len(val_x), -1)

                val_logits = detector(val_x, val_clinical).squeeze(-1)
                val_y = torch.FloatTensor(val_labels).to(device)
                val_loss = F.binary_cross_entropy_with_logits(val_logits, val_y).item()
                val_preds = (torch.sigmoid(val_logits) > 0.5).cpu().numpy()
                val_true = val_labels.astype(int)

                # F1 score
                tp = ((val_preds == 1) & (val_true == 1)).sum()
                fp = ((val_preds == 1) & (val_true == 0)).sum()
                fn = ((val_preds == 0) & (val_true == 1)).sum()
                precision = tp / max(tp + fp, 1)
                recall = tp / max(tp + fn, 1)
                f1 = 2 * precision * recall / max(precision + recall, 1e-8)

            history["val_loss"].append(val_loss)
            history["val_f1"].append(f1)

            if f1 > best_val_f1:
                best_val_f1 = f1
                best_lora_state = save_lora_state(detector)

    # Restore best LoRA state
    if best_lora_state is not None:
        load_lora_state(detector, best_lora_state)

    return {
        "history": history,
        "best_val_f1": best_val_f1,
        "lora_state": best_lora_state or save_lora_state(detector),
    }


def finetune_with_adaptation_curve(
    detector: PersonalizedFoGDetector,
    all_windows: np.ndarray,
    all_labels: np.ndarray,
    clinical_features: np.ndarray | None = None,
    normalization: dict | None = None,
    n_adaptation_samples: list[int] = [5, 10, 20, 50, 100],
    epochs_per_n: int = 30,
    device: str = "auto",
) -> dict[int, dict]:
    """
    Fine-tune with increasing amounts of patient data to generate
    an adaptation curve (performance vs n_adaptation_samples).

    Returns:
        dict mapping n_samples -> {lora_state, val_metrics}
    """
    results = {}

    for n in n_adaptation_samples:
        if n > len(all_windows):
            break

        # Use first n samples for adaptation, rest for evaluation
        train_w = all_windows[:n]
        train_l = all_labels[:n]
        val_w = all_windows[n:]
        val_l = all_labels[n:]

        if len(val_w) == 0:
            continue

        result = finetune_patient(
            detector, train_w, train_l, val_w, val_l,
            clinical_features, normalization,
            epochs=epochs_per_n, device=device,
        )
        results[n] = result
        print(f"  n={n}: best val F1 = {result['best_val_f1']:.4f}")

        # Reset LoRA for next round
        reset_all_lora(detector)

    return results
