"""
Online/streaming adaptation for deployed personalized FoG detector.

Simulates real-world deployment where labeled data arrives incrementally
and the model must adapt in real-time while avoiding catastrophic forgetting.
"""

import torch
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.personalized_detector import PersonalizedFoGDetector, OnlineAdapter
from src.models.lora_adapter import save_lora_state
from src.utils.metrics import compute_all_metrics


def simulate_online_deployment(
    detector: PersonalizedFoGDetector,
    windows: np.ndarray,
    labels: np.ndarray,
    clinical_features: np.ndarray | None = None,
    normalization: dict | None = None,
    chunk_size: int = 10,
    adaptation_steps: int = 5,
    adaptation_lr: float = 1e-4,
    buffer_size: int = 500,
    pseudo_label_threshold: float = 0.9,
    drift_threshold: float = 3.0,
    device: str = "auto",
) -> dict:
    """
    Simulate online deployment with streaming data.

    Data arrives in chunks of chunk_size windows. For each chunk:
    1. Predict (before seeing labels)
    2. Receive labels
    3. Adapt using new data + replay buffer
    4. Check for drift

    Args:
        detector: personalized detector with LoRA
        windows: (n_windows, window_size, n_channels) all test data
        labels: (n_windows,) ground truth (simulates delayed labeling)
        clinical_features: (n_clinical,) patient clinical features
        normalization: input normalization params
        chunk_size: number of windows per streaming chunk
        ...adaptation hyperparameters...

    Returns:
        dict with predictions, metrics over time, drift events
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Normalize
    if normalization is not None:
        windows = (windows - normalization["mean"]) / normalization["std"]

    # Prepare clinical
    clinical_tensor = None
    if clinical_features is not None:
        clinical_tensor = torch.FloatTensor(clinical_features).unsqueeze(0).to(device)

    # Initialize online adapter
    adapter = OnlineAdapter(
        detector,
        buffer_size=buffer_size,
        adaptation_lr=adaptation_lr,
        pseudo_label_threshold=pseudo_label_threshold,
        drift_threshold=drift_threshold,
    )

    # Streaming simulation
    n_windows = len(windows)
    n_chunks = (n_windows + chunk_size - 1) // chunk_size

    all_predictions = np.zeros(n_windows, dtype=np.int64)
    all_probabilities = np.zeros(n_windows, dtype=np.float32)
    chunk_metrics = []
    drift_events = []
    adaptation_losses = []

    for chunk_idx in range(n_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, n_windows)
        chunk_w = torch.FloatTensor(windows[start:end]).to(device)
        chunk_l = torch.FloatTensor(labels[start:end]).to(device)

        # 1. Predict (before adaptation)
        detector.eval()
        with torch.no_grad():
            chunk_clinical = None
            if clinical_tensor is not None:
                chunk_clinical = clinical_tensor.expand(len(chunk_w), -1)
            probs = detector.predict_proba(chunk_w, chunk_clinical)
            preds = (probs > 0.5).long()

        all_predictions[start:end] = preds.cpu().numpy()
        all_probabilities[start:end] = probs.cpu().numpy()

        # 2. Compute chunk metrics
        chunk_true = labels[start:end]
        chunk_pred = all_predictions[start:end]
        if len(np.unique(chunk_true)) > 1:
            chunk_metrics.append({
                "chunk": chunk_idx,
                "n_seen": end,
                **compute_all_metrics(chunk_true, chunk_pred),
            })

        # 3. Check for drift
        drift_score = adapter.check_drift(chunk_w)
        if adapter.should_readapt(chunk_w):
            drift_events.append({"chunk": chunk_idx, "drift_score": drift_score})

        # 4. Adapt
        loss = adapter.adapt_step(
            chunk_w, chunk_l, chunk_clinical,
            n_steps=adaptation_steps,
        )
        adaptation_losses.append(loss)

        # 5. Update feature distribution
        adapter.update_feature_distribution(chunk_w)

    # Overall metrics
    overall_metrics = compute_all_metrics(
        labels, all_predictions, all_probabilities
    )

    return {
        "predictions": all_predictions,
        "probabilities": all_probabilities,
        "overall_metrics": overall_metrics,
        "chunk_metrics": chunk_metrics,
        "drift_events": drift_events,
        "adaptation_losses": adaptation_losses,
        "lora_state": save_lora_state(detector),
    }
