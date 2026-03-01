"""
Evaluation metrics for FoG detection.

Includes window-level and event-level metrics.
"""

import numpy as np
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report,
)


def sensitivity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """True positive rate (recall for FoG class)."""
    return float(recall_score(y_true, y_pred, zero_division=0))


def specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """True negative rate."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                        y_prob: np.ndarray | None = None) -> dict:
    """
    Compute all window-level metrics.

    Args:
        y_true: ground truth binary labels
        y_pred: predicted binary labels
        y_prob: predicted probabilities (for AUROC)

    Returns:
        dict of metric name -> value
    """
    metrics = {
        "sensitivity": sensitivity(y_true, y_pred),
        "specificity": specificity(y_true, y_pred),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "n_samples": len(y_true),
        "n_positive": int(y_true.sum()),
        "n_predicted_positive": int(y_pred.sum()),
    }

    if y_prob is not None and len(np.unique(y_true)) > 1:
        metrics["auroc"] = float(roc_auc_score(y_true, y_prob))

    return metrics


def detect_fog_events(labels: np.ndarray) -> list[tuple[int, int]]:
    """
    Identify contiguous FoG=1 segments.

    Returns:
        list of (start_idx, end_idx) tuples
    """
    events = []
    in_event = False
    start = 0
    for i, val in enumerate(labels):
        if val == 1 and not in_event:
            in_event = True
            start = i
        elif val == 0 and in_event:
            in_event = False
            events.append((start, i - 1))
    if in_event:
        events.append((start, len(labels) - 1))
    return events


def event_level_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    overlap_threshold: float = 0.5,
) -> dict:
    """
    Compute event-level FoG detection metrics.

    A true FoG event is "detected" if at least overlap_threshold fraction
    of its windows are predicted as FoG.

    Args:
        y_true: ground truth labels
        y_pred: predicted labels
        overlap_threshold: minimum overlap to count as detected

    Returns:
        dict with event-level metrics
    """
    true_events = detect_fog_events(y_true)
    pred_events = detect_fog_events(y_pred)

    if not true_events:
        return {"n_true_events": 0, "n_pred_events": len(pred_events),
                "event_detection_rate": 0.0, "event_false_alarm_rate": 0.0}

    detected = 0
    detection_latencies = []

    for start, end in true_events:
        event_preds = y_pred[start:end + 1]
        overlap = event_preds.mean()
        if overlap >= overlap_threshold:
            detected += 1
            # Latency: index of first predicted FoG within event
            fog_indices = np.where(event_preds == 1)[0]
            if len(fog_indices) > 0:
                detection_latencies.append(fog_indices[0])

    # False alarms: predicted events that don't overlap with any true event
    false_alarms = 0
    for p_start, p_end in pred_events:
        overlaps_true = any(
            max(p_start, t_start) <= min(p_end, t_end)
            for t_start, t_end in true_events
        )
        if not overlaps_true:
            false_alarms += 1

    return {
        "n_true_events": len(true_events),
        "n_pred_events": len(pred_events),
        "n_detected": detected,
        "event_detection_rate": detected / len(true_events),
        "event_false_alarms": false_alarms,
        "mean_detection_latency_windows": float(np.mean(detection_latencies)) if detection_latencies else 0.0,
    }


def adaptation_curve_metrics(
    y_true: np.ndarray,
    y_pred_by_n: dict[int, np.ndarray],
    y_prob_by_n: dict[int, np.ndarray] | None = None,
) -> dict[int, dict]:
    """
    Compute metrics at different numbers of adaptation samples.

    Args:
        y_true: ground truth labels
        y_pred_by_n: dict mapping n_adaptation_samples -> predictions
        y_prob_by_n: optional dict mapping n -> probabilities

    Returns:
        dict mapping n -> metrics dict
    """
    results = {}
    for n, y_pred in y_pred_by_n.items():
        y_prob = y_prob_by_n[n] if y_prob_by_n and n in y_prob_by_n else None
        results[n] = compute_all_metrics(y_true, y_pred, y_prob)
    return results
