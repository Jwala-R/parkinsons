"""
Sliding window extraction for time-series sensor data.

Extracts fixed-length windows with configurable overlap,
assigns per-window labels via majority voting, and provides
train/test splits for LOPO (leave-one-patient-out) evaluation.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class WindowedData:
    """Container for windowed sensor data."""
    windows: np.ndarray      # (n_windows, window_size, n_channels)
    labels: np.ndarray       # (n_windows,) binary FoG labels
    subject_ids: np.ndarray  # (n_windows,) subject ID per window
    activities: np.ndarray   # (n_windows,) majority activity per window


def extract_windows(
    signal: np.ndarray,
    labels: np.ndarray,
    window_size: int,
    step_size: int,
    activities: np.ndarray | None = None,
    label_threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Extract sliding windows from a single continuous signal.

    Args:
        signal: (n_samples, n_channels) sensor data
        labels: (n_samples,) binary labels
        window_size: number of samples per window
        step_size: step between consecutive windows
        activities: optional (n_samples,) activity labels
        label_threshold: fraction of FoG samples for a window to be labeled FoG

    Returns:
        windows: (n_windows, window_size, n_channels)
        window_labels: (n_windows,) binary labels
        window_activities: (n_windows,) or None
    """
    n_samples = len(signal)
    if n_samples < window_size:
        return np.empty((0, window_size, signal.shape[1])), np.empty(0, dtype=np.int64), None

    starts = np.arange(0, n_samples - window_size + 1, step_size)
    n_windows = len(starts)

    windows = np.zeros((n_windows, window_size, signal.shape[1]), dtype=np.float32)
    window_labels = np.zeros(n_windows, dtype=np.int64)
    window_activities = np.zeros(n_windows, dtype=np.int64) if activities is not None else None

    for i, start in enumerate(starts):
        end = start + window_size
        windows[i] = signal[start:end]

        # Majority voting for label
        fog_ratio = labels[start:end].mean()
        window_labels[i] = 1 if fog_ratio >= label_threshold else 0

        if activities is not None:
            # Most common activity in window
            acts = activities[start:end]
            window_activities[i] = np.bincount(acts.astype(int)).argmax()

    return windows, window_labels, window_activities


def create_windowed_dataset(
    dataset,
    window_seconds: float = 2.0,
    overlap: float = 0.5,
    sampling_rate: int = 60,
    label_threshold: float = 0.5,
) -> WindowedData:
    """
    Create windowed dataset from a FoGStarDataset or similar.

    Args:
        dataset: object with get_subject_imu, get_subject_labels, get_subject_activities, subject_ids
        window_seconds: window length in seconds
        overlap: fraction of overlap between windows
        sampling_rate: sampling rate in Hz
        label_threshold: threshold for majority voting

    Returns:
        WindowedData with all subjects' windows concatenated
    """
    window_size = int(window_seconds * sampling_rate)
    step_size = int(window_size * (1 - overlap))

    all_windows = []
    all_labels = []
    all_subjects = []
    all_activities = []

    for sid in dataset.subject_ids:
        imu = dataset.get_subject_imu(sid)
        labels = dataset.get_subject_labels(sid)

        try:
            activities = dataset.get_subject_activities(sid)
        except AttributeError:
            activities = None

        # Handle NaN values: interpolate small gaps, zero-fill remaining
        if np.any(np.isnan(imu)):
            for col in range(imu.shape[1]):
                mask = np.isnan(imu[:, col])
                if mask.any() and not mask.all():
                    valid = np.where(~mask)[0]
                    imu[mask, col] = np.interp(
                        np.where(mask)[0], valid, imu[valid, col]
                    )
                elif mask.all():
                    imu[:, col] = 0.0

        windows, win_labels, win_acts = extract_windows(
            imu, labels, window_size, step_size, activities, label_threshold
        )

        if len(windows) == 0:
            continue

        all_windows.append(windows)
        all_labels.append(win_labels)
        all_subjects.append(np.full(len(windows), sid, dtype=np.int64))
        if win_acts is not None:
            all_activities.append(win_acts)

    return WindowedData(
        windows=np.concatenate(all_windows),
        labels=np.concatenate(all_labels),
        subject_ids=np.concatenate(all_subjects),
        activities=np.concatenate(all_activities) if all_activities else np.zeros(0, dtype=np.int64),
    )


def lopo_split(windowed: WindowedData, test_subject: int) -> tuple[dict, dict]:
    """
    Leave-one-patient-out split.

    Returns:
        train: dict with keys 'windows', 'labels', 'subject_ids', 'activities'
        test: dict with same keys
    """
    test_mask = windowed.subject_ids == test_subject
    train_mask = ~test_mask

    def _extract(mask):
        d = {
            "windows": windowed.windows[mask],
            "labels": windowed.labels[mask],
            "subject_ids": windowed.subject_ids[mask],
        }
        if len(windowed.activities) > 0:
            d["activities"] = windowed.activities[mask]
        return d

    return _extract(train_mask), _extract(test_mask)
