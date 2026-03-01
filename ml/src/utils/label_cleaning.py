"""
Label cleaning utilities for FoG detection.

FoG annotations have known temporal ambiguity:
  - Short gaps between FoG bouts are often still FoG (incomplete annotation)
  - Single-window FoG labels may be spurious noise
  - Non-FoG windows near real FoG with high Freeze Index are likely mislabeled

These utilities clean training labels to reduce label noise without modifying
the test set (labels stay intact for evaluation).
"""

import numpy as np
from scipy.signal import welch


# ── Temporal bridging ────────────────────────────────────────────────────────

def bridge_fog_gaps(labels: np.ndarray, max_gap: int = 3) -> np.ndarray:
    """
    Fill short gaps (<=max_gap consecutive non-FoG windows) between FoG segments.

    Rationale: If FoG is labeled at t=10 and t=14 with a gap at t=11-13,
    the gap windows are likely still in FoG (annotation imprecision or
    brief recovery not captured at 2s window granularity).

    Args:
        labels: (n_windows,) binary array
        max_gap: maximum gap size (in windows) to bridge

    Returns:
        cleaned labels with gaps filled
    """
    labels = labels.copy()
    n = len(labels)
    i = 0
    while i < n:
        if labels[i] == 1:
            # Find end of this FoG segment
            j = i
            while j < n and labels[j] == 1:
                j += 1
            # j is now first non-FoG after FoG segment
            # Look for next FoG segment within max_gap windows
            gap_start = j
            k = j
            while k < n and labels[k] == 0 and (k - gap_start) <= max_gap:
                k += 1
            if k < n and labels[k] == 1 and (k - gap_start) <= max_gap:
                # Fill the gap
                labels[gap_start:k] = 1
            i = k
        else:
            i += 1
    return labels


def remove_isolated_fog(labels: np.ndarray, min_duration: int = 2) -> np.ndarray:
    """
    Remove FoG segments shorter than min_duration windows.

    A single-window FoG in the middle of normal walking is likely an
    annotation error or sensor artifact. Require at least min_duration
    consecutive FoG windows for a valid FoG event.

    Args:
        labels: (n_windows,) binary array
        min_duration: minimum consecutive FoG windows to keep

    Returns:
        cleaned labels with isolated FoG removed
    """
    labels = labels.copy()
    n = len(labels)
    i = 0
    while i < n:
        if labels[i] == 1:
            j = i
            while j < n and labels[j] == 1:
                j += 1
            # Segment [i, j) is FoG
            if (j - i) < min_duration:
                labels[i:j] = 0  # too short — remove
            i = j
        else:
            i += 1
    return labels


# ── Ambiguous window detection ───────────────────────────────────────────────

def compute_freeze_indices(windows: np.ndarray, fs: float = 60.0) -> np.ndarray:
    """
    Compute per-window Freeze Index using vertical ankle accelerometer channels.

    FI = power(3-8 Hz) / power(0.5-3 Hz)

    Uses channels 1 (left ankle ay), 7 (right ankle ay), 13 (back ay).
    Averages across the three vertical acceleration channels.

    Args:
        windows: (n_windows, window_size, n_channels)
        fs: sampling rate in Hz

    Returns:
        freeze_indices: (n_windows,) per-window FI values
    """
    n_windows = len(windows)
    fi_values = np.zeros(n_windows, dtype=np.float32)

    # Vertical channels: left ankle ay=1, right ankle ay=7, back ay=13
    vertical_channels = [1, 7, 13]
    # Clamp to available channels
    n_ch = windows.shape[2]
    vertical_channels = [c for c in vertical_channels if c < n_ch]

    nperseg = min(windows.shape[1], 256)
    if nperseg < 16:
        return fi_values

    for i in range(n_windows):
        fi_sum = 0.0
        n_valid = 0
        for ch in vertical_channels:
            sig = windows[i, :, ch]
            freqs, psd = welch(sig, fs=fs, nperseg=nperseg)

            loco_mask = (freqs >= 0.5) & (freqs <= 3.0)
            freeze_mask = (freqs >= 3.0) & (freqs <= 8.0)

            loco_p = np.trapz(psd[loco_mask], freqs[loco_mask]) if loco_mask.any() else 0.0
            freeze_p = np.trapz(psd[freeze_mask], freqs[freeze_mask]) if freeze_mask.any() else 0.0

            if loco_p > 0:
                fi_sum += freeze_p / loco_p
                n_valid += 1

        fi_values[i] = fi_sum / n_valid if n_valid > 0 else 0.0

    return fi_values


def flag_ambiguous_windows(
    labels: np.ndarray,
    freeze_indices: np.ndarray,
    fi_threshold: float = 2.0,
    context_radius: int = 5,
) -> np.ndarray:
    """
    Flag non-FoG windows that are likely mislabeled as negative.

    A non-FoG window is considered ambiguous (should be excluded from
    training as a negative example) if:
      1. It has high Freeze Index (biomechanically looks like FoG), AND
      2. It is within context_radius windows of a confirmed FoG window

    These windows are NOT relabeled to FoG (that would be too aggressive),
    but instead receive zero weight so they don't confuse the classifier.

    Args:
        labels: (n_windows,) binary array (cleaned labels after bridging/removal)
        freeze_indices: (n_windows,) FI values
        fi_threshold: FI above this value triggers ambiguous flag
        context_radius: distance (in windows) to nearest FoG to be flagged

    Returns:
        ambiguous_mask: (n_windows,) boolean, True = exclude from training
    """
    n = len(labels)
    ambiguous = np.zeros(n, dtype=bool)

    # Find FoG positions
    fog_positions = np.where(labels == 1)[0]
    if len(fog_positions) == 0:
        return ambiguous

    # For each non-FoG window with high FI, check proximity to FoG
    high_fi_non_fog = np.where((labels == 0) & (freeze_indices >= fi_threshold))[0]

    for idx in high_fi_non_fog:
        # Distance to nearest FoG window
        min_dist = np.min(np.abs(fog_positions - idx))
        if min_dist <= context_radius:
            ambiguous[idx] = True

    return ambiguous


# ── Full pipeline ─────────────────────────────────────────────────────────────

def clean_labels(
    labels: np.ndarray,
    windows: np.ndarray,
    fs: float = 60.0,
    bridge_gaps: bool = True,
    max_gap: int = 3,
    remove_isolated: bool = True,
    min_fog_duration: int = 2,
    flag_ambiguous: bool = True,
    fi_threshold: float = 2.5,
    context_radius: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Full label cleaning pipeline for training data.

    Steps:
      1. Bridge short gaps between FoG segments (fill likely FoG)
      2. Remove isolated (very short) FoG segments (remove artifacts)
      3. Flag ambiguous non-FoG windows (high FI near real FoG)

    Args:
        labels: (n_windows,) original binary labels
        windows: (n_windows, window_size, n_channels) raw IMU data
        fs: sampling rate
        bridge_gaps: whether to fill short inter-FoG gaps
        max_gap: maximum gap size to bridge (windows)
        remove_isolated: whether to remove short FoG artifacts
        min_fog_duration: minimum FoG segment length (windows)
        flag_ambiguous: whether to flag ambiguous negatives
        fi_threshold: FI above which a non-FoG window is suspicious
        context_radius: proximity to FoG to flag as ambiguous

    Returns:
        cleaned_labels: (n_windows,) cleaned binary labels
        ambiguous_mask: (n_windows,) boolean, True = exclude from training
        sample_weights: (n_windows,) float, 0 for excluded windows else 1
    """
    cleaned = labels.copy().astype(np.int32)

    # Track changes for diagnostics
    n_bridged = 0
    n_removed = 0

    if bridge_gaps:
        before = cleaned.sum()
        cleaned = bridge_fog_gaps(cleaned, max_gap=max_gap)
        n_bridged = int(cleaned.sum() - before)

    if remove_isolated:
        before = cleaned.sum()
        cleaned = remove_isolated_fog(cleaned, min_duration=min_fog_duration)
        n_removed = int(before - cleaned.sum())

    ambiguous_mask = np.zeros(len(labels), dtype=bool)
    if flag_ambiguous:
        # Compute FI only if needed (relatively slow)
        fi_vals = compute_freeze_indices(windows, fs=fs)
        ambiguous_mask = flag_ambiguous_windows(
            cleaned, fi_vals,
            fi_threshold=fi_threshold,
            context_radius=context_radius,
        )

    n_fog_after = int(cleaned.sum())
    n_ambiguous = int(ambiguous_mask.sum())

    print(f"    Label cleaning: {labels.sum()} FoG -> {n_fog_after} FoG "
          f"(+{n_bridged} bridged, -{n_removed} removed, "
          f"{n_ambiguous} ambiguous negatives excluded)")

    # Sample weights: 0 for ambiguous windows, 1 elsewhere
    sample_weights = (~ambiguous_mask).astype(np.float32)

    return cleaned.astype(np.int32), ambiguous_mask, sample_weights
