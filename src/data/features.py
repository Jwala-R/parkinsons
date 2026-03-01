"""
Handcrafted feature extraction for FoG detection.

Computes time-domain, frequency-domain, and cross-sensor features
from windowed IMU data. These features serve as input to the
Bayesian ensemble (Approach A) and as baselines for comparison.
"""

import numpy as np
from scipy.signal import welch
from scipy.stats import entropy as scipy_entropy


def _safe_divide(a: np.ndarray, b: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """Division with zero protection."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(b != 0, a / b, fill)
    return result


# ---------- Time-domain features ----------

def rms(signal: np.ndarray) -> float:
    """Root mean square."""
    return float(np.sqrt(np.mean(signal ** 2)))


def jerk(signal: np.ndarray, fs: float = 60.0) -> float:
    """Mean absolute jerk (derivative of signal)."""
    d = np.diff(signal, axis=0) * fs
    return float(np.mean(np.abs(d)))


def zero_crossing_rate(signal: np.ndarray) -> float:
    """Fraction of zero crossings."""
    crossings = np.sum(np.diff(np.sign(signal)) != 0)
    return float(crossings / max(len(signal) - 1, 1))


def signal_entropy(signal: np.ndarray, n_bins: int = 20) -> float:
    """Shannon entropy of signal amplitude distribution."""
    hist, _ = np.histogram(signal, bins=n_bins, density=True)
    hist = hist[hist > 0]
    return float(scipy_entropy(hist))


def signal_range(signal: np.ndarray) -> float:
    return float(np.ptp(signal))


def mean_abs(signal: np.ndarray) -> float:
    return float(np.mean(np.abs(signal)))


def std(signal: np.ndarray) -> float:
    return float(np.std(signal))


def skewness(signal: np.ndarray) -> float:
    m = np.mean(signal)
    s = np.std(signal)
    if s == 0:
        return 0.0
    return float(np.mean(((signal - m) / s) ** 3))


def kurtosis(signal: np.ndarray) -> float:
    m = np.mean(signal)
    s = np.std(signal)
    if s == 0:
        return 0.0
    return float(np.mean(((signal - m) / s) ** 4) - 3.0)


# ---------- Frequency-domain features ----------

def freeze_index(signal: np.ndarray, fs: float = 60.0,
                 loco_band: tuple = (0.5, 3.0),
                 freeze_band: tuple = (3.0, 8.0)) -> float:
    """
    Freeze Index: ratio of power in freeze band to locomotion band.
    This is a well-validated biomarker for FoG detection.
    """
    nperseg = min(len(signal), 256)
    if nperseg < 16:
        return 0.0
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)

    loco_mask = (freqs >= loco_band[0]) & (freqs <= loco_band[1])
    freeze_mask = (freqs >= freeze_band[0]) & (freqs <= freeze_band[1])

    loco_power = np.trapz(psd[loco_mask], freqs[loco_mask]) if loco_mask.any() else 0.0
    freeze_power = np.trapz(psd[freeze_mask], freqs[freeze_mask]) if freeze_mask.any() else 0.0

    if loco_power == 0:
        return 0.0
    return float(freeze_power / loco_power)


def band_power(signal: np.ndarray, fs: float, band: tuple) -> float:
    """Power in a specific frequency band."""
    nperseg = min(len(signal), 256)
    if nperseg < 16:
        return 0.0
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    mask = (freqs >= band[0]) & (freqs <= band[1])
    if not mask.any():
        return 0.0
    return float(np.trapz(psd[mask], freqs[mask]))


def spectral_entropy(signal: np.ndarray, fs: float = 60.0) -> float:
    """Entropy of the power spectral density (normalized)."""
    nperseg = min(len(signal), 256)
    if nperseg < 16:
        return 0.0
    _, psd = welch(signal, fs=fs, nperseg=nperseg)
    psd_norm = psd / (psd.sum() + 1e-12)
    psd_norm = psd_norm[psd_norm > 0]
    return float(scipy_entropy(psd_norm))


def dominant_frequency(signal: np.ndarray, fs: float = 60.0) -> float:
    """Frequency with maximum power."""
    nperseg = min(len(signal), 256)
    if nperseg < 16:
        return 0.0
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    return float(freqs[np.argmax(psd)])


# ---------- Cross-sensor features ----------

def cross_correlation(signal_a: np.ndarray, signal_b: np.ndarray) -> float:
    """Max normalized cross-correlation between two signals."""
    a = signal_a - np.mean(signal_a)
    b = signal_b - np.mean(signal_b)
    norm = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))
    if norm == 0:
        return 0.0
    corr = np.correlate(a, b, mode="full")
    return float(np.max(np.abs(corr)) / norm)


# ---------- Full feature extraction ----------

# Define sensor groups for 24-channel IMU
SENSOR_GROUPS = {
    "ankleL_acc": (0, 3),   # channels 0-2
    "ankleL_gyro": (3, 6),  # channels 3-5
    "ankleR_acc": (6, 9),
    "ankleR_gyro": (9, 12),
    "back_acc": (12, 15),
    "back_gyro": (15, 18),
    "wrist_acc": (18, 21),
    "wrist_gyro": (21, 24),
}

SPECTRAL_BANDS = [(0.5, 3.0), (3.0, 8.0), (8.0, 20.0)]


def extract_window_features(window: np.ndarray, fs: float = 60.0) -> np.ndarray:
    """
    Extract all features from a single window.

    Args:
        window: (window_size, n_channels) sensor data
        fs: sampling rate

    Returns:
        features: (n_features,) feature vector
    """
    features = []

    # Per-channel features
    for ch in range(window.shape[1]):
        sig = window[:, ch]

        # Time-domain (8 features per channel)
        features.extend([
            rms(sig),
            jerk(sig, fs),
            zero_crossing_rate(sig),
            signal_entropy(sig),
            signal_range(sig),
            mean_abs(sig),
            skewness(sig),
            kurtosis(sig),
        ])

        # Frequency-domain (6 features per channel)
        features.append(freeze_index(sig, fs))
        features.append(spectral_entropy(sig, fs))
        features.append(dominant_frequency(sig, fs))
        for band in SPECTRAL_BANDS:
            features.append(band_power(sig, fs, band))

    # Cross-sensor features: left ankle vs right ankle correlation
    # Acc magnitude correlation
    ankleL_mag = np.sqrt(np.sum(window[:, 0:3] ** 2, axis=1))
    ankleR_mag = np.sqrt(np.sum(window[:, 6:9] ** 2, axis=1))
    features.append(cross_correlation(ankleL_mag, ankleR_mag))

    # Gyro magnitude correlation
    ankleL_gyro_mag = np.sqrt(np.sum(window[:, 3:6] ** 2, axis=1))
    ankleR_gyro_mag = np.sqrt(np.sum(window[:, 9:12] ** 2, axis=1))
    features.append(cross_correlation(ankleL_gyro_mag, ankleR_gyro_mag))

    # Back vs wrist acc correlation
    back_mag = np.sqrt(np.sum(window[:, 12:15] ** 2, axis=1))
    wrist_mag = np.sqrt(np.sum(window[:, 18:21] ** 2, axis=1))
    features.append(cross_correlation(back_mag, wrist_mag))

    return np.array(features, dtype=np.float32)


def extract_batch_features(windows: np.ndarray, fs: float = 60.0) -> np.ndarray:
    """
    Extract features from a batch of windows (optimized).

    Args:
        windows: (n_windows, window_size, n_channels)
        fs: sampling rate

    Returns:
        features: (n_windows, n_features)
    """
    return _extract_batch_fast(windows, fs)


def _extract_batch_fast(windows: np.ndarray, fs: float = 60.0) -> np.ndarray:
    """Vectorized batch feature extraction — much faster than per-window loop."""
    n_windows, win_size, n_ch = windows.shape
    nperseg = min(win_size, 256)

    all_features = []

    for i in range(n_windows):
        w = windows[i]  # (win_size, n_ch)
        feats = []

        # Compute PSD for all channels at once
        freqs, psd_all = welch(w, fs=fs, nperseg=nperseg, axis=0)  # (n_freqs, n_ch)

        for ch in range(n_ch):
            sig = w[:, ch]
            psd = psd_all[:, ch]

            # Time-domain (8)
            feats.append(float(np.sqrt(np.mean(sig ** 2))))  # rms
            feats.append(float(np.mean(np.abs(np.diff(sig) * fs))))  # jerk
            feats.append(float(np.sum(np.diff(np.sign(sig)) != 0) / max(len(sig) - 1, 1)))  # zcr
            hist, _ = np.histogram(sig, bins=20, density=True)
            hist = hist[hist > 0]
            feats.append(float(scipy_entropy(hist)) if len(hist) > 0 else 0.0)  # entropy
            feats.append(float(np.ptp(sig)))  # range
            feats.append(float(np.mean(np.abs(sig))))  # mean_abs
            s = np.std(sig)
            m = np.mean(sig)
            if s > 0:
                feats.append(float(np.mean(((sig - m) / s) ** 3)))  # skewness
                feats.append(float(np.mean(((sig - m) / s) ** 4) - 3.0))  # kurtosis
            else:
                feats.extend([0.0, 0.0])

            # Frequency-domain using pre-computed PSD (6)
            loco_mask = (freqs >= 0.5) & (freqs <= 3.0)
            freeze_mask = (freqs >= 3.0) & (freqs <= 8.0)
            loco_p = np.trapz(psd[loco_mask], freqs[loco_mask]) if loco_mask.any() else 0.0
            freeze_p = np.trapz(psd[freeze_mask], freqs[freeze_mask]) if freeze_mask.any() else 0.0
            feats.append(float(freeze_p / loco_p) if loco_p > 0 else 0.0)  # freeze_index

            psd_norm = psd / (psd.sum() + 1e-12)
            psd_pos = psd_norm[psd_norm > 0]
            feats.append(float(scipy_entropy(psd_pos)) if len(psd_pos) > 0 else 0.0)  # spec_entropy
            feats.append(float(freqs[np.argmax(psd)]))  # dom_freq

            for band in SPECTRAL_BANDS:
                bmask = (freqs >= band[0]) & (freqs <= band[1])
                feats.append(float(np.trapz(psd[bmask], freqs[bmask])) if bmask.any() else 0.0)

        # Cross-sensor (3)
        ankleL_mag = np.sqrt(np.sum(w[:, 0:3] ** 2, axis=1))
        ankleR_mag = np.sqrt(np.sum(w[:, 6:9] ** 2, axis=1))
        feats.append(cross_correlation(ankleL_mag, ankleR_mag))

        ankleL_gyro_mag = np.sqrt(np.sum(w[:, 3:6] ** 2, axis=1))
        ankleR_gyro_mag = np.sqrt(np.sum(w[:, 9:12] ** 2, axis=1))
        feats.append(cross_correlation(ankleL_gyro_mag, ankleR_gyro_mag))

        back_mag = np.sqrt(np.sum(w[:, 12:15] ** 2, axis=1))
        wrist_mag = np.sqrt(np.sum(w[:, 18:21] ** 2, axis=1))
        feats.append(cross_correlation(back_mag, wrist_mag))

        all_features.append(np.array(feats, dtype=np.float32))

    return np.stack(all_features)


def get_feature_names(n_channels: int = 24) -> list[str]:
    """Get human-readable feature names."""
    names = []
    channel_names = [
        "aL_ax", "aL_ay", "aL_az", "aL_gx", "aL_gy", "aL_gz",
        "aR_ax", "aR_ay", "aR_az", "aR_gx", "aR_gy", "aR_gz",
        "b_ax", "b_ay", "b_az", "b_gx", "b_gy", "b_gz",
        "w_ax", "w_ay", "w_az", "w_gx", "w_gy", "w_gz",
    ]

    time_feats = ["rms", "jerk", "zcr", "entropy", "range", "mean_abs", "skew", "kurt"]
    freq_feats = ["freeze_idx", "spec_entropy", "dom_freq", "bp_0.5-3", "bp_3-8", "bp_8-20"]

    for ch_name in channel_names[:n_channels]:
        for feat in time_feats:
            names.append(f"{ch_name}_{feat}")
        for feat in freq_feats:
            names.append(f"{ch_name}_{feat}")

    names.extend(["xcorr_ankle_acc", "xcorr_ankle_gyro", "xcorr_back_wrist"])
    return names
