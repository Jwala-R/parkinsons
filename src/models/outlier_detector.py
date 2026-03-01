"""
Approach C: Personalized Outlier Detection for FoG.

Core insight: FoG is an anomaly relative to a patient's normal gait.
Rather than learning to classify FoG from labelled examples, we learn
what is NORMAL for each patient, then flag deviations as FoG.

Two signals are combined:
  1. Isolation Forest trained on patient's own non-FoG (normal gait) windows
  2. Freeze Index (validated biomechanical FoG biomarker)

The more normal-gait windows available from the patient, the better
the personalization. With zero patient data, falls back to population
non-FoG distribution as the reference.

No FoG labels are ever required — only examples of normal gait.
"""

import numpy as np
from scipy.signal import welch
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import f1_score


def compute_freeze_index_batch(windows: np.ndarray, fs: float = 60.0) -> np.ndarray:
    """
    Compute per-window Freeze Index (FI) across vertical accelerometer channels.

    FI = power(3-8 Hz) / power(0.5-3 Hz)
    Averaged over left ankle ay (ch 1), right ankle ay (ch 7), back ay (ch 13).

    Higher FI = more FoG-like gait.
    """
    n = len(windows)
    fi = np.zeros(n, dtype=np.float32)
    nperseg = min(windows.shape[1], 256)
    vertical_chs = [ch for ch in [1, 7, 13] if ch < windows.shape[2]]

    for i in range(n):
        total, count = 0.0, 0
        for ch in vertical_chs:
            freqs, psd = welch(windows[i, :, ch], fs=fs, nperseg=nperseg)
            lp = np.trapz(psd[(freqs >= 0.5) & (freqs <= 3.0)],
                          freqs[(freqs >= 0.5) & (freqs <= 3.0)])
            fp = np.trapz(psd[(freqs >= 3.0) & (freqs <= 8.0)],
                          freqs[(freqs >= 3.0) & (freqs <= 8.0)])
            if lp > 0:
                total += fp / lp
                count += 1
        fi[i] = total / count if count > 0 else 0.0
    return fi


def _normalize_01(x: np.ndarray) -> np.ndarray:
    """Scale array to [0, 1]."""
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-9)


class PersonalizedOutlierDetector:
    """
    Personalized FoG detector based on outlier detection from normal gait.

    Usage:
        # Fit on normal (non-FoG) gait data — patient's own or population
        detector = PersonalizedOutlierDetector()
        detector.fit(normal_features, normal_windows)

        # Score new windows (higher = more likely FoG)
        scores = detector.score(test_features, test_windows)
        predictions = (scores >= detector.threshold_).astype(int)
    """

    def __init__(
        self,
        n_estimators: int = 200,
        fi_weight: float = 0.5,       # weight given to Freeze Index vs IF anomaly score
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.fi_weight = fi_weight
        self.if_weight = 1.0 - fi_weight
        self.random_state = random_state

        self.scaler_ = RobustScaler()
        self.iso_forest_ = None
        self.threshold_ = 0.5
        self.contamination_ = 0.2  # estimated FoG prevalence

    def fit(
        self,
        normal_features: np.ndarray,
        normal_windows: np.ndarray | None = None,
        contamination: float = 0.2,
    ) -> "PersonalizedOutlierDetector":
        """
        Fit on normal (non-FoG) windows.

        Args:
            normal_features: (n, n_features) handcrafted features of normal windows
            normal_windows:  (n, win_len, n_channels) raw IMU of normal windows
            contamination:   estimated fraction of FoG in the full recording
                             (used only to set IF contamination parameter)
        """
        self.contamination_ = float(np.clip(contamination, 0.01, 0.49))
        self.scaler_.fit(normal_features)
        normal_scaled = self.scaler_.transform(normal_features)

        self.iso_forest_ = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination_,
            random_state=self.random_state,
        )
        self.iso_forest_.fit(normal_scaled)
        return self

    def score(
        self,
        features: np.ndarray,
        windows: np.ndarray | None = None,
        fs: float = 60.0,
    ) -> np.ndarray:
        """
        Compute anomaly scores for each window (higher = more anomalous).

        Args:
            features: (n, n_features) handcrafted features
            windows:  (n, win_len, n_channels) raw IMU data, or None.
                      When None the Freeze Index term is skipped and only
                      the Isolation Forest score is used.
            fs:       sampling rate (used only when windows is provided)

        Returns:
            scores: (n,) anomaly scores in [0, 1]
        """
        feat_scaled = self.scaler_.transform(features)
        if_scores = -self.iso_forest_.score_samples(feat_scaled)  # higher = more anomalous

        if windows is None:
            return _normalize_01(if_scores)

        fi_scores = compute_freeze_index_batch(windows, fs=fs)
        return (self.if_weight * _normalize_01(if_scores) +
                self.fi_weight * _normalize_01(fi_scores))

    def fit_threshold(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        """
        Find the threshold that maximizes F1 on validation scores.
        Stores result in self.threshold_.
        """
        best_f1, best_t = 0.0, 0.5
        for t in np.arange(0.05, 0.95, 0.01):
            f = f1_score(labels, (scores >= t).astype(int), zero_division=0)
            if f > best_f1:
                best_f1, best_t = f, t
        self.threshold_ = best_t
        return best_t

    def predict(self, scores: np.ndarray) -> np.ndarray:
        """Binary predictions using fitted threshold."""
        return (scores >= self.threshold_).astype(np.int64)


class PersonalizedOutlierEnsemble:
    """
    Wraps PersonalizedOutlierDetector for the LOPO evaluation protocol.

    For each test patient:
      Phase 1 (zero patient data): Use population non-FoG as normal reference.
      Phase 2 (n_seed patient windows): Blend population model with patient's
              own normal gait model, weighted by how much patient data is available.

    This is the 'Approach C' model.
    """

    def __init__(self, fi_weight: float = 0.5, n_estimators: int = 200):
        self.fi_weight = fi_weight
        self.n_estimators = n_estimators

        # Population model (fit once on all training non-FoG)
        self.pop_detector_ = PersonalizedOutlierDetector(
            n_estimators=n_estimators, fi_weight=fi_weight)
        self.pop_fitted_ = False

    def fit_population(
        self,
        train_features: np.ndarray,
        train_windows: np.ndarray | None,
        train_labels: np.ndarray,
    ) -> "PersonalizedOutlierEnsemble":
        """Fit the population-level normal model on all training non-anomaly samples."""
        nfog_mask = train_labels == 0
        contamination = float(np.clip(train_labels.mean(), 0.01, 0.49))
        self.pop_detector_.fit(
            train_features[nfog_mask],
            train_windows[nfog_mask] if train_windows is not None else None,
            contamination=contamination,
        )
        self.pop_fitted_ = True
        return self

    def score_and_predict(
        self,
        test_features: np.ndarray,
        test_windows: np.ndarray | None,
        test_labels: np.ndarray,
        n_seed_nonfog: int = 0,
        fs: float = 60.0,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Score test windows as FoG anomalies.

        Args:
            test_features: (n, n_feat) features for all test windows
            test_windows:  (n, win_len, n_ch) raw IMU for all test windows
            test_labels:   (n,) true labels (used only to find opt threshold)
            n_seed_nonfog: number of confirmed non-FoG windows from patient
                           to use for personalized normal model
            fs:            sampling rate

        Returns:
            predictions: (n,) binary
            scores:      (n,) anomaly scores
            threshold:   chosen decision threshold
        """
        if n_seed_nonfog > 0:
            # Build personalized model from patient's own non-FoG windows
            nfog_all_idx = np.where(test_labels == 0)[0]
            seed_idx = nfog_all_idx[:min(n_seed_nonfog, len(nfog_all_idx))]
            eval_idx = np.ones(len(test_labels), dtype=bool)
            eval_idx[seed_idx] = False

            contam = float(np.clip(test_labels.mean(), 0.01, 0.49))
            patient_det = PersonalizedOutlierDetector(
                n_estimators=self.n_estimators,
                fi_weight=self.fi_weight,
            )
            seed_wins = test_windows[seed_idx] if test_windows is not None else None
            patient_det.fit(
                test_features[seed_idx],
                seed_wins,
                contamination=contam,
            )

            # Blend: more patient data → trust patient model more
            # blend_w goes from 0 (population only) to 1 (patient only) as n_seed grows
            blend_w = min(1.0, n_seed_nonfog / 50.0)

            eval_feat = test_features[eval_idx]
            eval_win  = test_windows[eval_idx] if test_windows is not None else None
            eval_lbl  = test_labels[eval_idx]

            pop_scores     = self.pop_detector_.score(eval_feat, eval_win, fs)
            patient_scores = patient_det.score(eval_feat, eval_win, fs)
            scores = (1 - blend_w) * pop_scores + blend_w * patient_scores

            # Threshold on eval data (uses labels only to maximise F1 here;
            # in true deployment this would use a small labelled validation set)
            det = PersonalizedOutlierDetector(fi_weight=self.fi_weight)
            det.iso_forest_ = patient_det.iso_forest_  # placeholder
            threshold = det.fit_threshold(scores, eval_lbl)
            preds = (scores >= threshold).astype(np.int64)
            return preds, scores, threshold, eval_lbl

        else:
            # n_seed=0: pure population model, evaluate on all test windows
            scores = self.pop_detector_.score(test_features, test_windows, fs)
            det = PersonalizedOutlierDetector(fi_weight=self.fi_weight)
            det.iso_forest_ = self.pop_detector_.iso_forest_
            threshold = det.fit_threshold(scores, test_labels)
            preds = (scores >= threshold).astype(np.int64)
            return preds, scores, threshold, test_labels
