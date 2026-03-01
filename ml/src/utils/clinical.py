"""
Clinical metadata processing and patient similarity computation.

Used for:
- Initializing Bayesian priors based on clinical profile similarity
- Clinical conditioning in neural models
- Patient embedding initialization
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist


class ClinicalProfiler:
    """Computes patient similarity based on clinical features."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.clinical_features: np.ndarray | None = None
        self.subject_ids: list[int] = []

    def fit(self, clinical_features: np.ndarray, subject_ids: list[int]) -> "ClinicalProfiler":
        """
        Fit the profiler on training patient clinical data.

        Args:
            clinical_features: (n_patients, n_features) clinical feature matrix
            subject_ids: list of subject IDs corresponding to rows
        """
        # Handle NaN values: impute with column median
        self.clinical_features = clinical_features.copy()
        for col in range(self.clinical_features.shape[1]):
            mask = np.isnan(self.clinical_features[:, col])
            if mask.any():
                median_val = np.nanmedian(self.clinical_features[:, col])
                self.clinical_features[mask, col] = median_val

        self.scaler.fit(self.clinical_features)
        self.clinical_features = self.scaler.transform(self.clinical_features)
        self.subject_ids = list(subject_ids)
        return self

    def get_similarity_weights(
        self,
        query_features: np.ndarray,
        bandwidth: float = 1.0,
    ) -> np.ndarray:
        """
        Compute RBF kernel similarity weights between a query patient
        and all training patients.

        Args:
            query_features: (n_features,) clinical features for query patient
            bandwidth: RBF kernel bandwidth

        Returns:
            weights: (n_training_patients,) normalized similarity weights
        """
        query = query_features.copy().reshape(1, -1)
        # Handle NaN in query
        for col in range(query.shape[1]):
            if np.isnan(query[0, col]):
                query[0, col] = 0.0  # after scaling, 0 = mean
        query_scaled = self.scaler.transform(query)

        distances = cdist(query_scaled, self.clinical_features, metric="euclidean")[0]
        weights = np.exp(-distances ** 2 / (2 * bandwidth ** 2))
        weights /= weights.sum() + 1e-12
        return weights

    def get_dirichlet_prior(
        self,
        query_features: np.ndarray,
        training_fog_ratios: dict[int, dict],
        n_specialists: int,
        bandwidth: float = 1.0,
        concentration: float = 10.0,
    ) -> np.ndarray:
        """
        Compute Dirichlet prior for Bayesian gating based on clinical similarity.

        The prior reflects which specialist models are likely most relevant
        for a patient with the given clinical profile, based on how similar
        training patients behaved.

        Args:
            query_features: (n_features,) clinical features
            training_fog_ratios: dict mapping subject_id -> {activity_context: fog_ratio}
            n_specialists: number of specialist models
            bandwidth: RBF kernel bandwidth
            concentration: Dirichlet concentration parameter

        Returns:
            alpha: (n_specialists,) Dirichlet concentration parameters
        """
        weights = self.get_similarity_weights(query_features, bandwidth)

        # Weighted average of training patients' specialist preferences
        alpha = np.ones(n_specialists) * (concentration / n_specialists)

        # Bias toward specialists where similar patients had more FoG
        for i, sid in enumerate(self.subject_ids):
            if sid in training_fog_ratios:
                patient_ratios = training_fog_ratios[sid]
                for spec_idx, (context, ratio) in enumerate(patient_ratios.items()):
                    if spec_idx < n_specialists:
                        alpha[spec_idx] += weights[i] * ratio * concentration

        return alpha

    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize clinical features using fitted scaler."""
        f = features.copy().reshape(1, -1)
        for col in range(f.shape[1]):
            if np.isnan(f[0, col]):
                f[0, col] = self.scaler.mean_[col]
        return self.scaler.transform(f)[0]
