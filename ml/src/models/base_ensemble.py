"""
Specialist XGBoost ensemble for Approach A.

Trains activity-context-aware specialist models:
- M_walk: walking segments (activity=1)
- M_turn: turn segments (activity=6,7)
- M_transition: sit/stand transitions (activity=4,5)
- M_dualtask: dual-task walking (tasks 5,6: walk+water, walk+count)
- M_general: all data (fallback)
"""

import numpy as np
import xgboost as xgb
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class SpecialistConfig:
    name: str
    activity_filter: list[int] | None = None  # filter by activity column
    task_filter: list[int] | None = None       # filter by task column


DEFAULT_SPECIALISTS = [
    SpecialistConfig("walk", activity_filter=[1]),
    SpecialistConfig("turn", activity_filter=[6, 7]),
    SpecialistConfig("transition", activity_filter=[4, 5]),
    SpecialistConfig("dualtask", task_filter=[5, 6]),
    SpecialistConfig("general"),  # no filter, uses all data
]


class SpecialistEnsemble:
    """
    Collection of activity-context specialist XGBoost classifiers.

    Each specialist is trained on a subset of data filtered by
    activity type, making it an expert for that movement context.
    """

    def __init__(
        self,
        specialists: list[SpecialistConfig] | None = None,
        xgb_params: dict | None = None,
    ):
        self.specialists = specialists or DEFAULT_SPECIALISTS
        self.xgb_params = xgb_params or {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "scale_pos_weight": 4.0,
            "eval_metric": "logloss",
            "use_label_encoder": False,
            "tree_method": "hist",
            "random_state": 42,
        }
        self.models: dict[str, xgb.XGBClassifier] = {}

    @property
    def n_specialists(self) -> int:
        return len(self.specialists)

    @property
    def specialist_names(self) -> list[str]:
        return [s.name for s in self.specialists]

    def _filter_data(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        activities: np.ndarray | None,
        task_ids: np.ndarray | None,
        spec: SpecialistConfig,
        sample_weights: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Filter data for a specialist's training context."""
        mask = np.ones(len(features), dtype=bool)

        if spec.activity_filter is not None and activities is not None:
            mask &= np.isin(activities, spec.activity_filter)

        if spec.task_filter is not None and task_ids is not None:
            mask &= np.isin(task_ids, spec.task_filter)

        # If filter is too restrictive (< 20 samples), fall back to all data
        if mask.sum() < 20:
            mask = np.ones(len(features), dtype=bool)

        sw = sample_weights[mask] if sample_weights is not None else None
        return features[mask], labels[mask], sw

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        activities: np.ndarray | None = None,
        task_ids: np.ndarray | None = None,
        sample_weights: np.ndarray | None = None,
    ) -> "SpecialistEnsemble":
        """
        Train all specialist models.

        Args:
            features: (n_samples, n_features) feature matrix
            labels: (n_samples,) binary labels
            activities: (n_samples,) activity labels (optional)
            task_ids: (n_samples,) task IDs (optional)
            sample_weights: (n_samples,) per-sample weights (optional)
        """
        for spec in self.specialists:
            X, y, sw = self._filter_data(features, labels, activities, task_ids,
                                         spec, sample_weights)

            model = xgb.XGBClassifier(**self.xgb_params)
            model.fit(X, y, sample_weight=sw)
            self.models[spec.name] = model

            n_pos = y.sum()
            print(f"  Specialist '{spec.name}': trained on {len(y)} samples "
                  f"({n_pos} FoG, {n_pos/len(y)*100:.1f}%)")

        return self

    def predict_proba_all(self, features: np.ndarray) -> np.ndarray:
        """
        Get FoG probability from each specialist.

        Args:
            features: (n_samples, n_features)

        Returns:
            probs: (n_samples, n_specialists) probability of FoG from each specialist
        """
        probs = np.zeros((len(features), self.n_specialists), dtype=np.float32)
        for i, spec in enumerate(self.specialists):
            model = self.models[spec.name]
            probs[:, i] = model.predict_proba(features)[:, 1]
        return probs

    def predict_uniform(self, features: np.ndarray, threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict with uniform mixture weights (non-personalized baseline).

        Returns:
            predictions: (n_samples,) binary predictions
            probabilities: (n_samples,) averaged FoG probability
        """
        probs = self.predict_proba_all(features)
        avg_prob = probs.mean(axis=1)
        return (avg_prob >= threshold).astype(np.int64), avg_prob
