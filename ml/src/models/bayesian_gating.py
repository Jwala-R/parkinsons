"""
Bayesian gating network for personalized specialist ensemble.

Uses Dirichlet-Multinomial conjugate model to maintain per-patient
mixture weights over specialist models. Clinical priors initialize
the Dirichlet concentration, and online updates refine it.
"""

import numpy as np
from typing import Optional


class BayesianGating:
    """
    Per-patient Bayesian mixture weights over specialist models.

    The gating mechanism uses a Dirichlet prior (initialized from
    clinical similarity to training patients) and updates the posterior
    with each new labeled observation.
    """

    def __init__(self, n_specialists: int, prior_alpha: np.ndarray | None = None):
        """
        Args:
            n_specialists: number of specialist models
            prior_alpha: (n_specialists,) Dirichlet concentration params.
                         If None, uses uniform prior.
        """
        self.n_specialists = n_specialists
        if prior_alpha is not None:
            self.alpha = prior_alpha.copy().astype(np.float64)
        else:
            self.alpha = np.ones(n_specialists, dtype=np.float64)
        self.initial_alpha = self.alpha.copy()
        self.n_updates = 0

    @property
    def mixture_weights(self) -> np.ndarray:
        """Current expected mixture weights (Dirichlet mean)."""
        return self.alpha / self.alpha.sum()

    @property
    def uncertainty(self) -> float:
        """Entropy of the mixture weight distribution (lower = more certain)."""
        w = self.mixture_weights
        return float(-np.sum(w * np.log(w + 1e-12)))

    def predict(
        self,
        specialist_probs: np.ndarray,
        threshold: float = 0.5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Make weighted ensemble prediction.

        Args:
            specialist_probs: (n_samples, n_specialists) probabilities from each specialist
            threshold: decision threshold

        Returns:
            predictions: (n_samples,) binary predictions
            probabilities: (n_samples,) weighted FoG probability
        """
        weights = self.mixture_weights
        weighted_prob = specialist_probs @ weights  # (n_samples,)
        predictions = (weighted_prob >= threshold).astype(np.int64)
        return predictions, weighted_prob

    def update(
        self,
        specialist_probs: np.ndarray,
        true_labels: np.ndarray,
        learning_rate: float = 1.0,
    ):
        """
        Online Bayesian update of mixture weights.

        Updates the Dirichlet posterior based on which specialists were
        most accurate on the new labeled data. Specialists that correctly
        predicted the outcome get their concentration increased.

        Args:
            specialist_probs: (n_samples, n_specialists) specialist predictions
            true_labels: (n_samples,) ground truth binary labels
            learning_rate: scale factor for updates (1.0 = full Bayesian update)
        """
        for i in range(len(true_labels)):
            label = true_labels[i]
            probs = specialist_probs[i]  # (n_specialists,)

            # Likelihood of each specialist being "correct"
            if label == 1:
                likelihoods = probs  # higher prob = better prediction for FoG
            else:
                likelihoods = 1.0 - probs  # higher (1-prob) = better for non-FoG

            # Update Dirichlet concentration proportional to likelihood
            self.alpha += learning_rate * likelihoods
            self.n_updates += 1

    def reset(self):
        """Reset to initial prior."""
        self.alpha = self.initial_alpha.copy()
        self.n_updates = 0

    def get_specialist_ranking(self) -> list[tuple[str, float]]:
        """Get specialists ranked by weight."""
        weights = self.mixture_weights
        indices = np.argsort(-weights)
        return [(int(i), float(weights[i])) for i in indices]


class PersonalizedEnsemblePredictor:
    """
    Combines SpecialistEnsemble with BayesianGating for per-patient prediction.

    This is the full Approach A pipeline:
    1. Specialist ensemble produces per-specialist FoG probabilities
    2. Bayesian gating combines them using patient-specific weights
    3. Online updates refine weights as new labeled data arrives
    """

    def __init__(self, ensemble, gating: BayesianGating):
        self.ensemble = ensemble
        self.gating = gating
        self.prediction_history: list[dict] = []

    def predict(
        self,
        features: np.ndarray,
        threshold: float = 0.5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict FoG with personalized ensemble.

        Args:
            features: (n_samples, n_features)
            threshold: decision threshold

        Returns:
            predictions: (n_samples,) binary
            probabilities: (n_samples,) FoG probability
        """
        specialist_probs = self.ensemble.predict_proba_all(features)
        predictions, probabilities = self.gating.predict(specialist_probs, threshold)
        return predictions, probabilities

    def adapt(
        self,
        features: np.ndarray,
        true_labels: np.ndarray,
        learning_rate: float = 1.0,
    ):
        """
        Adapt gating weights using new labeled data.

        Args:
            features: (n_samples, n_features)
            true_labels: (n_samples,) ground truth
            learning_rate: adaptation rate
        """
        specialist_probs = self.ensemble.predict_proba_all(features)
        self.gating.update(specialist_probs, true_labels, learning_rate)

    def predict_and_adapt(
        self,
        features: np.ndarray,
        true_labels: np.ndarray,
        threshold: float = 0.5,
        learning_rate: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict first, then adapt (simulates online deployment)."""
        predictions, probabilities = self.predict(features, threshold)
        self.adapt(features, true_labels, learning_rate)

        self.prediction_history.append({
            "n_updates": self.gating.n_updates,
            "weights": self.gating.mixture_weights.copy(),
            "uncertainty": self.gating.uncertainty,
        })

        return predictions, probabilities
