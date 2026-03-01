"""
LOPO evaluation pipeline for personalized FoG detection.

Implements leave-one-patient-out cross-validation for both
Approach A (Bayesian Ensemble) and Approach B (SSL + LoRA).
"""

import numpy as np
import torch
from pathlib import Path
import json
import time

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.fog_star_loader import FoGStarDataset
from src.data.windowing import create_windowed_dataset, lopo_split, WindowedData
from src.data.features import extract_batch_features
from src.models.base_ensemble import SpecialistEnsemble
from src.models.bayesian_gating import BayesianGating, PersonalizedEnsemblePredictor
from src.utils.clinical import ClinicalProfiler
from src.utils.metrics import compute_all_metrics, event_level_metrics


def evaluate_approach_a_lopo(
    data_dir: str,
    window_seconds: float = 2.0,
    overlap: float = 0.5,
    n_adaptation_samples: list[int] = [0, 5, 10, 20, 50],
    save_dir: str = "results",
) -> dict:
    """
    Full LOPO evaluation of Approach A (Bayesian Personalized Ensemble).

    For each patient:
    1. Train specialist ensemble on remaining 21 patients
    2. Compute clinical prior for test patient
    3. Evaluate with and without online adaptation

    Returns:
        dict with per-patient and aggregate results
    """
    print("=" * 60)
    print("LOPO Evaluation: Approach A (Bayesian Personalized Ensemble)")
    print("=" * 60)

    # Load dataset
    dataset = FoGStarDataset(data_dir).load()
    print(dataset.summary())

    # Create windowed data
    print("\nExtracting windows...")
    windowed = create_windowed_dataset(dataset, window_seconds, overlap)
    print(f"Total windows: {len(windowed.labels):,} "
          f"(FoG: {windowed.labels.sum():,}, {windowed.labels.mean()*100:.1f}%)")

    # Clinical profiler
    clinical_features, clinical_ids = dataset.get_all_clinical_features()
    profiler = ClinicalProfiler()

    results = {}
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    for test_sid in dataset.subject_ids:
        print(f"\n--- Patient {test_sid} (left out) ---")
        start_time = time.time()

        # LOPO split
        train_data, test_data = lopo_split(windowed, test_sid)
        print(f"  Train: {len(train_data['labels']):,} windows, "
              f"Test: {len(test_data['labels']):,} windows")

        if len(test_data["labels"]) == 0 or test_data["labels"].sum() == 0:
            print(f"  Skipping: no test data or no FoG events")
            continue

        # Extract features
        print("  Extracting features...")
        train_features = extract_batch_features(train_data["windows"])
        test_features = extract_batch_features(test_data["windows"])

        # Handle NaN/Inf in features
        train_features = np.nan_to_num(train_features, nan=0.0, posinf=0.0, neginf=0.0)
        test_features = np.nan_to_num(test_features, nan=0.0, posinf=0.0, neginf=0.0)

        # Train specialist ensemble
        print("  Training specialists...")
        ensemble = SpecialistEnsemble()
        activities = train_data.get("activities")
        ensemble.fit(train_features, train_data["labels"], activities)

        # Fit clinical profiler on training patients
        train_sids = np.unique(train_data["subject_ids"])
        train_clinical_mask = np.isin(clinical_ids, train_sids)
        train_clinical = clinical_features[train_clinical_mask]
        train_clinical_ids = [cid for cid in clinical_ids if cid in train_sids]
        profiler.fit(train_clinical, train_clinical_ids)

        patient_results = {}

        for n_adapt in n_adaptation_samples:
            # Initialize Bayesian gating with clinical prior
            try:
                test_clinical = dataset.get_clinical_features(test_sid)
                prior_alpha = profiler.get_dirichlet_prior(
                    test_clinical, {}, ensemble.n_specialists, concentration=10.0
                )
            except (ValueError, KeyError):
                prior_alpha = None

            gating = BayesianGating(ensemble.n_specialists, prior_alpha)
            predictor = PersonalizedEnsemblePredictor(ensemble, gating)

            if n_adapt > 0 and n_adapt <= len(test_features):
                # Use first n_adapt samples for adaptation
                adapt_features = test_features[:n_adapt]
                adapt_labels = test_data["labels"][:n_adapt]
                predictor.adapt(adapt_features, adapt_labels)

                # Evaluate on remaining
                eval_features = test_features[n_adapt:]
                eval_labels = test_data["labels"][n_adapt:]
            else:
                eval_features = test_features
                eval_labels = test_data["labels"]

            if len(eval_labels) == 0:
                continue

            predictions, probabilities = predictor.predict(eval_features)
            metrics = compute_all_metrics(eval_labels, predictions, probabilities)
            event_metrics = event_level_metrics(eval_labels, predictions)

            patient_results[f"n_adapt={n_adapt}"] = {
                **metrics,
                **event_metrics,
                "weights": gating.mixture_weights.tolist(),
            }

            print(f"  n_adapt={n_adapt}: F1={metrics['f1']:.3f}, "
                  f"Sens={metrics['sensitivity']:.3f}, Spec={metrics['specificity']:.3f}")

        elapsed = time.time() - start_time
        results[test_sid] = {"metrics": patient_results, "time_s": elapsed}

    # Aggregate results
    print("\n" + "=" * 60)
    print("AGGREGATE RESULTS")
    print("=" * 60)

    for n_adapt_key in [f"n_adapt={n}" for n in n_adaptation_samples]:
        f1_scores = []
        sens_scores = []
        spec_scores = []
        for sid, res in results.items():
            if n_adapt_key in res["metrics"]:
                f1_scores.append(res["metrics"][n_adapt_key]["f1"])
                sens_scores.append(res["metrics"][n_adapt_key]["sensitivity"])
                spec_scores.append(res["metrics"][n_adapt_key]["specificity"])

        if f1_scores:
            print(f"\n{n_adapt_key}:")
            print(f"  F1:   {np.mean(f1_scores):.3f} +/- {np.std(f1_scores):.3f}")
            print(f"  Sens: {np.mean(sens_scores):.3f} +/- {np.std(sens_scores):.3f}")
            print(f"  Spec: {np.mean(spec_scores):.3f} +/- {np.std(spec_scores):.3f}")

    # Save results
    # Convert numpy types for JSON serialization
    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    results_serializable = json.loads(
        json.dumps(results, default=_convert)
    )
    with open(save_path / "approach_a_lopo_results.json", "w") as f:
        json.dump(results_serializable, f, indent=2)

    print(f"\nResults saved to {save_path / 'approach_a_lopo_results.json'}")
    return results


if __name__ == "__main__":
    import yaml

    config_path = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    project_root = Path(__file__).resolve().parents[2]
    data_dir = str(project_root / "datasets" / "fog")

    results = evaluate_approach_a_lopo(
        data_dir=data_dir,
        window_seconds=config["windowing"]["window_seconds"],
        overlap=config["windowing"]["overlap"],
    )
