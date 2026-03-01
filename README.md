# Parkinson's Disease Detection

Personalized detection system across two modalities — body sensors (FoG) and voice recordings — using three algorithms per dataset.

---

## Sensor Data (FoG-STAR — IMU / Accelerometers)

Freezing of Gait detection from 60 Hz wrist/ankle/back sensors, 16 patients, leave-one-patient-out evaluation.

| Figure | Description |
|--------|-------------|
| ![](results/figures/combined_acc_f1_roc_mean.png) | Mean ± 1 SD for Accuracy, F1, and AUC-ROC across all patients as adaptation data increases. |
| ![](results/figures/combined_all_metrics.png) | All six metrics (including sensitivity, specificity, episode detection rate) for all three approaches. |
| ![](results/figures/per_patient_f1_all.png) | Individual F1 curves per patient for every approach. |
| ![](results/figures/per_patient_bars_n0.png) | Per-patient bar comparison with zero patient-specific data. |
| ![](results/figures/per_patient_bars_n50.png) | Per-patient bar comparison after 50 patient-specific windows. |
| ![](results/figures/approach_c_cluster_outliers.png) | PCA showing normal-gait cluster vs FoG outliers detected by Approach C. |

---

## Voice Data (Parkinson Speech Dataset — Acoustic Features)

Parkinson's disease detection from 26 acoustic voice features, 40 patients, leave-one-patient-out evaluation.

| Figure | Description |
|--------|-------------|
| ![](results/figures/speech_combined_metrics_mean.png) | F1, sensitivity, specificity, and AUC across all patients as adaptation recordings increase. |
| ![](results/figures/speech_per_patient_f1.png) | Individual F1 curves per patient for every approach. |
| ![](results/figures/speech_per_patient_bars_n0.png) | Per-patient bar comparison with zero patient-specific recordings. |
| ![](results/figures/speech_per_patient_bars_n20.png) | Per-patient bar comparison after 20 patient-specific recordings. |
| ![](results/figures/speech_approach_c_cluster_outliers.png) | PCA showing healthy voice cluster vs Parkinson's outliers detected by Approach C. |

---

## Algorithms

### A — Bayesian Personalized Ensemble
- XGBoost specialists trained per activity context, combined via Dirichlet-Multinomial Bayesian gating.
- Clinical similarity to other patients initialises the prior; updates online with each new labelled sample.
- **Sensor: F1 0.396 → 0.412 | Voice: F1 0.723** — strong cold-start on voice, weaker FoG sensitivity.
- Details: [docs/approach_a.md](docs/approach_a.md) · Code: [src/models/base_ensemble.py](src/models/base_ensemble.py), [src/models/bayesian_gating.py](src/models/bayesian_gating.py)

### B — SSL Pre-training + Fine-tuning
- Sensor: Transformer masked autoencoder (T-MAE) pre-trained on raw IMU, then LoRA adapters personalise per patient. Voice: MLP fine-tuned per patient.
- No labels needed for pre-training; only a handful of labelled samples required to adapt.
- **Sensor: F1 0.400 → 0.443 | Voice: F1 0.771** — best cold-start on voice, moderate FoG improvement.
- Details: [docs/approach_b.md](docs/approach_b.md) · Code: [src/models/transformer_mae.py](src/models/transformer_mae.py), [src/models/lora_adapter.py](src/models/lora_adapter.py)

### C — Personalized Outlier Detection ✓ Selected
- Isolation Forest trained only on the patient's normal (non-FoG / healthy) data; anomalies flagged as disease.
- No disease labels ever required — only examples of normal behaviour, making real-world data collection trivial.
- **Sensor: F1 0.427 → 0.553, episode detection 81.6% | Voice: F1 0.535 → 0.873** — best personalisation gain across both modalities.
- Details: [docs/approach_c.md](docs/approach_c.md) · Why chosen: [docs/why_approach_c.md](docs/why_approach_c.md) · Code: [src/models/outlier_detector.py](src/models/outlier_detector.py)

---

## Run

```bash
pip install numpy pandas scipy scikit-learn xgboost torch matplotlib joblib

# Sensor (FoG-STAR)
python run_approach_c.py        # Approach C — recommended
python run_evaluation.py        # Approaches A + B
python make_figures.py

# Voice (Speech dataset)
python run_speech_approach_c.py
python run_speech_approach_a.py
python run_speech_approach_b.py
python make_speech_figures.py
```
