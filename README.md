# Personalized FoG Detection for Parkinson's Disease

Freezing of Gait (FoG) detection system that **adapts to each individual patient**
rather than using a one-size-fits-all model.

Three approaches were developed and compared under a Leave-One-Patient-Out protocol
on the FoG-STAR dataset (16 patients with FoG events, 60 Hz IMU).

**We moved forward with Approach C** — see [docs/why_approach_c.md](docs/why_approach_c.md).

---

## Approaches

| | Approach | F1 (n=0) | Sensitivity | FoG Episode Detection |
|---|---|:---:|:---:|:---:|
| A | [Bayesian Personalized Ensemble](docs/approach_a.md) | 0.396 | 0.338 | 32.1% |
| B | [SSL Pre-training + LoRA](docs/approach_b.md) | 0.400 | 0.503 | 52.3% |
| C | [Personalized Outlier Detection](docs/approach_c.md) | **0.427** | **0.788** | **81.6%** |

---

## Project Structure

```
parkinsons/
  src/
    data/
      fog_star_loader.py     # FoG-STAR dataset loader
      windowing.py           # 2s sliding windows, 50% overlap
      features.py            # 339 handcrafted IMU features
    models/
      base_ensemble.py       # Approach A: XGBoost specialists
      bayesian_gating.py     # Approach A: Dirichlet gating
      transformer_mae.py     # Approach B: T-MAE self-supervised encoder
      lora_adapter.py        # Approach B: LoRA personalisation adapters
      personalized_detector.py  # Approach B: full detector
      outlier_detector.py    # Approach C: Isolation Forest + Freeze Index
    training/
      pretrain.py            # T-MAE pre-training script
      finetune.py            # LoRA fine-tuning
      online_adapt.py        # Continual adaptation
      evaluate.py            # Shared evaluation utilities
    utils/
      clinical.py            # Clinical profile similarity
      label_cleaning.py      # FoG label noise handling
      metrics.py             # Window + event-level metrics
  docs/
    approach_a.md            # Approach A explanation
    approach_b.md            # Approach B explanation
    approach_c.md            # Approach C explanation
    why_approach_c.md        # Recommendation rationale
  results/
    figures/                 # Performance plots + cluster visualisation
    *.json                   # LOPO evaluation results per approach
    summary_metrics.csv      # Combined metrics table
  run_evaluation.py          # Run Approach A + B LOPO evaluation
  run_approach_c.py          # Run Approach C LOPO evaluation + export
  make_figures.py            # Generate all comparison figures
  configs/default.yaml       # Hyperparameter configuration
```

---

## Quick Start

### Requirements
```bash
pip install numpy pandas scipy scikit-learn xgboost torch matplotlib joblib
```

### Run Approach C (recommended)
```bash
python run_approach_c.py
```

### Generate all figures
```bash
python make_figures.py
```

### Run full evaluation (all approaches)
```bash
python run_evaluation.py  # Approaches A + B
python run_approach_c.py  # Approach C
```

---

## Dataset

The FoG-STAR dataset (`datasets/fog/`) is not included in this repository due to
size and data governance. Place the dataset at `datasets/fog/` with:
- `sensor_data.csv` — 60 Hz IMU data, 24 channels, all subjects
- `clinical_data.csv` — demographic and clinical features per subject

---

## Key Results

**Approach C at n=50 personalisation windows:**
- F1: 0.553
- Sensitivity: 0.801 (detects 4 in 5 FoG episodes)
- FoG Episode Detection Rate: 87.4%

See [results/figures/](results/figures/) for full visualisations including the
cluster/outlier plot showing how FoG windows separate from normal gait in PCA space.
