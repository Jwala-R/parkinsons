# Approach A — Bayesian Personalized Ensemble

## What is it?

Approach A is a **collection of specialist models** that each focus on a specific
activity a Parkinson's patient might be doing — walking, turning, transitioning
between tasks, dual-tasking (e.g. walking while talking), or any general movement.
Each specialist learns from a large pool of labelled data from *other* patients.

When a new patient arrives, the system combines the specialists' predictions using
**Bayesian gating**: it starts with a prior guess about which specialist to trust
most (based on how clinically similar this patient is to past patients), then
updates those trust weights every time a new labelled sample comes in.

---

## How it works — step by step

1. **Feature extraction**
   Raw 60 Hz accelerometer / gyroscope signals from ankle and back sensors are
   turned into ~339 handcrafted numbers per 2-second window: frequency-band power
   ratios, RMS acceleration, jerk, zero-crossing rate, cross-sensor correlations,
   and the clinically validated *Freeze Index* (power in 3–8 Hz divided by power
   in 0.5–3 Hz).

2. **Five specialist XGBoost classifiers**
   - `M_walk`       — trained only on walking windows
   - `M_turn`       — trained only on turning windows
   - `M_transition` — trained on activity transitions
   - `M_dualtask`   — trained on dual-task conditions
   - `M_general`    — trained on all windows combined

   Each specialist sees a weighted version of the training data where patients
   who are *clinically similar* to the test patient count for more.

3. **Bayesian gating (Dirichlet–Multinomial)**
   A probability distribution over the five specialists is maintained per patient.
   - *Prior*: initialised from clinical profile similarity (age, disease duration,
     UPDRS motor score, etc.) — so the model starts with a sensible guess about
     which context will dominate this patient's freezing.
   - *Posterior update*: each new labelled sample shifts the mixture weights
     toward whichever specialist predicted it best.

4. **Prediction**
   `P(FoG) = Σ_k  weight_k × specialist_k(features)`

---

## Strengths

- **Interpretable**: clinicians can see which activity context drives FoG for each patient.
- **No catastrophic forgetting**: base models are frozen; only the mixture weights adapt.
- **Clinical prior**: even with zero patient data it starts better than a blank slate.

## Weaknesses

- Relies entirely on handcrafted features — misses patterns the raw signal contains.
- Requires activity labels to route correctly; errors in activity labelling hurt performance.
- Low sensitivity (detects only ~34% of true FoG events at n=0).
