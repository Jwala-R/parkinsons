# Approach B — Self-Supervised Pre-training + LoRA Personalization

## What is it?

Approach B uses a **deep learning model** pre-trained to understand IMU (motion
sensor) signals without needing any labels, then fine-tunes tiny personalisation
adapters for each individual patient using only a handful of labelled examples.

The core idea is *transfer learning*: train a powerful general model on large
amounts of unlabelled data, then cheaply adapt it to each person.

---

## How it works — step by step

### Stage 1 — Self-Supervised Pre-training (T-MAE)

A **Temporal Masked Autoencoder (T-MAE)** is trained on all available sensor
windows — no FoG labels needed.

- The 2-second IMU window is split into short patches (4-sample chunks across
  24 sensor channels).
- 50% of patches are randomly masked (hidden).
- The model — a 4-layer Transformer (d=128, 4 attention heads) — learns to
  reconstruct the hidden patches from the visible ones.
- This forces the model to learn the *structure* of human movement: how different
  sensors relate to each other, what normal gait looks like, etc.

After pre-training the encoder weights capture rich motion representations.

### Stage 2 — Per-Patient Fine-Tuning with LoRA

The pre-trained encoder is **frozen** (its weights are not changed). Instead,
tiny **Low-Rank Adaptation (LoRA)** matrices are injected into the Transformer
layers — only ~400 extra numbers per patient.

- A classification head is added on top: encoder output → 64-unit hidden layer →
  FoG probability.
- Clinical metadata (age, disease duration, UPDRS score, etc.) is concatenated
  with the encoder features to condition predictions on patient profile.
- Only the LoRA adapters + classification head are trained on the patient's
  labelled data (focal loss to handle class imbalance).

### Stage 3 — Continual Online Adaptation

As more labelled windows arrive:
- Gradient updates continue on the LoRA + head.
- **Experience replay**: a buffer of past examples prevents forgetting older data.
- **EWC-lite regularisation**: penalises large deviations from the fine-tuned
  weights, reducing catastrophic forgetting.
- **Pseudo-labelling**: high-confidence predictions on unlabelled windows are
  used as additional training signal.

---

## Strengths

- Learns directly from raw sensor signals — no manual feature engineering.
- LoRA keeps adaptation cost tiny (~400 parameters vs millions in the full model).
- Can improve continuously as new labelled data arrives.

## Weaknesses

- Requires GPU pre-training (hours of compute up front).
- More complex to deploy and debug than classical ML.
- In our experiments, performance gains over Approach A were modest — deep models
  need more data to shine at this dataset size.
- Less interpretable: hard to explain *why* a specific prediction was made.
