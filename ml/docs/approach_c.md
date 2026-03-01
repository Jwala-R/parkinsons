# Approach C — Personalized Outlier Detection

## What is it?

Approach C reframes FoG detection as an **anomaly detection problem**.

The key insight: *FoG is not what a patient normally does.* Normal walking has a
characteristic rhythm captured in the sensor signals. FoG — whether it appears as
shuffling, trembling, or akinesia — breaks that rhythm in a measurable way. So
instead of asking "does this window look like FoG I've seen before?", Approach C
asks **"does this window look unlike this patient's normal gait?"**

This means we never need a single FoG-labelled training example. We only need
examples of the patient walking normally.

---

## How it works — step by step

### Signal 1 — Isolation Forest (learned anomaly model)

An **Isolation Forest** is trained *exclusively on normal-gait (non-FoG) windows*.

- Isolation Forest works by randomly partitioning the feature space into trees.
- Normal windows cluster together and require many splits to isolate.
- Anomalous windows (FoG) are "isolated" in far fewer splits — giving them a
  higher anomaly score.

With zero patient data the forest is trained on all *other* patients' normal gait
(population model). As the patient provides their own normal-walking windows,
a second forest personalised to that patient is blended in — more patient data
means more weight on the personal model.

### Signal 2 — Freeze Index (biomechanical FoG biomarker)

The **Freeze Index (FI)** is a clinically validated formula:

```
FI = Power(3–8 Hz) / Power(0.5–3 Hz)
```

Computed on the vertical accelerometer at both ankles and the lower back.

- Normal walking: most power is in the locomotion band (0.5–3 Hz). FI is low.
- FoG: the patient trembles rapidly (3–8 Hz) without moving forward. FI spikes.

FI provides a physics-based, interpretable signal that requires no training at all.

### Combining the two signals

The final anomaly score is a weighted average:

```
score = 0.5 × IsolationForest_score + 0.5 × FreezeIndex_score
```

Both signals are normalised to [0, 1] before combining. A score near 1 means the
window is highly anomalous relative to the patient's normal gait → predicted FoG.

### Personalization via blending

```
blend_weight = min(1.0, n_patient_windows / 50)

final_score = (1 - blend_weight) × population_score
            +      blend_weight  × patient_score
```

- **n=0**: purely population model (no patient data needed).
- **n=50**: fully personalised to the patient's own normal gait.

---

## Strengths

- **No FoG labels ever required** — only examples of the patient walking normally.
- Highly **interpretable**: the Freeze Index has a direct biomechanical meaning.
- Works immediately at n=0 (cold-start) using the population normal-gait model.
- Improves gracefully as more patient data is collected.
- Lightest computational footprint of the three approaches.
- Highest sensitivity (78.8%) and FoG episode detection rate (81.6%) at n=0.

## Weaknesses

- Threshold selection still benefits from a small validation set.
- May be confused by other unusual movements (falls, sudden stops) that are not FoG.
- Specificity can be lower than Approach A (more false positives in some patients).
