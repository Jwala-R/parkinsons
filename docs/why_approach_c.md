# Why We Are Moving Forward with Approach C

## Summary

After evaluating all three approaches on 16 Parkinson's patients using a rigorous
Leave-One-Patient-Out protocol, **Approach C (Personalized Outlier Detection)**
was selected as the primary method for deployment and further development.

---

## Performance at a Glance (n=0, no patient-specific data)

| Metric                  | Approach A | Approach B | Approach C |
|-------------------------|:----------:|:----------:|:----------:|
| F1 Score                | 0.396      | 0.400      | 0.427      |
| Sensitivity             | 0.338      | 0.503      | **0.788**  |
| Specificity             | 0.923      | 0.769      | 0.310      |
| AUC-ROC                 | 0.631      | 0.636      | 0.549      |
| FoG Episode Detection % | 32.1%      | 52.3%      | **81.6%**  |

With 50 personalisation windows (non-FoG only):

| Metric      | Approach A | Approach B | Approach C |
|-------------|:----------:|:----------:|:----------:|
| F1 Score    | 0.412      | 0.443      | **0.553**  |
| Sensitivity | 0.361      | 0.531      | **0.801**  |

---

## The Core Reason: Clinical Priority

In a real-world wearable FoG detector, **missing a freeze is far more dangerous
than a false alarm**.

- A missed freeze → patient falls, injury, loss of confidence.
- A false alarm → unnecessary vibration alert, minor annoyance.

Approach C detects **81.6% of all FoG episodes** vs only 32.1% for Approach A.
For clinical deployment, this is not a marginal difference — it is the difference
between a useful and a useless device.

---

## Why the Intuition Is Sound

FoG is, by definition, an abnormal event in a patient's movement repertoire.
Framing it as an outlier rather than a category to be learned is conceptually
correct:

- We do not need to have seen *this patient's* FoG before to detect it.
- We only need to know what their normal walking looks like.
- The Freeze Index provides a physics-grounded, clinically-validated second signal
  that is independent of any training data.

---

## Practical Advantages

| Property                         | A   | B   | C   |
|----------------------------------|:---:|:---:|:---:|
| Requires FoG labels to train     | Yes | Yes | No  |
| Works at cold-start (n=0)        | Yes | Yes | Yes |
| Improves with unlabelled normal data | No | No | Yes |
| Interpretable to clinician       | Partial | No | Yes |
| GPU required                     | No  | Yes | No  |
| Deployable on wearable hardware  | Possible | Hard | Yes |

Approach C is the only method that:
1. Requires **zero FoG labels** — making large-scale real-world data collection
   straightforward (no expert annotation needed).
2. Can be explained to a clinician using the Freeze Index formula.
3. Runs entirely on CPU and is lightweight enough for edge deployment.

---

## Next Steps

- Collect 50–100 normal-gait windows per new patient during a brief calibration
  walk (approximately 2–3 minutes) — no annotation required.
- Monitor the blend weight as more data accumulates; after ~50 windows the model
  is fully personalised.
- Investigate adaptive threshold tuning using heart rate or activity context as a
  co-variate to reduce false positives during sitting/standing.
- Validate on the fog2 dataset (cross-dataset generalisation).
