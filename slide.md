# Presentation Slides — Parkinson's FoG Detection

---

## Slide A: Literature Review & Scientific Background

**What is Freezing of Gait (FoG)?**
- Paroxysmal inability to initiate or continue walking despite the intention to move
- Affects ~40% of PD patients overall; up to 92% at advanced disease stages [Cao et al., *CNS Journal*, 2020]
- Major fall risk factor — FoG episodes often occur at turns, doorways, and under cognitive load [Frontiers Neurology, 2023]

**Clinical Detection Gap**
- FoG is episodic and difficult to capture in clinic; patients under-report and clinicians underestimate [Silberstein & Bhatt, 2020]
- Gold standard: video annotation by neurologist — subjective, expensive, not continuous
- Wearable IMU sensors (ankle/back) + spectral Freeze Index (Moore et al.) opened the door to automated detection [Sensors, 2019]

**Why Personalization?**
- FoG presentation is highly patient-specific in severity, trigger context, and frequency [Frontiers Neurology, 2023]
- Meta-analysis of cueing: responses are "highly individual" — 22/31 subjects responded; no single modality worked for all [PMC, 2023]
- One-size-fits-all ML models trained on population data transfer poorly to unseen patients (domain shift, class imbalance)

**Sensor-Based ML — State of the Art**
- Deep learning + wearables: accuracy up to 92.3%, sensitivity 90% in supervised settings [Brain and Behavior, 2025]
- Nature Communications 2024 ML contest: best automated FoG detector leverages ensemble methods + temporal context
- Key limitation of prior work: all require labelled FoG windows per patient — clinically impractical

**Sources:** [Cao et al. 2020](https://cnjournal.biomedcentral.com/articles/10.1186/s41016-020-00197-y) · [Frontiers Triggers Review 2023](https://www.frontiersin.org/journals/neurology/articles/10.3389/fneur.2023.1326300/full) · [Wearable-Sensor Review 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6928783/) · [Nature Comms ML Contest 2024](https://www.nature.com/articles/s41467-024-49027-0) · [Tactile Cueing PMC 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10267272/)

---

## Slide B: Virtual Environment — Scientific Rationale

**Why a Virtual / Game-Based Environment?**
- VR + exergames outperform traditional physical therapy in gait and balance for PD (meta-analysis) [JMIR 2024]
- Goal-directed repetitive tasks promote neuroplasticity and motor-cognitive integration
- Gamification increases adherence — critical for PD where home exercise compliance is low [JMIR Serious Games 2025]

**Why Haptic Cueing (not visual or auditory)?**
- Rhythmic auditory cueing (RAC) improves stride length and reduces FoG episodes — meta-analysis confirms positive effect [Scientific Reports, 2018; Frontiers Neurology, 2022]
- Tactile (haptic) cueing = equivalent therapeutic effect to auditory cueing, with less cognitive distraction [*Good Vibrations*, PMC 2023]
- Visual cues (lines on floor) require attention; haptic is ambient — does not interrupt task engagement
- Our design: triple-pulse haptic burst on FoG detection, 2 s cooldown — mirrors RAC cadence without headphones

**Why Fine Motor Tasks (eating, whack-a-mole)?**
- Dual-task conditions (motor + cognitive) are the highest-risk FoG triggers [Frontiers Neurology, 2023]
- PD fine motor deficits (bradykinesia, tremor) are targetable with wrist-based grip and tilt tasks
- Serious games for fine motor PD rehabilitation show significant improvement in motor scores [PubMed, 2018]

**Closed-Loop Design Principle**
- Detect → Cue → Continue: detection does not interrupt therapy, preserving dual-task challenge
- Threshold tunable per patient — reflects evidence that cueing response is highly individual [PMC 2023]
- Home-deployable: wrist sensor + laptop, no clinic required; aligns with telerehabilitation trend [PMC VR Review 2025]

**Sources:** [JMIR Effectiveness Review 2024](https://games.jmir.org/2024/1/e53431) · [Good Vibrations PMC 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10267272/) · [RAC Meta-Analysis Sci Reports 2018](https://www.nature.com/articles/s41598-017-16232-5) · [Serious Games Fine Motor PubMed](https://pubmed.ncbi.nlm.nih.gov/29295055/) · [VR PD Bibliometric 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC11902477/)

---

## Slide 0: Datasets Used

**FoG-STAR (Sensor)**
- 22 Parkinson's patients, wearable IMU sensors (wrist, ankles, back)
- 24 channels (accelerometer + gyroscope), 60 Hz sampling
- 5,366 windows (2 s, 50% overlap) — 19.9% FoG-positive
- Clinical metadata per patient: H&Y stage, UPDRS-III, FoG-Q, MoCA
- Evaluation: Leave-One-Patient-Out (LOPO), 22 folds

**Parkinson Speech Dataset (Voice)**
- 40 subjects (20 PD, 20 healthy), UCI repository
- 26 acoustic features per recording: jitter, shimmer, HNR, pitch metrics
- 26 training + 6 test recordings per subject
- Labels: Parkinson's present / absent
- Evaluation: Leave-One-Patient-Out (LOPO), 40 folds

---

## Slide 1: Sensor FoG Detection — 3 ML Algorithms

**Approach A — Bayesian Personalized Ensemble**
- 5 specialist XGBoost models trained per activity context (walk, turn, transition, dual-task, general)
- 339 handcrafted time/frequency/cross-sensor features + Freeze Index biomarker
- Dirichlet Bayesian gating initialized from clinical profile similarity (RBF kernel)
- Online Bayesian update as patient-specific labelled windows accumulate

**Approach B — SSL Transformer + LoRA Personalization**
- Stage 1: Temporal Masked Autoencoder (T-MAE) pre-trains 4-layer Transformer on raw IMU (no labels)
- Stage 2: Per-patient LoRA adapters (rank=8, ~400 params) injected into Transformer layers
- Stage 3: Continual adaptation with focal loss, experience replay, and EWC regularization

**Approach C — Personalized Outlier Detection** *(selected)*
- Isolation Forest trained exclusively on normal-gait windows (no FoG labels ever required)
- Freeze Index (power ratio 3–8 Hz / 0.5–3 Hz) fused as a clinically validated second signal
- Cold-start: population model; as patient data grows, smoothly blends in patient-specific model

---

## Slide 2: Voice FoG Detection — 3 ML Algorithms

**Approach A — Bayesian Ensemble (Acoustic)**
- 5 specialist models trained on 26 acoustic features, gated by recording type
- Clinical prior from patient metadata (age, disease duration, UPDRS-III)
- Cold-start F1 = 0.723 using population-level priors only

**Approach B — MLP Fine-tuning (Acoustic)**
- Shared MLP backbone pre-trained on all patients' acoustic features
- Per-patient fine-tuning on a small held-in set (n recordings)
- Best cold-start performance: F1 = 0.771

**Approach C — Outlier Detection (Acoustic)** *(selected)*
- Isolation Forest trained only on healthy voice features (no PD labels needed)
- Combined with acoustic anomaly scoring over 26 features
- Strongest personalization gain: F1 = 0.535 → 0.873 from n=0 to n=20 recordings

---

## Slide 3: Sensor FoG — Results Graphs

**Figures to include:**

`results/figures/combined_all_metrics.png`
- All 6 metrics (sensitivity, specificity, F1, precision, event detection rate, AUC) vs. adaptation windows for all 3 approaches

`results/figures/per_patient_bars_n0.png` + `per_patient_bars_n50.png`
- Per-patient F1 bar charts at cold-start (n=0) and after 50 windows (n=50)

`results/figures/approach_c_cluster_outliers.png`
- PCA plot: normal-gait cluster vs FoG outliers detected by Approach C

**Key numbers to annotate:**
- Approach C event detection: **81.6%** vs A: 32.1%, B: 52.3% (cold-start)
- Approach C F1 after adaptation: **0.553** (n=50), sensitivity **0.801**
- Approach A highest specificity: **0.923** (n=0)

---

## Slide 4: Voice FoG — Results Graphs

**Figures to include:**

`results/figures/speech_combined_metrics_mean.png`
- F1, sensitivity, specificity, AUC curves across all 3 approaches as recordings accumulate

`results/figures/speech_per_patient_bars_n0.png` + `speech_per_patient_bars_n20.png`
- Per-patient comparison at n=0 and n=20 recordings

`results/figures/speech_approach_c_cluster_outliers.png`
- PCA: healthy voice cluster vs Parkinson's outliers detected by Approach C

**Key numbers to annotate:**
- Approach C: F1 **0.535 → 0.873** (n=0 to n=20 recordings) — largest gain of any approach
- Approach B: best cold-start F1 = **0.771**
- Approach A: stable across adaptation (F1 ≈ 0.72)

---

## Slide 5: Virtual Environment — FoG Detection & Sensor Tuning

**Real-Time Detection Pipeline**
- MPU-6050 IMU on wrist (60 Hz) → Arduino Uno → USB serial → ring buffer → Approach C outlier detector
- Freeze Index computed live on vertical accelerometer; score blended with Isolation Forest anomaly signal
- HUD displays live FoG risk bar so clinician can observe detection confidence in real time

**In-Environment Sensor Tuning**
- The game environment *is* the calibration tool — patient plays normally while the detector collects clean normal-gait windows
- Each game session passively accumulates patient-specific normal-gait data; Isolation Forest retrains in the background, sharpening the personal baseline
- Clinician watches the HUD risk bar during play and adjusts the detection threshold (`--threshold` flag) until false positives and missed episodes balance for that patient
- After ~50 windows of game play, the model transitions from population-level to fully personalised detection — same curve as the lab evaluation

**Tuning the FoG Response (Haptic)**
- Triple-pulse haptic burst fires on detection; 2 s cooldown prevents over-cueing
- If patient reports too many false alarms during a session → raise threshold live; if episodes are missed → lower it
- Cooldown duration also configurable: shorter for patients with frequent brief freezes, longer for patients who self-recover quickly
- All tuning decisions are logged per session so clinician can track drift over time

**Why the Game Context Matters for Tuning**
- Dual-task (game + walking) provokes realistic FoG — calibration reflects real-world conditions, not sterile walk tests
- Variety of arm movements (eating arc vs. whack-a-mole jabs) stress-tests the wrist IMU against non-FoG motion artefacts, exposing threshold weaknesses early

---

## Slide 6: Virtual Environment — Patient Training & Exercise

**Therapy Game 1: Eating Task**
- Guide a spoon from bowl to mouth for 5 cycles using wrist tilt (acc_x) and swing (acc_y)
- FSR pressure pad detects grip loss (spoon drop event)
- Trains fine motor control and hand-eye coordination

**Therapy Game 2: Whack-a-Mole**
- 3×3 grid of moles; aim mallet cursor with wrist tilt, squeeze FSR to whack
- Hit 10 moles to complete the session
- Trains rapid wrist movement, reaction time, and grip strength

**Session Design**
- FoG alerts are silent to patient — haptic-only to avoid anxiety or distraction
- Games run in Panda3D 3D environment, no specialized hardware beyond wrist sensor
- Session progress tracked; clinician can adjust difficulty and threshold between sessions
- Supports live Arduino sensor or fully offline demo mode for clinic demonstrations

---

## Slide 7: Results Discussion

**Sensor Modality**
- Approach C dominates on episode detection (81.6%) — the clinically critical metric for preventing falls
- Approach A highest specificity (0.923) — useful when false alarms have high cost
- Approach B best sensitivity trend with data — viable if labelled data collection is feasible
- All approaches improve meaningfully with only 10–20 patient-specific windows

**Voice Modality**
- Approach C shows the steepest personalization curve (F1 +0.34 over 20 recordings)
- Voice features alone achieve F1 > 0.77 at cold-start — strong signal even without sensor data
- Potential for remote monitoring via phone microphone (no wearable required)

**Cross-Modality Insight**
- Outlier detection paradigm (Approach C) generalizes across both sensor and acoustic domains
- "No FoG labels ever" constraint is clinically realistic — annotation is expensive and subjective
- Personalization gains are largest in the first 10–20 samples — fast adaptation curve

---

## Slide 8: Conclusion

- **Three algorithms, two modalities:** Bayesian Ensemble, SSL+LoRA Transformer, and Outlier Detection evaluated rigorously with leave-one-patient-out cross-validation across 22 sensor patients and 40 voice patients
- **Approach C selected:** Freeze Index + Isolation Forest achieves 81.6% FoG episode detection at cold-start without any FoG-labelled training data — the only approach deployable out-of-the-box
- **Personalization is fast:** 10–20 patient-specific samples drive the largest performance gains across all approaches and both modalities
- **Virtual therapy environment:** Closes the loop from detection to intervention — real-time haptic cueing during motor-rehabilitation games, runnable on standard hardware with or without sensor
- **Clinical takeaway:** A one-size-fits-all model is insufficient for FoG; patient-adaptive systems with interpretable biomarkers are both feasible and necessary
