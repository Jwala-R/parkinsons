"""
Parkinson Speech Dataset loader.

Dataset: UCI Parkinson's Speech Dataset with Multiple Types of Sound Recordings
Format:  subject_id, 26 acoustic features, recording_type, label
Label:   0 = healthy, 1 = Parkinson's disease
Subjects: 40 patients, 26 recordings each in train, 6 each in test
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path


# UCI Parkinson's Speech Dataset — known feature names (columns 1-26)
ACOUSTIC_FEATURE_NAMES = [
    "Jitter_pct",          # 1  Jitter (%)
    "Jitter_abs",          # 2  Jitter (Abs)
    "Jitter_RAP",          # 3  Jitter (RAP)
    "Jitter_PPQ5",         # 4  Jitter (PPQ5)
    "Jitter_DDP",          # 5  Jitter (DDP)
    "Shimmer_dB",          # 6  Shimmer (dB)
    "Shimmer_APQ3",        # 7  Shimmer (APQ3)
    "Shimmer_APQ5",        # 8  Shimmer (APQ5)
    "Shimmer_APQ11",       # 9  Shimmer (APQ11)
    "Shimmer_DDA",         # 10 Shimmer (DDA)
    "AC",                  # 11 Autocorrelation
    "NTH",                 # 12 Noise-to-Harmonics ratio
    "HTN",                 # 13 Harmonics-to-Noise ratio
    "Median_pitch",        # 14 Median pitch (Hz)
    "Mean_pitch",          # 15 Mean pitch (Hz)
    "SD_pitch",            # 16 Std deviation of pitch
    "Min_pitch",           # 17 Min pitch
    "Max_pitch",           # 18 Max pitch
    "Num_pulses",          # 19 Number of pulses
    "Num_periods",         # 20 Number of periods
    "Mean_period",         # 21 Mean period
    "SD_period",           # 22 Std deviation of period
    "Frac_unvoiced",       # 23 Fraction of unvoiced frames
    "Num_voice_breaks",    # 24 Number of voice breaks
    "Degree_voice_breaks", # 25 Degree of voice breaks
    "UPDRS",               # 26 UPDRS motor score proxy / additional feature
]


@dataclass
class AcousticData:
    """
    Container for Parkinson's Speech dataset.

    Attributes:
        features:        (n, 26) float32 — acoustic features
        labels:          (n,) int        — 0=healthy, 1=Parkinson's
        subject_ids:     (n,) int        — patient ID (1-40)
        recording_types: (n,) int        — recording type / session ID
        feature_names:   list of 26 feature name strings
        subject_ids_unique: sorted list of unique patient IDs
    """
    features:        np.ndarray
    labels:          np.ndarray
    subject_ids:     np.ndarray
    recording_types: np.ndarray
    feature_names:   list

    @property
    def subject_ids_unique(self):
        return sorted(np.unique(self.subject_ids).tolist())

    @property
    def n_subjects(self):
        return len(self.subject_ids_unique)

    def get_subject_features(self, sid):
        mask = self.subject_ids == sid
        return self.features[mask]

    def get_subject_labels(self, sid):
        mask = self.subject_ids == sid
        return self.labels[mask]


class AcousticDataset:
    """
    Load the Parkinson Speech Dataset.

    Usage:
        ds = AcousticDataset("datasets/parkinson+speech+dataset+.../Parkinson_Multiple_Sound_Recording").load()
    """

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def load(self, use_test: bool = True) -> AcousticData:
        """
        Load train (and optionally test) data.

        The test set contains only Parkinson's patients (label=1), so by default
        we include it to maximise the number of Parkinson's examples available
        for LOPO evaluation. Subject IDs in test are a subset of the 40 training
        subjects.

        Args:
            use_test: whether to include test_data.txt (default True)

        Returns:
            AcousticData with all samples concatenated
        """
        train_path = self.data_dir / "train_data.txt"
        test_path  = self.data_dir / "test_data.txt"

        # train_data.txt: 29 cols — subject_id, 26 feats, recording_type, label
        # test_data.txt:  28 cols — subject_id, 26 feats, recording_type (no label)
        df_train = pd.read_csv(train_path, header=None).dropna(how="all")
        print(f"  Loaded train: {len(df_train)} rows, {df_train.shape[1]} cols")

        train_sids = df_train.iloc[:, 0].values.astype(np.int32)
        train_feat = df_train.iloc[:, 1:27].values.astype(np.float32)
        train_rec  = df_train.iloc[:, 27].values.astype(np.int32)
        train_lbl  = df_train.iloc[:, 28].values.astype(np.int32)

        sid_list  = [train_sids]
        feat_list = [train_feat]
        rec_list  = [train_rec]
        lbl_list  = [train_lbl]

        if use_test and test_path.exists():
            df_test = pd.read_csv(test_path, header=None).dropna(how="all")
            print(f"  Loaded test:  {len(df_test)} rows, {df_test.shape[1]} cols")
            if df_test.shape[1] >= 28:
                t_sids = df_test.iloc[:, 0].values.astype(np.int32)
                t_feat = df_test.iloc[:, 1:27].values.astype(np.float32)
                t_rec  = df_test.iloc[:, 27].values.astype(np.int32)
                # test has no label column — all are Parkinson's (1)
                t_lbl  = (np.ones(len(df_test), dtype=np.int32)
                          if df_test.shape[1] < 29
                          else df_test.iloc[:, 28].values.astype(np.int32))
                sid_list.append(t_sids)
                feat_list.append(t_feat)
                rec_list.append(t_rec)
                lbl_list.append(t_lbl)

        subject_ids     = np.concatenate(sid_list)
        features        = np.vstack(feat_list).astype(np.float32)
        recording_types = np.concatenate(rec_list)
        labels          = np.concatenate(lbl_list)
        total           = len(labels)
        print(f"  Total:        {total} rows")

        # Replace inf / nan
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        n_pos = labels.sum()
        n_tot = len(labels)
        print(f"  Labels: {n_pos} Parkinson's ({n_pos/n_tot*100:.1f}%), "
              f"{n_tot-n_pos} healthy -- {len(np.unique(subject_ids))} subjects")

        return AcousticData(
            features=features,
            labels=labels,
            subject_ids=subject_ids,
            recording_types=recording_types,
            feature_names=ACOUSTIC_FEATURE_NAMES,
        )
