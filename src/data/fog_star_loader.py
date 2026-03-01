"""
FoG-STAR dataset loader.

Loads the 22-patient IMU dataset with clinical metadata.
sensor_data.csv: 323,830 rows x 31 columns @ 60Hz
clinical_data.csv: 22 patients with demographic/clinical features
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

IMU_COLUMNS = [
    "ankleL_acc_x", "ankleL_acc_y", "ankleL_acc_z",
    "ankleL_gyro_x", "ankleL_gyro_y", "ankleL_gyro_z",
    "ankleR_acc_x", "ankleR_acc_y", "ankleR_acc_z",
    "ankleR_gyro_x", "ankleR_gyro_y", "ankleR_gyro_z",
    "back_acc_x", "back_acc_y", "back_acc_z",
    "back_gyro_x", "back_gyro_y", "back_gyro_z",
    "wrist_acc_x", "wrist_acc_y", "wrist_acc_z",
    "wrist_gyro_x", "wrist_gyro_y", "wrist_gyro_z",
]

LABEL_COLUMNS = ["activity", "fog", "fog_severity", "subjectID", "sessionID", "taskID"]

CLINICAL_FEATURES = [
    "age", "gender", "disease_duration", "h_y", "updrs_iii",
    "fog_q", "moca", "fes-i", "pdq8",
]

ACTIVITY_MAP = {
    1: "Walk", 2: "Sit", 3: "Stand", 4: "Sit-to-Stand",
    5: "Stand-to-Sit", 6: "Turn-Right", 7: "Turn-Left",
}

SAMPLING_RATE = 60  # Hz


class FoGStarDataset:
    """Loads and provides access to FoG-STAR sensor and clinical data."""

    def __init__(self, data_dir: str, filter_inactive: bool = True):
        self.data_dir = Path(data_dir)
        self.sensor_path = self.data_dir / "sensor_data.csv"
        self.clinical_path = self.data_dir / "clinical_data.csv"

        self.sensor_data: Optional[pd.DataFrame] = None
        self.clinical_data: Optional[pd.DataFrame] = None

    def load(self, filter_inactive: bool = True) -> "FoGStarDataset":
        """Load sensor and clinical data from CSV files."""
        self.sensor_data = pd.read_csv(self.sensor_path)
        self.clinical_data = pd.read_csv(self.clinical_path)

        # Filter out rows with activity=0 (no labeled activity)
        if filter_inactive:
            self.sensor_data = self.sensor_data[self.sensor_data["activity"] > 0].reset_index(drop=True)

        # Drop rows where all IMU channels are NaN (sensor dropout)
        imu_mask = self.sensor_data[IMU_COLUMNS].notna().any(axis=1)
        n_dropped = (~imu_mask).sum()
        if n_dropped > 0:
            self.sensor_data = self.sensor_data[imu_mask].reset_index(drop=True)
            print(f"Dropped {n_dropped} rows with all-NaN IMU channels")

        # Forward-fill small NaN gaps in IMU data (sensor glitches)
        self.sensor_data[IMU_COLUMNS] = self.sensor_data[IMU_COLUMNS].ffill(limit=5)

        # Encode gender as numeric in clinical data
        if "gender" in self.clinical_data.columns:
            self.clinical_data["gender_numeric"] = (
                self.clinical_data["gender"].map({"M": 0, "F": 1})
            )

        return self

    @property
    def subject_ids(self) -> list[int]:
        return sorted(self.sensor_data["subjectID"].unique().tolist())

    @property
    def n_subjects(self) -> int:
        return len(self.subject_ids)

    def get_subject_data(self, subject_id: int) -> pd.DataFrame:
        """Get all sensor data for a single subject."""
        return self.sensor_data[self.sensor_data["subjectID"] == subject_id].reset_index(drop=True)

    def get_subject_imu(self, subject_id: int) -> np.ndarray:
        """Get IMU data as numpy array (n_samples, 24) for a subject."""
        df = self.get_subject_data(subject_id)
        return df[IMU_COLUMNS].values.astype(np.float32)

    def get_subject_labels(self, subject_id: int) -> np.ndarray:
        """Get FoG labels (0/1) for a subject."""
        df = self.get_subject_data(subject_id)
        return df["fog"].values.astype(np.int64)

    def get_subject_activities(self, subject_id: int) -> np.ndarray:
        """Get activity labels for a subject."""
        df = self.get_subject_data(subject_id)
        return df["activity"].values.astype(np.int64)

    def get_clinical_features(self, subject_id: int) -> np.ndarray:
        """Get clinical feature vector for a subject (9 features)."""
        row = self.clinical_data[self.clinical_data["subjectID"] == subject_id]
        if row.empty:
            raise ValueError(f"No clinical data for subject {subject_id}")

        features = ["age", "disease_duration", "h_y", "updrs_iii",
                     "fog_q", "moca", "fes-i", "pdq8", "gender_numeric"]
        vals = row[features].values[0].astype(np.float32)
        return vals

    def get_all_clinical_features(self) -> tuple[np.ndarray, list[int]]:
        """Get clinical features for all subjects. Returns (features, subject_ids)."""
        subject_ids = sorted(self.clinical_data["subjectID"].unique())
        features = []
        valid_ids = []
        for sid in subject_ids:
            try:
                features.append(self.get_clinical_features(sid))
                valid_ids.append(sid)
            except (ValueError, KeyError):
                continue
        return np.stack(features), valid_ids

    def get_fog_statistics(self) -> dict:
        """Compute per-subject FoG statistics."""
        stats = {}
        for sid in self.subject_ids:
            labels = self.get_subject_labels(sid)
            n_total = len(labels)
            n_fog = labels.sum()
            stats[sid] = {
                "n_samples": n_total,
                "n_fog": int(n_fog),
                "fog_ratio": float(n_fog / n_total) if n_total > 0 else 0.0,
                "duration_min": n_total / SAMPLING_RATE / 60,
                "fog_duration_sec": n_fog / SAMPLING_RATE,
            }
        return stats

    def summary(self) -> str:
        """Print dataset summary."""
        lines = [
            f"FoG-STAR Dataset",
            f"  Subjects: {self.n_subjects}",
            f"  Total samples: {len(self.sensor_data):,}",
            f"  Total duration: {len(self.sensor_data) / SAMPLING_RATE / 60:.1f} min",
            f"  FoG samples: {self.sensor_data['fog'].sum():,} ({self.sensor_data['fog'].mean()*100:.1f}%)",
            f"  IMU channels: {len(IMU_COLUMNS)}",
            f"  Sampling rate: {SAMPLING_RATE} Hz",
        ]
        return "\n".join(lines)
