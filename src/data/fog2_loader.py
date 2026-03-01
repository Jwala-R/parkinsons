"""
fog2 (Multimodal FoG) dataset loader.

Loads the 3-patient multimodal dataset with EEG, EMG, ECG, and ACC data.
Original sampling rate: 500 Hz, resampled to 60 Hz for compatibility with FoG-STAR.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import resample_poly
from math import gcd
from typing import Optional

# Column mapping for the 60-column filtered data files
COL_TIMESTAMP = [0]
COL_EEG = list(range(1, 26))  # 25 EEG channels
COL_EMG_ECG = list(range(26, 31))  # 5 EMG/ECG/EOG channels
COL_ACC_GYRO = list(range(31, 59))  # 28 ACC/Gyro channels (4 sensors x 7 cols)
COL_LABEL = [59]  # FoG label

EEG_NAMES = [
    "FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "P7", "P8", "Fz", "Cz", "Pz", "FC1", "FC2", "CP1",
    "CP2", "FC5", "FC6", "CP5", "CP6",
]

# ACC/Gyro layout: 4 sensors x 7 columns each
# Each sensor: acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, NC/SC
ACC_SENSOR_NAMES = ["LShank", "RShank", "Waist", "Arm"]
ACC_AXES = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z", "extra"]

# Columns that map to FoG-STAR-compatible ACC channels (excluding NC/SC columns)
# LShank: cols 31-36 (skip 37), RShank: 38-43 (skip 44), Waist: 45-50 (skip 51), Arm: 52-57 (skip 58)
ACC_USEFUL_COLS = [31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 43,
                   45, 46, 47, 48, 49, 50, 52, 53, 54, 55, 56, 57]

ORIGINAL_SR = 500  # Hz
TARGET_SR = 60  # Hz

# Patient directory structure
PATIENT_DIRS = {
    "004": {"tasks": ["task_1.txt", "task_2.txt", "task_3.txt", "task_4.txt", "task_5.txt"]},
    "008": {"sessions": {
        "OFF_1": ["task_1.txt", "task_2.txt", "task_3.txt", "task_4.txt", "task_5.txt"],
        "OFF_2": ["task_1.txt", "task_2.txt", "task_3.txt", "task_4.txt"],
    }},
    "011": {"tasks": ["task_1.txt", "task_2.txt", "task_3.txt", "task_4.txt"]},
}


def _resample_signal(data: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    """Resample signal from from_sr to to_sr using polyphase filtering."""
    g = gcd(from_sr, to_sr)
    up = to_sr // g
    down = from_sr // g
    return resample_poly(data, up, down, axis=0)


def _load_task_file(filepath: Path) -> Optional[np.ndarray]:
    """Load a single task .txt file as numpy array."""
    if not filepath.exists():
        print(f"Warning: {filepath} not found, skipping")
        return None
    try:
        data = np.loadtxt(filepath)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


class Fog2Dataset:
    """Loads and provides access to multimodal FoG dataset (fog2)."""

    def __init__(self, data_dir: str, target_sr: int = TARGET_SR):
        self.data_dir = Path(data_dir)
        self.target_sr = target_sr
        self.patient_data: dict[str, dict] = {}

    def load(self) -> "Fog2Dataset":
        """Load all patient data from task files."""
        for patient_id, structure in PATIENT_DIRS.items():
            patient_path = self.data_dir / patient_id
            if not patient_path.exists():
                print(f"Warning: patient directory {patient_path} not found")
                continue

            task_arrays = []

            if "sessions" in structure:
                for session_name, task_files in structure["sessions"].items():
                    session_path = patient_path / session_name
                    for tf in task_files:
                        data = _load_task_file(session_path / tf)
                        if data is not None:
                            task_arrays.append({
                                "data": data,
                                "session": session_name,
                                "task": tf,
                            })
            else:
                for tf in structure["tasks"]:
                    data = _load_task_file(patient_path / tf)
                    if data is not None:
                        task_arrays.append({
                            "data": data,
                            "session": "default",
                            "task": tf,
                        })

            if task_arrays:
                self.patient_data[patient_id] = task_arrays
                total_samples = sum(t["data"].shape[0] for t in task_arrays)
                print(f"Patient {patient_id}: loaded {len(task_arrays)} tasks, "
                      f"{total_samples:,} samples @ {ORIGINAL_SR}Hz")

        return self

    @property
    def patient_ids(self) -> list[str]:
        return sorted(self.patient_data.keys())

    @property
    def n_patients(self) -> int:
        return len(self.patient_data)

    def get_patient_raw(self, patient_id: str) -> np.ndarray:
        """Get concatenated raw data for a patient (all tasks). Shape: (n_samples, 60)."""
        tasks = self.patient_data[patient_id]
        return np.vstack([t["data"] for t in tasks])

    def get_patient_eeg(self, patient_id: str, resample: bool = True) -> np.ndarray:
        """Get EEG data (25 channels) for a patient, optionally resampled."""
        raw = self.get_patient_raw(patient_id)
        eeg = raw[:, COL_EEG].astype(np.float32)
        if resample:
            eeg = _resample_signal(eeg, ORIGINAL_SR, self.target_sr).astype(np.float32)
        return eeg

    def get_patient_acc(self, patient_id: str, resample: bool = True) -> np.ndarray:
        """Get ACC/Gyro data (24 useful channels) for a patient, optionally resampled.
        This matches the channel count of FoG-STAR IMU data.
        """
        raw = self.get_patient_raw(patient_id)
        acc = raw[:, ACC_USEFUL_COLS].astype(np.float32)
        if resample:
            acc = _resample_signal(acc, ORIGINAL_SR, self.target_sr).astype(np.float32)
        return acc

    def get_patient_labels(self, patient_id: str, resample: bool = True) -> np.ndarray:
        """Get FoG labels for a patient, optionally resampled (nearest neighbor)."""
        raw = self.get_patient_raw(patient_id)
        labels = raw[:, 59].astype(np.float32)
        if resample:
            n_out = int(len(labels) * self.target_sr / ORIGINAL_SR)
            indices = np.linspace(0, len(labels) - 1, n_out).astype(int)
            labels = labels[indices]
        return (labels > 0.5).astype(np.int64)

    def get_patient_multimodal(self, patient_id: str, resample: bool = True) -> dict:
        """Get all modalities for a patient as a dict of arrays."""
        return {
            "eeg": self.get_patient_eeg(patient_id, resample),
            "acc": self.get_patient_acc(patient_id, resample),
            "labels": self.get_patient_labels(patient_id, resample),
        }

    def get_fog_statistics(self) -> dict:
        """Compute per-patient FoG statistics."""
        stats = {}
        for pid in self.patient_ids:
            labels = self.get_patient_labels(pid, resample=True)
            n_total = len(labels)
            n_fog = labels.sum()
            stats[pid] = {
                "n_samples": n_total,
                "n_fog": int(n_fog),
                "fog_ratio": float(n_fog / n_total) if n_total > 0 else 0.0,
                "duration_min": n_total / self.target_sr / 60,
                "fog_duration_sec": n_fog / self.target_sr,
            }
        return stats

    def summary(self) -> str:
        lines = [f"fog2 Multimodal Dataset", f"  Patients: {self.n_patients}"]
        for pid in self.patient_ids:
            raw = self.get_patient_raw(pid)
            labels = raw[:, 59]
            n_fog = (labels > 0.5).sum()
            lines.append(
                f"  Patient {pid}: {raw.shape[0]:,} samples @ {ORIGINAL_SR}Hz, "
                f"FoG: {n_fog:,} ({n_fog/len(labels)*100:.1f}%)"
            )
        return "\n".join(lines)
