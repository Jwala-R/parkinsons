"""
Self-supervised pre-training script for Temporal Masked Autoencoder.

Pre-trains the T-MAE encoder on all available unlabeled IMU data
from both FoG-STAR and fog2 datasets. The resulting encoder learns
universal gait representations for downstream personalized FoG detection.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import time

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.transformer_mae import TemporalMAE
from src.data.fog_star_loader import FoGStarDataset
from src.data.fog2_loader import Fog2Dataset
from src.data.windowing import extract_windows


def prepare_pretrain_data(
    fog_star_dir: str,
    fog2_dir: str,
    window_seconds: float = 2.0,
    overlap: float = 0.5,
    target_sr: int = 60,
) -> np.ndarray:
    """
    Collect all IMU windows from both datasets (labels discarded for SSL).

    Returns:
        windows: (n_windows, window_size, n_channels)
    """
    window_size = int(window_seconds * target_sr)
    step_size = int(window_size * (1 - overlap))
    all_windows = []

    # FoG-STAR: 22 patients, 24 IMU channels
    print("Loading FoG-STAR...")
    fog_star = FoGStarDataset(fog_star_dir).load()
    for sid in fog_star.subject_ids:
        imu = fog_star.get_subject_imu(sid)
        labels = fog_star.get_subject_labels(sid)  # not used for SSL, but needed for windowing
        # Handle NaN
        if np.any(np.isnan(imu)):
            for col in range(imu.shape[1]):
                mask = np.isnan(imu[:, col])
                if mask.any() and not mask.all():
                    valid = np.where(~mask)[0]
                    imu[mask, col] = np.interp(np.where(mask)[0], valid, imu[valid, col])
                elif mask.all():
                    imu[:, col] = 0.0

        windows, _, _ = extract_windows(imu, labels, window_size, step_size)
        if len(windows) > 0:
            all_windows.append(windows)

    # fog2: 3 patients, extract ACC channels (24 channels matching FoG-STAR)
    print("Loading fog2...")
    fog2 = Fog2Dataset(fog2_dir, target_sr=target_sr).load()
    for pid in fog2.patient_ids:
        acc = fog2.get_patient_acc(pid, resample=True)  # (n_samples, 24)
        labels = fog2.get_patient_labels(pid, resample=True)
        windows, _, _ = extract_windows(acc, labels, window_size, step_size)
        if len(windows) > 0:
            all_windows.append(windows)

    all_windows = np.concatenate(all_windows, axis=0)
    print(f"Total pre-training windows: {len(all_windows):,}")
    return all_windows


def pretrain_mae(
    windows: np.ndarray,
    n_channels: int = 24,
    d_model: int = 128,
    n_heads: int = 4,
    n_encoder_layers: int = 4,
    n_decoder_layers: int = 2,
    d_ff: int = 256,
    dropout: float = 0.1,
    mask_ratio: float = 0.5,
    patch_size: int = 4,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    save_dir: str = "checkpoints",
    device: str = "auto",
) -> TemporalMAE:
    """
    Pre-train the T-MAE model.

    Args:
        windows: (n_windows, window_size, n_channels) training data
        ... model and training hyperparameters ...

    Returns:
        Trained TemporalMAE model
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Normalize data per channel (zero mean, unit variance)
    mean = windows.mean(axis=(0, 1), keepdims=True)
    std_val = windows.std(axis=(0, 1), keepdims=True)
    std_val[std_val == 0] = 1.0
    windows = (windows - mean) / std_val

    # Create dataloader
    tensor_data = torch.FloatTensor(windows)
    dataset = TensorDataset(tensor_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                           num_workers=0, pin_memory=True, drop_last=True)

    # Create model
    model = TemporalMAE(
        n_channels=n_channels,
        patch_size=patch_size,
        d_model=d_model,
        n_heads=n_heads,
        n_encoder_layers=n_encoder_layers,
        n_decoder_layers=n_decoder_layers,
        d_ff=d_ff,
        dropout=dropout,
        mask_ratio=mask_ratio,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"T-MAE parameters: {total_params:,}")

    # Optimizer with cosine LR schedule
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        start_time = time.time()

        for (batch,) in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss, _, _ = model(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / n_batches
        elapsed = time.time() - start_time

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.2e} | Time: {elapsed:.1f}s")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "model_state": model.state_dict(),
                "encoder_state": model.get_encoder_state(),
                "epoch": epoch,
                "loss": best_loss,
                "normalization": {"mean": mean, "std": std_val},
            }, save_path / "tmae_best.pt")

    # Save final
    torch.save({
        "model_state": model.state_dict(),
        "encoder_state": model.get_encoder_state(),
        "epoch": epochs,
        "loss": avg_loss,
        "normalization": {"mean": mean, "std": std_val},
    }, save_path / "tmae_final.pt")

    print(f"Pre-training complete. Best loss: {best_loss:.6f}")
    return model


if __name__ == "__main__":
    import yaml

    config_path = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    project_root = Path(__file__).resolve().parents[2]

    # Prepare data
    windows = prepare_pretrain_data(
        fog_star_dir=str(project_root / config["data"]["fog_star_path"]).replace("sensor_data.csv", "").rstrip("/"),
        fog2_dir=str(project_root / config["data"]["fog2_path"]),
        window_seconds=config["windowing"]["window_seconds"],
        overlap=config["windowing"]["overlap"],
        target_sr=config["data"]["sampling_rate"],
    )

    # Pre-train
    model = pretrain_mae(
        windows,
        d_model=config["transformer"]["d_model"],
        n_heads=config["transformer"]["n_heads"],
        n_encoder_layers=config["transformer"]["n_encoder_layers"],
        n_decoder_layers=config["transformer"]["n_decoder_layers"],
        d_ff=config["transformer"]["d_ff"],
        dropout=config["transformer"]["dropout"],
        patch_size=config["transformer"]["patch_size"],
        mask_ratio=config["training"]["pretrain"]["mask_ratio"],
        epochs=config["training"]["pretrain"]["epochs"],
        batch_size=config["training"]["pretrain"]["batch_size"],
        lr=config["training"]["pretrain"]["lr"],
        weight_decay=config["training"]["pretrain"]["weight_decay"],
        save_dir=str(project_root / "checkpoints"),
    )
