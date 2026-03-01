"""
Personalized FoG Detector (Approach B).

Combines:
1. Pre-trained T-MAE encoder for feature extraction
2. LoRA adapters for per-patient personalization
3. Classification head with clinical conditioning
4. Online adaptation with experience replay and drift detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque

from .transformer_mae import TemporalMAE, TransformerEncoder, PatchEmbedding
from .lora_adapter import inject_lora, get_lora_params, save_lora_state, load_lora_state, reset_all_lora


class ClinicalConditioner(nn.Module):
    """Maps clinical features to a conditioning vector."""

    def __init__(self, n_clinical: int = 9, d_model: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_clinical, 64),
            nn.ReLU(),
            nn.Linear(64, d_model),
        )

    def forward(self, clinical: torch.Tensor) -> torch.Tensor:
        """
        Args:
            clinical: (batch, n_clinical)
        Returns:
            conditioning: (batch, d_model)
        """
        return self.net(clinical)


class PersonalizedFoGDetector(nn.Module):
    """
    End-to-end personalized FoG detector.

    Uses a pre-trained encoder (from T-MAE) with LoRA adapters
    for per-patient personalization, and a classification head
    conditioned on clinical metadata.
    """

    def __init__(
        self,
        n_channels: int = 24,
        patch_size: int = 4,
        d_model: int = 128,
        n_heads: int = 4,
        n_encoder_layers: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
        n_clinical: int = 9,
        lora_rank: int = 4,
        lora_alpha: float = 8.0,
    ):
        super().__init__()
        self.d_model = d_model

        # Encoder (will be loaded from pre-trained T-MAE)
        self.patch_embed = PatchEmbedding(n_channels, patch_size, d_model)
        self.encoder = TransformerEncoder(d_model, n_heads, n_encoder_layers, d_ff, dropout)

        # Clinical conditioning
        self.clinical_conditioner = ClinicalConditioner(n_clinical, d_model)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, 64),  # encoder features + clinical conditioning
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

        # LoRA will be injected after loading pre-trained weights
        self.lora_layers = []
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha

    def load_pretrained_encoder(self, mae: TemporalMAE):
        """Load encoder weights from pre-trained T-MAE."""
        self.patch_embed.load_state_dict(mae.patch_embed.state_dict())
        self.encoder.load_state_dict(mae.encoder.state_dict())
        print("Loaded pre-trained encoder weights from T-MAE")

    def inject_lora_adapters(self):
        """Inject LoRA adapters into encoder and freeze base weights."""
        # Freeze encoder
        for param in self.patch_embed.parameters():
            param.requires_grad = False
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Inject LoRA
        self.encoder, self.lora_layers = inject_lora(
            self.encoder,
            rank=self.lora_rank,
            alpha=self.lora_alpha,
        )

        # Move newly created LoRA params to same device as model
        device = next(self.parameters()).device
        for lora_layer in self.lora_layers:
            lora_layer.to(device)

    def get_trainable_params(self) -> list[nn.Parameter]:
        """Get parameters that should be updated during fine-tuning."""
        params = get_lora_params(self)
        params.extend(self.classifier.parameters())
        params.extend(self.clinical_conditioner.parameters())
        return params

    def forward(
        self,
        x: torch.Tensor,
        clinical: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_channels) IMU window
            clinical: (batch, n_clinical) clinical features (optional)

        Returns:
            logits: (batch, 1) FoG logits
        """
        # Encode
        patches = self.patch_embed(x)
        encoded = self.encoder(patches)
        features = encoded.mean(dim=1)  # (batch, d_model) global pooling

        # Clinical conditioning
        if clinical is not None:
            cond = self.clinical_conditioner(clinical)
            combined = torch.cat([features, cond], dim=-1)  # (batch, d_model*2)
        else:
            # Zero conditioning if no clinical data
            combined = torch.cat([features, torch.zeros_like(features)], dim=-1)

        logits = self.classifier(combined)
        return logits

    def predict_proba(
        self,
        x: torch.Tensor,
        clinical: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Get FoG probability."""
        logits = self.forward(x, clinical)
        return torch.sigmoid(logits).squeeze(-1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Get encoder features only (for hybrid approach)."""
        patches = self.patch_embed(x)
        encoded = self.encoder(patches)
        return encoded.mean(dim=1)


class OnlineAdapter:
    """
    Online adaptation manager for the personalized detector.

    Handles:
    - Experience replay buffer
    - Pseudo-labeling for unlabeled data
    - Distribution drift detection
    - Continual LoRA adaptation
    """

    def __init__(
        self,
        model: PersonalizedFoGDetector,
        buffer_size: int = 500,
        adaptation_lr: float = 1e-4,
        pseudo_label_threshold: float = 0.9,
        drift_threshold: float = 3.0,
    ):
        self.model = model
        self.buffer_size = buffer_size
        self.adaptation_lr = adaptation_lr
        self.pseudo_label_threshold = pseudo_label_threshold
        self.drift_threshold = drift_threshold

        # Experience replay buffer (stratified)
        self.fog_buffer = deque(maxlen=buffer_size // 2)
        self.nonfog_buffer = deque(maxlen=buffer_size // 2)

        # Feature distribution tracking for drift detection
        self.feature_mean: torch.Tensor | None = None
        self.feature_cov_inv: torch.Tensor | None = None
        self.n_features_seen = 0

        # Optimizer for LoRA params + classifier
        trainable = model.get_trainable_params()
        self.optimizer = torch.optim.Adam(trainable, lr=adaptation_lr)

    def add_to_buffer(
        self,
        windows: torch.Tensor,
        labels: torch.Tensor,
        clinical: torch.Tensor | None = None,
    ):
        """Add labeled samples to the replay buffer."""
        for i in range(len(windows)):
            entry = {
                "window": windows[i].detach().cpu(),
                "label": labels[i].item(),
                "clinical": clinical[i].detach().cpu() if clinical is not None else None,
            }
            if labels[i].item() == 1:
                self.fog_buffer.append(entry)
            else:
                self.nonfog_buffer.append(entry)

    def sample_replay_batch(self, batch_size: int = 32) -> tuple | None:
        """Sample a balanced batch from replay buffer."""
        n_fog = min(batch_size // 2, len(self.fog_buffer))
        n_nonfog = min(batch_size // 2, len(self.nonfog_buffer))
        if n_fog == 0 and n_nonfog == 0:
            return None

        import random
        fog_samples = random.sample(list(self.fog_buffer), n_fog) if n_fog > 0 else []
        nonfog_samples = random.sample(list(self.nonfog_buffer), n_nonfog) if n_nonfog > 0 else []
        samples = fog_samples + nonfog_samples
        random.shuffle(samples)

        windows = torch.stack([s["window"] for s in samples])
        labels = torch.tensor([s["label"] for s in samples], dtype=torch.float32)
        clinical = None
        if samples[0]["clinical"] is not None:
            clinical = torch.stack([s["clinical"] for s in samples])

        return windows, labels, clinical

    def adapt_step(
        self,
        windows: torch.Tensor,
        labels: torch.Tensor,
        clinical: torch.Tensor | None = None,
        n_steps: int = 5,
    ) -> float:
        """
        Perform adaptation steps on new labeled data + replay.

        Returns:
            average loss over adaptation steps
        """
        device = next(self.model.parameters()).device
        self.model.train()
        total_loss = 0.0

        # Add new data to buffer
        self.add_to_buffer(windows, labels, clinical)

        for step in range(n_steps):
            self.optimizer.zero_grad()

            # Mix new data with replay
            replay = self.sample_replay_batch(batch_size=32)
            if replay is not None:
                r_windows, r_labels, r_clinical = replay
                all_windows = torch.cat([windows, r_windows.to(device)])
                all_labels = torch.cat([labels, r_labels.to(device)])
                if clinical is not None and r_clinical is not None:
                    all_clinical = torch.cat([clinical, r_clinical.to(device)])
                else:
                    all_clinical = None
            else:
                all_windows = windows
                all_labels = labels
                all_clinical = clinical

            logits = self.model(all_windows, all_clinical).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, all_labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / n_steps

    def pseudo_label(
        self,
        windows: torch.Tensor,
        clinical: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """
        Generate pseudo-labels for unlabeled data using high-confidence predictions.

        Returns:
            (high_conf_windows, pseudo_labels) or None if no confident predictions
        """
        self.model.eval()
        with torch.no_grad():
            probs = self.model.predict_proba(windows, clinical)

        high_conf_mask = (probs > self.pseudo_label_threshold) | (probs < (1 - self.pseudo_label_threshold))
        if high_conf_mask.sum() == 0:
            return None

        pseudo_labels = (probs[high_conf_mask] > 0.5).float()
        return windows[high_conf_mask], pseudo_labels

    def update_feature_distribution(self, windows: torch.Tensor):
        """Update running feature distribution for drift detection."""
        self.model.eval()
        with torch.no_grad():
            features = self.model.encode(windows)  # (batch, d_model)

        features_np = features.cpu().numpy()
        batch_mean = features_np.mean(axis=0)
        batch_cov = np.cov(features_np.T) if len(features_np) > 1 else np.eye(features_np.shape[1])

        if self.feature_mean is None:
            self.feature_mean = batch_mean
            self.feature_cov_inv = np.linalg.pinv(batch_cov + np.eye(batch_cov.shape[0]) * 1e-6)
            self.n_features_seen = len(features_np)
        else:
            # Exponential moving average
            alpha = 0.1
            self.feature_mean = (1 - alpha) * self.feature_mean + alpha * batch_mean
            new_cov = (1 - alpha) * np.linalg.pinv(self.feature_cov_inv) + alpha * batch_cov
            self.feature_cov_inv = np.linalg.pinv(new_cov + np.eye(new_cov.shape[0]) * 1e-6)
            self.n_features_seen += len(features_np)

    def check_drift(self, windows: torch.Tensor) -> float:
        """
        Check for distribution drift using Mahalanobis distance.

        Returns:
            drift_score: Mahalanobis distance (higher = more drift)
        """
        if self.feature_mean is None:
            return 0.0

        self.model.eval()
        with torch.no_grad():
            features = self.model.encode(windows)

        features_np = features.cpu().numpy().mean(axis=0)
        diff = features_np - self.feature_mean
        mahal = float(np.sqrt(diff @ self.feature_cov_inv @ diff))
        return mahal

    def should_readapt(self, windows: torch.Tensor) -> bool:
        """Check if drift warrants re-adaptation."""
        drift = self.check_drift(windows)
        return drift > self.drift_threshold
