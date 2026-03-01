"""
Temporal Masked Autoencoder (T-MAE) for self-supervised pre-training
on IMU sensor data.

Architecture:
- Patches 2-second IMU windows into temporal patches
- Masks a fraction of patches and reconstructs them
- The encoder learns universal gait representations usable for
  downstream FoG detection with per-patient adaptation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class PatchEmbedding(nn.Module):
    """Converts raw IMU windows into temporal patch embeddings."""

    def __init__(self, n_channels: int = 24, patch_size: int = 4, d_model: int = 128):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.proj = nn.Linear(n_channels * patch_size, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_channels)
        Returns:
            patches: (batch, n_patches, d_model)
        """
        B, L, C = x.shape
        n_patches = L // self.patch_size
        # Reshape into patches: (B, n_patches, patch_size * C)
        x = x[:, :n_patches * self.patch_size, :]
        x = x.reshape(B, n_patches, self.patch_size * C)
        x = self.norm(self.proj(x))
        return x


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class TransformerEncoder(nn.Module):
    """Standard Transformer encoder for T-MAE."""

    def __init__(self, d_model: int = 128, n_heads: int = 4, n_layers: int = 4,
                 d_ff: int = 256, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.pos_enc = SinusoidalPositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: (batch, n_patches, d_model)
            mask: optional attention mask
        Returns:
            encoded: (batch, n_patches, d_model)
        """
        x = self.pos_enc(x)
        x = self.encoder(x, src_key_padding_mask=mask)
        return self.norm(x)


class TemporalMAE(nn.Module):
    """
    Temporal Masked Autoencoder for self-supervised pre-training.

    Masks random temporal patches and trains the model to reconstruct
    the original sensor values from the visible patches.
    """

    def __init__(
        self,
        n_channels: int = 24,
        patch_size: int = 4,
        d_model: int = 128,
        n_heads: int = 4,
        n_encoder_layers: int = 4,
        n_decoder_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        mask_ratio: float = 0.5,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.patch_size = patch_size
        self.d_model = d_model
        self.mask_ratio = mask_ratio

        # Encoder
        self.patch_embed = PatchEmbedding(n_channels, patch_size, d_model)
        self.encoder = TransformerEncoder(d_model, n_heads, n_encoder_layers, d_ff, dropout)

        # Learnable mask token
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Decoder (lightweight)
        self.decoder_pos_enc = SinusoidalPositionalEncoding(d_model)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=n_decoder_layers)

        # Reconstruction head: predict original patch values
        self.reconstruct_head = nn.Linear(d_model, n_channels * patch_size)

    def _random_mask(self, n_patches: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate random mask indices."""
        n_mask = int(n_patches * self.mask_ratio)
        noise = torch.rand(n_patches, device=device)
        ids_shuffle = torch.argsort(noise)
        ids_masked = ids_shuffle[:n_mask]
        ids_visible = ids_shuffle[n_mask:]
        ids_visible = torch.sort(ids_visible).values
        ids_masked = torch.sort(ids_masked).values
        return ids_visible, ids_masked

    def forward_encoder(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode only visible patches.

        Returns:
            encoded: (batch, n_visible, d_model)
            ids_visible: (n_visible,)
            ids_masked: (n_masked,)
        """
        patches = self.patch_embed(x)  # (B, N, D)
        B, N, D = patches.shape

        ids_visible, ids_masked = self._random_mask(N, x.device)

        # Select visible patches
        visible = patches[:, ids_visible, :]  # (B, n_visible, D)
        encoded = self.encoder(visible)

        return encoded, ids_visible, ids_masked

    def forward_decoder(
        self, encoded: torch.Tensor, ids_visible: torch.Tensor,
        ids_masked: torch.Tensor, n_patches: int,
    ) -> torch.Tensor:
        """
        Reconstruct masked patches.

        Returns:
            reconstructed: (batch, n_masked, patch_size * n_channels)
        """
        B = encoded.shape[0]
        n_masked = len(ids_masked)

        # Create full sequence with mask tokens in masked positions
        full_tokens = torch.zeros(B, n_patches, self.d_model, device=encoded.device)
        full_tokens[:, ids_visible, :] = encoded
        full_tokens[:, ids_masked, :] = self.mask_token.expand(B, n_masked, -1)

        # Decode
        decoded = self.decoder_pos_enc(full_tokens)
        decoded = self.decoder(decoded)

        # Only reconstruct masked positions
        masked_decoded = decoded[:, ids_masked, :]
        reconstructed = self.reconstruct_head(masked_decoded)
        return reconstructed

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass: mask, encode, decode, reconstruct.

        Args:
            x: (batch, seq_len, n_channels)

        Returns:
            loss: scalar reconstruction loss
            reconstructed: (batch, n_masked, patch_size * n_channels)
            target: (batch, n_masked, patch_size * n_channels)
        """
        patches = self.patch_embed.proj.weight  # just to get device
        B, L, C = x.shape
        n_patches = L // self.patch_size

        # Get target patches (before masking)
        x_trimmed = x[:, :n_patches * self.patch_size, :]
        target_patches = x_trimmed.reshape(B, n_patches, self.patch_size * C)

        # Encode visible
        encoded, ids_visible, ids_masked = self.forward_encoder(x)

        # Decode and reconstruct masked
        reconstructed = self.forward_decoder(encoded, ids_visible, ids_masked, n_patches)

        # Compute loss only on masked patches
        target = target_patches[:, ids_masked, :]
        loss = F.mse_loss(reconstructed, target)

        return loss, reconstructed, target

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode without masking (for downstream tasks).

        Args:
            x: (batch, seq_len, n_channels)

        Returns:
            features: (batch, d_model) global representation
        """
        patches = self.patch_embed(x)
        encoded = self.encoder(patches)
        # Global average pooling over patches
        return encoded.mean(dim=1)  # (batch, d_model)

    def get_encoder_state(self) -> dict:
        """Get only encoder weights (for transfer to downstream)."""
        return {
            "patch_embed": self.patch_embed.state_dict(),
            "encoder": self.encoder.state_dict(),
        }

    def load_encoder_state(self, state: dict):
        """Load pre-trained encoder weights."""
        self.patch_embed.load_state_dict(state["patch_embed"])
        self.encoder.load_state_dict(state["encoder"])
