"""
LoRA (Low-Rank Adaptation) for per-patient personalization.

Injects low-rank trainable weight deltas into Transformer attention layers.
Only the LoRA parameters (~200-400 per patient) are updated during
personalization, keeping the pre-trained encoder frozen.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LoRALinear(nn.Module):
    """
    Linear layer with LoRA adaptation.

    Replaces W with W + BA where B is (d_out, rank) and A is (rank, d_in).
    Only A and B are trainable; the original W is frozen.
    """

    def __init__(
        self,
        original_linear: nn.Linear,
        rank: int = 4,
        alpha: float = 8.0,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.original = original_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        d_in = original_linear.in_features
        d_out = original_linear.out_features

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(rank, d_in) * (1.0 / math.sqrt(d_in)))
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Freeze original weights
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False

    @property
    def weight(self) -> torch.Tensor:
        """Expose original weight for compatibility with nn.MultiheadAttention."""
        return self.original.weight

    @property
    def bias(self) -> torch.Tensor | None:
        """Expose original bias for compatibility with nn.MultiheadAttention."""
        return self.original.bias

    @property
    def in_features(self) -> int:
        return self.original.in_features

    @property
    def out_features(self) -> int:
        return self.original.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original forward
        result = self.original(x)
        # LoRA delta
        lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
        return result + lora_out

    def reset_lora(self):
        """Reset LoRA parameters (for new patient)."""
        nn.init.normal_(self.lora_A, std=1.0 / math.sqrt(self.lora_A.shape[1]))
        nn.init.zeros_(self.lora_B)

    @property
    def n_lora_params(self) -> int:
        return self.lora_A.numel() + self.lora_B.numel()


def inject_lora(
    model: nn.Module,
    rank: int = 4,
    alpha: float = 8.0,
    dropout: float = 0.05,
    target_modules: list[str] | None = None,
) -> tuple[nn.Module, list[LoRALinear]]:
    """
    Inject LoRA adapters into a model's linear layers.

    By default, targets attention projection layers (q, k, v, out).

    Args:
        model: the model to modify
        rank: LoRA rank
        alpha: LoRA scaling factor
        dropout: LoRA dropout rate
        target_modules: list of substrings to match in parameter names.
                       If None, targets all linear layers in attention.

    Returns:
        model: modified model (in-place)
        lora_layers: list of injected LoRALinear modules
    """
    if target_modules is None:
        # Only target FFN layers in TransformerEncoderLayer.
        # Skip attention projections (in_proj, out_proj) because PyTorch's
        # nn.MultiheadAttention uses F.multi_head_attention_forward which
        # bypasses module.forward() and accesses .weight/.bias directly.
        target_modules = ["linear1", "linear2"]

    lora_layers = []

    # Collect replacements first to avoid modifying during iteration
    replacements = []
    for module_name, module in model.named_modules():
        for child_name, child in module.named_children():
            if isinstance(child, nn.Linear):
                full_name = f"{module_name}.{child_name}" if module_name else child_name
                if any(target in full_name for target in target_modules):
                    replacements.append((module, child_name, child))

    # Apply replacements
    for parent, name, linear in replacements:
        lora = LoRALinear(linear, rank=rank, alpha=alpha, dropout=dropout)
        setattr(parent, name, lora)
        lora_layers.append(lora)

    total_lora_params = sum(l.n_lora_params for l in lora_layers)
    print(f"Injected LoRA into {len(lora_layers)} layers, "
          f"total LoRA params: {total_lora_params}")

    return model, lora_layers


def get_lora_params(model: nn.Module) -> list[nn.Parameter]:
    """Get only LoRA parameters from a model (for optimizer)."""
    params = []
    for module in model.modules():
        if isinstance(module, LoRALinear):
            params.extend([module.lora_A, module.lora_B])
    return params


def save_lora_state(model: nn.Module) -> dict:
    """Save only LoRA adapter weights."""
    state = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            state[name] = {
                "lora_A": module.lora_A.data.clone(),
                "lora_B": module.lora_B.data.clone(),
            }
    return state


def load_lora_state(model: nn.Module, state: dict):
    """Load LoRA adapter weights."""
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear) and name in state:
            module.lora_A.data.copy_(state[name]["lora_A"])
            module.lora_B.data.copy_(state[name]["lora_B"])


def reset_all_lora(model: nn.Module):
    """Reset all LoRA adapters in a model."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.reset_lora()
