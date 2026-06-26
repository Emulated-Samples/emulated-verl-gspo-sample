"""Score = max(0, 1 - relative_error), giving a smooth 0.0-1.0 signal
suitable for RL training rather than binary pass/fail.
"""

from __future__ import annotations

import torch


def compute_scalar_score(agent_val: torch.Tensor, gold_val: torch.Tensor) -> float:
    """Linear falloff: 1.0 at exact match, 0.0 at 100%+ relative error."""
    if not torch.isfinite(agent_val):
        return 0.0
    rel_error = (agent_val - gold_val).abs() / (gold_val.abs() + 1e-8)
    return max(0.0, 1.0 - rel_error.item())


def compute_tensor_score(
    agent_tensor: torch.Tensor, gold_tensor: torch.Tensor, mask: torch.Tensor
) -> float:
    """Linear falloff on masked tensor similarity."""
    mask_bool = mask.bool()
    if mask_bool.sum() == 0:
        return 1.0
    agent_masked = agent_tensor[mask_bool]
    gold_masked = gold_tensor[mask_bool]
    if not torch.isfinite(agent_masked).all():
        return 0.0
    rel_errors = (agent_masked - gold_masked).abs() / (gold_masked.abs() + 1e-8)
    return max(0.0, 1.0 - rel_errors.mean().item())
