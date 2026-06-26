from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, List

import torch
from verl.workers.config.actor import ActorConfig
from verl.workers.config.optimizer import OptimizerConfig

from gspo_helpers import normal_gen, uniform_gen, laplace_gen, cauchy_gen, exponential_gen


ExtraCheckFn = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor],
    None,
]
InputTransformFn = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
]


@dataclass
class Scenario:
    name: str
    old_log_prob_gen: Callable[[], torch.Tensor]
    log_prob_gen: Callable[[torch.Tensor], torch.Tensor]
    adv_gen: Callable[[], torch.Tensor]
    mask_mode: str
    seed: int
    batch_size: int = 32
    seq_len: int = 2000
    config: Optional[ActorConfig] = None
    loss_agg_mode: str = "seq-mean-token-mean"
    input_transform: Optional[InputTransformFn] = None
    extra_checks: Optional[ExtraCheckFn] = None


def _default_config() -> ActorConfig:
    return ActorConfig(
        strategy="fsdp",
        clip_ratio=0.2,
        loss_agg_mode="seq-mean-token-mean",
        use_dynamic_bsz=True,
        optim=OptimizerConfig(lr=1e-4),
    )


def _zero_adv_checks(pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower, grad_gt, response_mask):
    mask_bool = response_mask.bool()
    assert torch.allclose(pg_loss, torch.tensor(0.0, device=pg_loss.device), atol=1e-6), \
        "[zero_adv] pg_loss not ~0"
    if grad_gt is not None:
        assert torch.allclose(grad_gt[mask_bool], torch.zeros_like(grad_gt[mask_bool]), atol=1e-7), \
            "[zero_adv] gradients not zero under mask"


def _padding_grad_checks(pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower, grad, response_mask):
    if grad is None:
        return
    pad = response_mask == 0
    assert pad.any(), "[padding] scenario did not include padded tokens"
    assert torch.allclose(grad[pad], torch.zeros_like(grad[pad]), atol=1e-8), \
        "[padding] padded tokens received non-zero gradients"


def _add_padding_junk(old_log_prob, log_prob, advantages, response_mask):
    pad = response_mask == 0
    old_junk = 50.0 * torch.randn_like(old_log_prob) + 10.0
    log_junk = -50.0 * torch.randn_like(log_prob) - 10.0
    adv_junk = 1000.0 * torch.randn_like(advantages)
    return (
        torch.where(pad, old_junk, old_log_prob),
        torch.where(pad, log_junk, log_prob),
        torch.where(pad, adv_junk, advantages),
        response_mask,
    )


BS, SL = 32, 2000


def build_scenarios() -> List[Scenario]:
    """Lazy generators — tensors are created after torch.manual_seed in the runner."""
    return [
        # 1: Baseline normal with random mask
        Scenario(
            name="normal_random_mask",
            old_log_prob_gen=lambda: torch.randn(BS, SL),
            log_prob_gen=lambda old: old + 0.1 * torch.randn(BS, SL),
            adv_gen=lambda: torch.randn(BS, SL),
            mask_mode="random",
            seed=1000,
        ),
        # 2: Uniform with all-ones mask
        Scenario(
            name="uniform_all_ones",
            old_log_prob_gen=uniform_gen(-3.0, 0.0, BS, SL),
            log_prob_gen=lambda old: old + 0.2 * (uniform_gen(-1.0, 1.0, BS, SL)()),
            adv_gen=uniform_gen(-5.0, 5.0, BS, SL),
            mask_mode="all_ones",
            seed=2000,
        ),
        # 3: Laplace noise with single-token mask
        Scenario(
            name="laplace_single_token",
            old_log_prob_gen=lambda: laplace_gen(1.5, BS, SL)(),
            log_prob_gen=lambda old: old + 0.3 * laplace_gen(0.5, BS, SL)(),
            adv_gen=lambda: laplace_gen(2.0, BS, SL)(),
            mask_mode="single_token",
            seed=3000,
        ),
        # 4: Heavy-tailed Cauchy with sparse mask
        Scenario(
            name="cauchy_sparse",
            old_log_prob_gen=lambda: cauchy_gen(1.0, 50.0, BS, SL)(),
            log_prob_gen=lambda old: old + 0.1 * cauchy_gen(1.0, 50.0, BS, SL)(),
            adv_gen=lambda: cauchy_gen(2.0, 50.0, BS, SL)(),
            mask_mode="sparse_10pct",
            seed=4000,
        ),
        # 5: Exponential positive advantages, dense mask
        Scenario(
            name="exponential_adv_dense",
            old_log_prob_gen=lambda: torch.randn(BS, SL),
            log_prob_gen=lambda old: old + 0.05 * torch.randn(BS, SL),
            adv_gen=lambda: exponential_gen(0.5, BS, SL)(),
            mask_mode="dense_90pct",
            seed=5000,
        ),
        # 6: Large magnitudes in log-probs and advantages
        Scenario(
            name="big_magnitudes",
            old_log_prob_gen=lambda: -1000.0 + 5.0 * torch.randn(BS, SL),
            log_prob_gen=lambda old: old + 5.0 + 0.5 * torch.randn(BS, SL),
            adv_gen=lambda: 1e6 * (0.5 * torch.randn(BS, SL) + 1.0),
            mask_mode="all_ones",
            seed=6000,
        ),
        # 7: Zero advantages yield zero loss and gradients
        Scenario(
            name="zero_advantages",
            old_log_prob_gen=lambda: torch.randn(BS, SL),
            log_prob_gen=lambda old: old + 0.1 * torch.randn(BS, SL),
            adv_gen=lambda: torch.zeros(BS, SL),
            mask_mode="contiguous_front",
            seed=7000,
            extra_checks=_zero_adv_checks,
        ),
        # 8: Gradient isolation — uniform offset makes forward trivially correct,
        # but gradients differ with/without the detach trick.
        # Advantages MUST vary across tokens; constant advantages produce identical
        # gradients regardless of detach usage because the coupling term cancels.
        Scenario(
            name="gradient_isolation",
            old_log_prob_gen=lambda: torch.randn(4, 8),
            log_prob_gen=lambda old: old + 0.1,
            adv_gen=lambda: torch.randn(4, 8),
            mask_mode="all_ones",
            seed=8000,
            batch_size=4,
            seq_len=8,
        ),
        # 9: Inputs near clip boundary to test clipping precision
        Scenario(
            name="ratio_near_clip_boundary",
            old_log_prob_gen=lambda: torch.randn(8, 16) * 2.0 - 3.0,
            log_prob_gen=lambda old: old + torch.tensor([
                -0.25, -0.21, -0.16, -0.105, 0.105, 0.14, 0.175, 0.20,
            ]).unsqueeze(1).expand(8, 16) + torch.randn(8, 16) * 0.01,
            adv_gen=lambda: torch.randn(8, 16) * 3.0,
            mask_mode="all_ones",
            seed=9000,
            batch_size=8,
            seq_len=16,
        ),
        # 10: Asymmetric clip ratios (clip_ratio_low=0.1, clip_ratio_high=0.3)
        Scenario(
            name="asymmetric_clip",
            old_log_prob_gen=lambda: torch.randn(16, 16) * 2.0 - 3.0,
            log_prob_gen=lambda old: old + torch.randn(16, 16) * 0.3,
            adv_gen=lambda: torch.randn(16, 16) * 3.0,
            mask_mode="random",
            seed=10000,
            batch_size=16,
            seq_len=16,
            config=ActorConfig(
                strategy="fsdp",
                clip_ratio=0.2,
                clip_ratio_low=0.1,
                clip_ratio_high=0.3,
                loss_agg_mode="seq-mean-token-mean",
                use_dynamic_bsz=True,
                optim=OptimizerConfig(lr=1e-4),
            ),
        ),
        # 11: On-policy (log_prob == old_log_prob), ratio = 1.0 everywhere
        Scenario(
            name="on_policy",
            old_log_prob_gen=lambda: torch.randn(8, 32),
            log_prob_gen=lambda old: old.clone(),
            adv_gen=lambda: torch.randn(8, 32),
            mask_mode="random",
            seed=11000,
            batch_size=8,
            seq_len=32,
        ),
        # 12: Call-site integration -- GSPO must use seq-mean-token-mean even
        # if a caller passes a different aggregation mode.
        Scenario(
            name="aggregation_override",
            old_log_prob_gen=lambda: torch.randn(12, 32),
            log_prob_gen=lambda old: old + 0.15 * torch.randn(12, 32),
            adv_gen=lambda: torch.randn(12, 32),
            mask_mode="random",
            seed=12000,
            batch_size=12,
            seq_len=32,
            loss_agg_mode="token-mean",
        ),
        # 13: Padded-token invariance -- masked padding can contain arbitrary
        # stale values from packed batches and must not affect values or gradients.
        Scenario(
            name="padding_junk_mask",
            old_log_prob_gen=lambda: torch.randn(12, 64),
            log_prob_gen=lambda old: old + 0.12 * torch.randn(12, 64),
            adv_gen=lambda: torch.randn(12, 64),
            mask_mode="contiguous_front_with_padding",
            seed=13000,
            batch_size=12,
            seq_len=64,
            input_transform=_add_padding_junk,
            extra_checks=_padding_grad_checks,
        ),
    ]


SCENARIOS = build_scenarios()
