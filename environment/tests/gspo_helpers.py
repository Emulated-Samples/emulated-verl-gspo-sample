from __future__ import annotations

import torch


def make_mask(mode: str, batch_size: int, seq_len: int) -> torch.Tensor:
    """Create response masks with various sparsity patterns."""
    mask = torch.zeros(batch_size, seq_len, dtype=torch.float32)

    if mode == "random":
        for i in range(batch_size):
            k = torch.randint(1, seq_len + 1, (1,)).item()
            idx = torch.randperm(seq_len)[:k]
            mask[i, idx] = 1.0
    elif mode == "all_ones":
        mask[:] = 1.0
    elif mode == "single_token":
        for i in range(batch_size):
            j = torch.randint(0, seq_len, (1,)).item()
            mask[i, j] = 1.0
    elif mode == "sparse_10pct":
        k = max(1, int(0.1 * seq_len))
        for i in range(batch_size):
            idx = torch.randperm(seq_len)[:k]
            mask[i, idx] = 1.0
    elif mode == "dense_90pct":
        k = max(1, int(0.9 * seq_len))
        for i in range(batch_size):
            idx = torch.randperm(seq_len)[:k]
            mask[i, idx] = 1.0
    elif mode == "contiguous_front":
        for i in range(batch_size):
            k = torch.randint(1, seq_len + 1, (1,)).item()
            mask[i, :k] = 1.0
    elif mode == "contiguous_front_with_padding":
        for i in range(batch_size):
            k = torch.randint(1, seq_len, (1,)).item()
            mask[i, :k] = 1.0
    else:
        raise ValueError(f"Unknown mask mode: {mode}")

    return mask


def normal_gen(bs: int, sl: int):
    return lambda: torch.randn(bs, sl)


def uniform_gen(low: float, high: float, bs: int, sl: int):
    return lambda: (high - low) * torch.rand(bs, sl) + low


def laplace_gen(scale: float, bs: int, sl: int):
    d = torch.distributions.Laplace(loc=torch.tensor(0.0), scale=torch.tensor(scale))
    return lambda: d.sample((bs, sl))


def cauchy_gen(scale: float, clip: float, bs: int, sl: int):
    d = torch.distributions.Cauchy(loc=torch.tensor(0.0), scale=torch.tensor(scale))
    return lambda: d.sample((bs, sl)).clamp(min=-clip, max=clip)


def exponential_gen(rate: float, bs: int, sl: int):
    d = torch.distributions.Exponential(rate=torch.tensor(rate))
    return lambda: d.sample((bs, sl))
