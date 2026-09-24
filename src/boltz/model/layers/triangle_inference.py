"""Eligibility for the opt-in, temporary-buffer-only triangle inference paths."""

import torch
from torch import Tensor, nn


def use_bf16_triangle_inference(module: nn.Module, x: Tensor) -> bool:
    """Keep training and the FP16/FP32 numerical policies on their original paths."""
    if (
        not module.inference_optimized
        or module.training
        or torch.is_grad_enabled()
        or x.device.type != "cuda"
        or x.dtype not in (torch.float32, torch.bfloat16)
        or not torch.is_autocast_enabled()
    ):
        return False
    # get_autocast_dtype was added after the oldest supported PyTorch release.
    dtype = (
        torch.get_autocast_dtype("cuda")
        if hasattr(torch, "get_autocast_dtype")
        else torch.get_autocast_gpu_dtype()
    )
    return dtype == torch.bfloat16
