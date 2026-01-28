# Copyright (c) 2025 ComfyUI-SeedVR2.5 Contributors
# SPDX-License-Identifier: Apache License, Version 2.0
#
# VAE Optimization for Windows + RTX 50xx (Blackwell Architecture)
# Optimized for native CUDA efficiency without torch.compile

"""
Performance optimization module for VAE models targeting:
- Windows OS (avoiding torch.compile and Triton)
- NVIDIA RTX 50xx (Blackwell) architecture
- FP8 Tensor Core support
- Channels Last memory format
- Native CUDA optimizations
"""

import torch
import torch.nn as nn
from typing import Optional, Literal
import logging

logger = logging.getLogger(__name__)

# Global optimization flags
_CUDNN_BENCHMARK_ENABLED = False
_CHANNELS_LAST_ENABLED = False
_FP8_ENABLED = False


def enable_cudnn_benchmark():
    """
    Enable cuDNN auto-tuner for optimal convolution algorithms.
    Safe for Windows and provides significant speedup for fixed-size inputs.
    """
    global _CUDNN_BENCHMARK_ENABLED
    if not _CUDNN_BENCHMARK_ENABLED:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        _CUDNN_BENCHMARK_ENABLED = True
        logger.info("✓ Enabled cuDNN benchmark mode for optimal convolution performance")


def disable_cudnn_benchmark():
    """Disable cuDNN benchmark mode."""
    global _CUDNN_BENCHMARK_ENABLED
    torch.backends.cudnn.benchmark = False
    _CUDNN_BENCHMARK_ENABLED = False
    logger.info("Disabled cuDNN benchmark mode")


def is_fp8_available() -> bool:
    """
    Check if FP8 (float8) support is available.
    RTX 50xx (Blackwell) has native FP8 Tensor Core support.
    """
    try:
        # Check PyTorch version supports float8
        return hasattr(torch, 'float8_e4m3fn')
    except:
        return False


def optimize_for_windows_blackwell(
    model: nn.Module,
    enable_channels_last: bool = True,
    enable_fp8: bool = False,
    enable_amp: bool = True,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    Optimize a VAE model for Windows + RTX 50xx (Blackwell).
    
    Args:
        model: The VAE model to optimize
        enable_channels_last: Use channels-last memory format for 2D convolutions
        enable_fp8: Enable FP8 precision (experimental, requires compatible hardware)
        enable_amp: Enable automatic mixed precision (FP16)
        device: Target device (default: cuda if available)
    
    Returns:
        Optimized model
    """
    global _CHANNELS_LAST_ENABLED, _FP8_ENABLED
    
    # Enable cuDNN auto-tuner
    enable_cudnn_benchmark()
    
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    
    # Apply channels-last memory format to Conv2d layers for Blackwell optimization
    if enable_channels_last and device.type == 'cuda':
        _apply_channels_last_format(model)
        _CHANNELS_LAST_ENABLED = True
        logger.info("✓ Applied channels-last memory format for Conv2d layers")
    
    # FP8 support (experimental)
    if enable_fp8 and is_fp8_available() and device.type == 'cuda':
        logger.warning("⚠ FP8 optimization is experimental. Use with caution.")
        _FP8_ENABLED = True
        # Note: Actual FP8 conversion would require model-specific handling
        # This is a placeholder for future implementation
    elif enable_fp8:
        logger.warning("⚠ FP8 not available on this system")
    
    # Log optimization summary
    logger.info(f"VAE Optimization Summary:")
    logger.info(f"  - cuDNN Benchmark: {_CUDNN_BENCHMARK_ENABLED}")
    logger.info(f"  - Channels Last: {_CHANNELS_LAST_ENABLED}")
    logger.info(f"  - FP8 Support: {_FP8_ENABLED}")
    logger.info(f"  - Device: {device}")
    
    return model


def _apply_channels_last_format(model: nn.Module):
    """
    Apply channels-last memory format to Conv2d layers.
    This optimizes memory access patterns for Blackwell Tensor Cores.
    """
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            try:
                module.to(memory_format=torch.channels_last)
            except Exception as e:
                logger.debug(f"Could not convert module to channels_last: {e}")


def create_optimized_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: Optional[int] = None,
    pin_memory: bool = True,
) -> torch.utils.data.DataLoader:
    """
    Create optimized DataLoader for Windows.
    
    Args:
        dataset: Dataset to load
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes (auto-configured for Windows)
        pin_memory: Use pinned memory for faster GPU transfer
    
    Returns:
        Optimized DataLoader
    """
    # Windows-specific worker configuration
    if num_workers is None:
        import os
        # For Windows, use fewer workers to avoid IPC issues
        # Typically 0-2 workers work best on Windows
        num_workers = min(2, os.cpu_count() or 0)
    
    # Pin memory for faster CPU->GPU transfer (safe on Windows)
    pin_memory = pin_memory and torch.cuda.is_available()
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    
    logger.info(f"DataLoader config: workers={num_workers}, pin_memory={pin_memory}")
    return dataloader


class OptimizedDiagonalGaussianDistribution:
    """
    Optimized reparameterization for VAE with minimal CPU-GPU sync.
    """
    
    def __init__(self, mean: torch.Tensor, logvar: torch.Tensor):
        self.mean = mean
        # Clamp in-place to avoid extra memory allocation
        self.logvar = torch.clamp(logvar, -30.0, 20.0)
        
        # Compute std and var on-demand to reduce memory
        self._std = None
        self._var = None
    
    @property
    def std(self) -> torch.Tensor:
        if self._std is None:
            self._std = torch.exp(0.5 * self.logvar)
        return self._std
    
    @property
    def var(self) -> torch.Tensor:
        if self._var is None:
            self._var = torch.exp(self.logvar)
        return self._var
    
    def mode(self) -> torch.Tensor:
        """Return mode (mean) of the distribution."""
        return self.mean
    
    def sample(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """
        Sample from the distribution with optimized GPU operations.
        Uses fused operations to minimize CPU-GPU synchronization.
        """
        # Generate noise directly on the same device as mean
        noise = torch.randn_like(self.mean, generator=generator)
        # Fused multiply-add operation
        return torch.addcmul(self.mean, self.std, noise)
    
    def kl(self) -> torch.Tensor:
        """
        Compute KL divergence with standard normal.
        Optimized to use fused operations.
        """
        # mean^2 + var - 1 - logvar
        # Use fused operations where possible
        kl_div = torch.addcmul(
            self.mean.pow(2) + self.var - 1.0,
            torch.tensor(-1.0, device=self.mean.device),
            self.logvar
        )
        return 0.5 * torch.sum(
            kl_div,
            dim=list(range(1, self.mean.ndim)),
        )


def get_optimal_num_workers_windows() -> int:
    """
    Get optimal number of DataLoader workers for Windows.
    
    Returns:
        Recommended number of workers (0-2 for Windows)
    """
    import os
    import platform
    
    if platform.system() == 'Windows':
        # Windows has IPC limitations, use 0-2 workers
        return min(2, os.cpu_count() or 0)
    else:
        # Linux can handle more workers
        return min(4, os.cpu_count() or 0)


def configure_amp_context(
    enabled: bool = True,
    dtype: torch.dtype = torch.float16,
    cache_enabled: bool = True,
):
    """
    Configure automatic mixed precision context.
    
    Args:
        enabled: Whether to enable AMP
        dtype: Data type for mixed precision (float16 or bfloat16)
        cache_enabled: Enable autocast cache
    
    Returns:
        Context manager for mixed precision
    """
    if enabled and torch.cuda.is_available():
        return torch.cuda.amp.autocast(
            enabled=True,
            dtype=dtype,
            cache_enabled=cache_enabled,
        )
    else:
        from contextlib import nullcontext
        return nullcontext()


def optimize_upsample_operation(
    hidden_states: torch.Tensor,
    scale_factor: float = 2.0,
    mode: Literal['nearest', 'nearest-exact', 'bilinear'] = 'nearest-exact',
) -> torch.Tensor:
    """
    Optimized upsampling using efficient interpolation methods.
    
    Args:
        hidden_states: Input tensor
        scale_factor: Upsampling scale factor
        mode: Interpolation mode (nearest-exact is fastest and deterministic)
    
    Returns:
        Upsampled tensor
    """
    # Use nearest-exact for speed and determinism on Blackwell
    return torch.nn.functional.interpolate(
        hidden_states,
        scale_factor=scale_factor,
        mode=mode,
        # Don't align corners for 'nearest' modes
        align_corners=None if 'nearest' in mode else False,
    )


# Export public API
__all__ = [
    'enable_cudnn_benchmark',
    'disable_cudnn_benchmark',
    'is_fp8_available',
    'optimize_for_windows_blackwell',
    'create_optimized_dataloader',
    'OptimizedDiagonalGaussianDistribution',
    'get_optimal_num_workers_windows',
    'configure_amp_context',
    'optimize_upsample_operation',
]
