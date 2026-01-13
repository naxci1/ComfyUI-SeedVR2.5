"""
SeedVR2 Optimization Module

This module provides GPU memory management, CUDA stream optimization,
and torch.compile support for high-performance video upscaling.

Key Components:
- memory_manager: Unified memory management and model offloading
- nvfp4: Native FP4 support for Blackwell (RTX 50-series) GPUs
- async_transfer: Async data loading with pinned memory
- windows_triton_compat: Windows torch.compile compatibility
- blockswap: Layer-by-layer GPU memory optimization
- compatibility: Flash/Sage Attention and FP8 compatibility layers

Usage:
    from src.optimization import (
        clear_memory,
        get_gpu_backend,
        manage_tensor,
        AsyncDataLoader,
        safe_compile_model,
    )
"""

# Core memory management
from .memory_manager import (
    clear_memory,
    get_gpu_backend,
    get_device_list,
    get_vram_usage,
    is_cuda_available,
    is_mps_available,
    manage_tensor,
    manage_tensor_async,
    manage_model_device,
    complete_cleanup,
)

# Async transfer utilities
from .async_transfer import (
    PinnedTensorAllocator,
    StreamedTransferManager,
    AsyncDataLoader,
    prefetch_next_batch,
    create_async_transfer_context,
    cleanup_async_context,
)

# Windows torch.compile support
from .windows_triton_compat import (
    is_windows,
    ensure_windows_triton_compat,
    safe_compile_model,
    get_compilation_status,
)

__all__ = [
    # Memory management
    'clear_memory',
    'get_gpu_backend',
    'get_device_list',
    'get_vram_usage',
    'is_cuda_available',
    'is_mps_available',
    'manage_tensor',
    'manage_tensor_async',
    'manage_model_device',
    'complete_cleanup',
    # Async transfer
    'PinnedTensorAllocator',
    'StreamedTransferManager',
    'AsyncDataLoader',
    'prefetch_next_batch',
    'create_async_transfer_context',
    'cleanup_async_context',
    # torch.compile
    'is_windows',
    'ensure_windows_triton_compat',
    'safe_compile_model',
    'get_compilation_status',
]
