"""
SeedVR2 Optimization Module

Contains memory optimization, performance utilities, and hardware acceleration support.
"""

from .memory_manager import (
    get_device_list,
    get_basic_vram_info,
    get_vram_usage,
    clear_memory,
    is_cuda_available,
    is_mps_available,
    get_gpu_backend,
)

from .compatibility import (
    GGUF_AVAILABLE,
    TRITON_AVAILABLE,
    FLASH_ATTN_AVAILABLE,
    SAGE_ATTN_AVAILABLE,
    NVFP4_AVAILABLE,
    BLACKWELL_GPU_DETECTED,
    COMPUTE_DTYPE,
    BFLOAT16_SUPPORTED,
    validate_attention_mode,
)

from .nvfp4 import (
    NVFP4Config,
    NVFP4Tensor,
    NvFP4LinearLayer,
    AsyncModelOffloader,
    is_nvfp4_supported,
    is_blackwell_gpu,
    get_nvfp4_status,
    should_preserve_precision,
    load_nvfp4_weights,
    is_nvfp4_checkpoint,
)

__all__ = [
    # Memory manager
    'get_device_list',
    'get_basic_vram_info',
    'get_vram_usage',
    'clear_memory',
    'is_cuda_available',
    'is_mps_available',
    'get_gpu_backend',
    # Compatibility
    'GGUF_AVAILABLE',
    'TRITON_AVAILABLE',
    'FLASH_ATTN_AVAILABLE',
    'SAGE_ATTN_AVAILABLE',
    'NVFP4_AVAILABLE',
    'BLACKWELL_GPU_DETECTED',
    'COMPUTE_DTYPE',
    'BFLOAT16_SUPPORTED',
    'validate_attention_mode',
    # NVFP4 / Blackwell
    'NVFP4Config',
    'NVFP4Tensor',
    'NvFP4LinearLayer',
    'AsyncModelOffloader',
    'is_nvfp4_supported',
    'is_blackwell_gpu',
    'get_nvfp4_status',
    'should_preserve_precision',
    'load_nvfp4_weights',
    'is_nvfp4_checkpoint',
]
