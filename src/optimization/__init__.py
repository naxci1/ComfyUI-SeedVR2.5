"""
SeedVR2 Optimization Module

NVIDIA Blackwell (RTX 50-series) optimizations and performance enhancements.
"""

from .nvfp4 import (
    is_blackwell_gpu,
    is_nvfp4_supported,
    get_nvfp4_status,
    log_blackwell_status,
    replace_linear_with_nvfp4,
    NvFP4LinearLayer,
    NVFP4Config,
    AsyncModelOffloader,
    PinnedMemoryPool,
)

from .async_streaming import (
    AsyncWeightStreamer,
    StreamingConfig,
    configure_pinned_memory_streaming,
    stream_model_weights_async,
    initialize_streaming_for_inference,
)

__all__ = [
    # NVFP4 Blackwell acceleration
    'is_blackwell_gpu',
    'is_nvfp4_supported',
    'get_nvfp4_status',
    'log_blackwell_status',
    'replace_linear_with_nvfp4',
    'NvFP4LinearLayer',
    'NVFP4Config',
    'AsyncModelOffloader',
    'PinnedMemoryPool',
    # Async weight streaming
    'AsyncWeightStreamer',
    'StreamingConfig',
    'configure_pinned_memory_streaming',
    'stream_model_weights_async',
    'initialize_streaming_for_inference',
]
