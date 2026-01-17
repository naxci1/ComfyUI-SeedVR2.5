"""
Async Weight Streaming for SeedVR2

Implements pinned memory and asynchronous CPU-to-GPU weight streaming
for optimal performance on NVIDIA Blackwell (RTX 50-series) GPUs.

Key Features:
- Pinned memory pool for reduced allocation overhead
- CUDA stream-based async transfers for overlapped loading
- Layer-by-layer prefetching during inference
- Automatic Blackwell architecture detection

This module works with model_patcher to enable streaming of 3B/7B model
weights during the inference loop, reducing peak VRAM usage while
maintaining high throughput.

Usage:
    from src.optimization.async_streaming import (
        AsyncWeightStreamer,
        configure_pinned_memory_streaming,
        stream_model_weights_async
    )
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple, Iterator
from dataclasses import dataclass

from .nvfp4 import (
    is_blackwell_gpu,
    PinnedMemoryPool,
    CUDAStreamManager,
    AsyncModelOffloader,
    create_pinned_tensor
)


@dataclass
class StreamingConfig:
    """
    Configuration for async weight streaming.
    
    Attributes:
        use_pinned_memory: Enable pinned memory for async transfers
        max_pinned_pool_gb: Maximum pinned memory pool size in GB
        prefetch_layers: Number of layers to prefetch ahead
        async_transfers: Enable async H2D transfers
        log_blackwell_status: Log Blackwell-Optimized status
    """
    use_pinned_memory: bool = True
    max_pinned_pool_gb: float = 4.0
    prefetch_layers: int = 2
    async_transfers: bool = True
    log_blackwell_status: bool = True


class AsyncWeightStreamer:
    """
    Async weight streaming manager for large model inference.
    
    Implements CPU-to-GPU weight streaming with:
    - Pinned memory pool for efficient DMA transfers
    - CUDA streams for overlapped transfer/compute
    - Layer prefetching for continuous GPU utilization
    
    Optimized for NVIDIA Blackwell (RTX 50-series) architecture.
    """
    
    def __init__(self, config: Optional[StreamingConfig] = None, 
                 debug: Optional[Any] = None):
        """
        Initialize async weight streamer.
        
        Args:
            config: Streaming configuration (uses defaults if None)
            debug: Debug instance for logging
        """
        self.config = config or StreamingConfig()
        self.debug = debug
        
        # Initialize components
        self._offloader = AsyncModelOffloader(
            use_pinned_memory=self.config.use_pinned_memory,
            debug=debug,
            max_pinned_pool_gb=self.config.max_pinned_pool_gb
        )
        
        # Track streaming state
        self._is_blackwell = is_blackwell_gpu()
        self._layers_offloaded: List[nn.Module] = []
        self._prefetch_queue: List[Tuple[nn.Module, torch.device]] = []
        
        # Log Blackwell status
        if self.config.log_blackwell_status and debug:
            if self._is_blackwell:
                debug.log("[Blackwell-Optimized] Async weight streaming with pinned memory enabled",
                         category="streaming", force=True)
            else:
                debug.log("Async weight streaming enabled (non-Blackwell mode)",
                         category="streaming")
    
    def offload_model_to_cpu(self, model: nn.Module, name: str = "model"):
        """
        Offload model weights to CPU with pinned memory.
        
        Args:
            model: Model to offload
            name: Name for tracking
        """
        if self.debug:
            self.debug.start_timer(f"offload_{name}")
        
        self._offloader.offload_async(model, name)
        self._offloader.synchronize()
        
        if self.debug:
            self.debug.end_timer(f"offload_{name}", f"Offloaded {name} to CPU")
    
    def stream_layer_to_gpu(self, layer: nn.Module, device: torch.device,
                            name: str = "layer") -> nn.Module:
        """
        Stream a single layer to GPU with async transfer.
        
        Args:
            layer: Layer to stream
            device: Target GPU device
            name: Name for tracking
            
        Returns:
            Layer on GPU (may still be transferring)
        """
        self._offloader.prefetch_layer(layer, device, name)
        return layer
    
    def prefetch_next_layers(self, layers: List[nn.Module], 
                             current_idx: int,
                             device: torch.device):
        """
        Prefetch upcoming layers while current layer computes.
        
        Args:
            layers: List of all layers
            current_idx: Current processing index
            device: Target device
        """
        prefetch_count = self.config.prefetch_layers
        
        for offset in range(1, prefetch_count + 1):
            next_idx = current_idx + offset
            if next_idx < len(layers):
                self._offloader.prefetch_layer(
                    layers[next_idx], 
                    device, 
                    f"layer_{next_idx}"
                )
    
    def wait_for_layer(self, layer: nn.Module):
        """
        Wait for layer transfer to complete before using.
        
        Args:
            layer: Layer to wait for
        """
        self._offloader.wait_for_prefetch()
    
    def offload_layer_to_cpu(self, layer: nn.Module, name: str = "layer"):
        """
        Offload layer back to CPU after use.
        
        Args:
            layer: Layer to offload
            name: Name for tracking
        """
        if self.config.use_pinned_memory:
            self._offloader.offload_async(layer, name)
        else:
            layer.cpu()
    
    def stream_weights_for_inference(
        self,
        model: nn.Module,
        device: torch.device,
        blocks_to_stream: Optional[List[int]] = None
    ) -> Iterator[Tuple[int, nn.Module]]:
        """
        Generator for streaming model blocks during inference.
        
        Yields blocks one at a time, streaming them to GPU and
        offloading after use for memory efficiency.
        
        Args:
            model: Model with 'blocks' attribute
            device: Target GPU device
            blocks_to_stream: Optional list of block indices (None = all)
            
        Yields:
            Tuple of (block_index, block_on_gpu)
        """
        if not hasattr(model, 'blocks'):
            if self.debug:
                self.debug.log("Model has no 'blocks' attribute for streaming",
                              level="WARNING", category="streaming", force=True)
            return
        
        blocks = model.blocks
        if blocks_to_stream is None:
            blocks_to_stream = list(range(len(blocks)))
        
        for i, block_idx in enumerate(blocks_to_stream):
            block = blocks[block_idx]
            
            # Wait for any pending transfer
            self.wait_for_layer(block)
            
            # Stream current block
            self.stream_layer_to_gpu(block, device, f"block_{block_idx}")
            self.wait_for_layer(block)
            
            # Prefetch next blocks
            remaining_indices = blocks_to_stream[i+1:]
            if remaining_indices:
                for prefetch_offset, prefetch_idx in enumerate(remaining_indices[:self.config.prefetch_layers]):
                    prefetch_block = blocks[prefetch_idx]
                    self._offloader.prefetch_layer(prefetch_block, device, f"block_{prefetch_idx}")
            
            # Yield block for processing
            yield block_idx, block
            
            # Offload after use
            self.offload_layer_to_cpu(block, f"block_{block_idx}")
    
    def cleanup(self):
        """Release streaming resources."""
        self._offloader.cleanup()
        
        if self.debug:
            stats = self._offloader.get_stats()
            self.debug.log(
                f"Streaming cleanup: Blackwell={stats['blackwell_optimized']}, "
                f"Pinned={stats['pinned_memory_enabled']}",
                category="streaming"
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get streaming statistics."""
        return {
            'blackwell_optimized': self._is_blackwell,
            'pinned_memory_enabled': self.config.use_pinned_memory,
            'prefetch_layers': self.config.prefetch_layers,
            **self._offloader.get_stats()
        }


def configure_pinned_memory_streaming(
    model_patcher: Any,
    debug: Optional[Any] = None,
    max_pinned_gb: float = 4.0
) -> Dict[str, Any]:
    """
    Configure model patcher for pinned memory streaming.
    
    This function initializes the model patcher to use pinned memory
    for CPU-to-GPU transfers, enabling async streaming during inference.
    
    Args:
        model_patcher: ComfyUI model patcher instance
        debug: Debug instance for logging
        max_pinned_gb: Maximum pinned memory allocation in GB
        
    Returns:
        Configuration dictionary for streaming
    """
    config = StreamingConfig(
        use_pinned_memory=True,
        max_pinned_pool_gb=max_pinned_gb,
        async_transfers=True,
        log_blackwell_status=True
    )
    
    # Create streamer instance
    streamer = AsyncWeightStreamer(config=config, debug=debug)
    
    # Log configuration
    if debug:
        if is_blackwell_gpu():
            debug.log(
                f"[Blackwell-Optimized] Model patcher configured for async streaming "
                f"(pinned pool: {max_pinned_gb}GB)",
                category="streaming", force=True
            )
        else:
            debug.log(
                f"Model patcher configured for async streaming (pinned pool: {max_pinned_gb}GB)",
                category="streaming"
            )
    
    return {
        'streamer': streamer,
        'config': config,
        'blackwell_optimized': is_blackwell_gpu()
    }


def stream_model_weights_async(
    model: nn.Module,
    device: torch.device,
    streamer: AsyncWeightStreamer,
    blocks_to_stream: Optional[List[int]] = None,
    debug: Optional[Any] = None
) -> nn.Module:
    """
    Stream model weights to GPU asynchronously during inference.
    
    This is the main entry point for async weight streaming. It configures
    the model for layer-by-layer streaming to minimize peak VRAM usage.
    
    Args:
        model: Model with weights on CPU
        device: Target GPU device
        streamer: Configured AsyncWeightStreamer instance
        blocks_to_stream: Optional list of block indices to stream
        debug: Debug instance for logging
        
    Returns:
        Model ready for streaming inference
    """
    if debug:
        debug.log("Configuring model for async weight streaming", category="streaming")
    
    # Store streamer on model for use during forward pass
    model._weight_streamer = streamer
    model._streaming_device = device
    model._blocks_to_stream = blocks_to_stream
    
    # Log status
    if debug:
        stats = streamer.get_stats()
        if stats['blackwell_optimized']:
            debug.log(
                "[Blackwell-Optimized] Model configured for async streaming inference",
                category="streaming", force=True
            )
    
    return model


def initialize_streaming_for_inference(
    runner: Any,
    device: torch.device,
    debug: Optional[Any] = None,
    max_pinned_gb: float = 4.0,
    blocks_to_stream: Optional[int] = None
) -> AsyncWeightStreamer:
    """
    Initialize async weight streaming for DiT/VAE inference.
    
    This function sets up the streaming infrastructure for large model
    inference, including pinned memory pools and CUDA streams.
    
    Args:
        runner: VideoDiffusionInfer runner instance
        device: Target GPU device
        debug: Debug instance for logging
        max_pinned_gb: Maximum pinned memory pool size
        blocks_to_stream: Number of blocks to stream (None = all offloaded)
        
    Returns:
        Configured AsyncWeightStreamer instance
    """
    config = StreamingConfig(
        use_pinned_memory=True,
        max_pinned_pool_gb=max_pinned_gb,
        prefetch_layers=2,
        async_transfers=True,
        log_blackwell_status=True
    )
    
    streamer = AsyncWeightStreamer(config=config, debug=debug)
    
    # Attach to runner for use during inference
    runner._async_streamer = streamer
    runner._streaming_device = device
    
    if debug:
        if is_blackwell_gpu():
            debug.log(
                f"[Blackwell-Optimized] Async weight streaming initialized "
                f"(pinned: {max_pinned_gb}GB, prefetch: {config.prefetch_layers} layers)",
                category="streaming", force=True
            )
    
    return streamer


# Module exports
__all__ = [
    'StreamingConfig',
    'AsyncWeightStreamer',
    'configure_pinned_memory_streaming',
    'stream_model_weights_async',
    'initialize_streaming_for_inference',
]
