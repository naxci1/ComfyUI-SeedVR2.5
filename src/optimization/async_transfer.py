"""
Async Transfer Optimization Module for SeedVR2

This module provides advanced async data transfer utilities following the 
"ComfyUI Optimization" standard for maximum GPU utilization.

Key Features:
- Pinned memory allocation for efficient DMA transfers
- CUDA stream management for overlapped compute and data movement
- Prefetch utilities for next-batch loading during current-batch computation
- Automatic Blackwell (RTX 50xx) optimization detection

Technical Background:
- Pinned (page-locked) memory enables Direct Memory Access (DMA) transfers
- Non-blocking transfers allow overlap between CPU and GPU operations
- CUDA streams enable concurrent execution of independent operations
- Prefetching hides data transfer latency behind compute latency

Usage:
    from src.optimization.async_transfer import (
        AsyncDataLoader,
        PinnedTensorAllocator,
        StreamedTransferManager,
        prefetch_next_batch
    )

Author: SeedVR2 Team
"""

import torch
import torch.nn as nn
from typing import Iterator, List, Optional, Tuple, Dict, Any, Union, Generator, Protocol, runtime_checkable
from collections import OrderedDict
import threading
import time


@runtime_checkable
class DebugLogger(Protocol):
    """Protocol defining the debug logger interface"""
    def log(self, message: str, level: str = "INFO", category: str = "general", 
            force: bool = False, indent_level: int = 0) -> None: ...


class PinnedTensorAllocator:
    """
    Efficient pinned memory tensor allocator with buffer reuse.
    
    Allocates tensors in page-locked (pinned) memory for efficient
    CPU-GPU transfers via DMA (Direct Memory Access).
    
    Benefits of pinned memory:
    - Enables non-blocking async transfers
    - Higher bandwidth utilization on PCIe bus
    - Required for CUDA stream-based overlapped transfers
    
    This allocator maintains a pool of buffers to reduce allocation overhead.
    """
    
    def __init__(
        self,
        max_pool_mb: float = 2048.0,
        enable_pooling: bool = True,
        debug: Optional[DebugLogger] = None
    ):
        """
        Initialize pinned tensor allocator.
        
        Args:
            max_pool_mb: Maximum pool size in MB
            enable_pooling: Whether to reuse buffers
            debug: Debug instance for logging
        """
        self._enabled = torch.cuda.is_available()
        self._pool: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._max_pool_bytes = int(max_pool_mb * 1024 * 1024)
        self._current_pool_bytes = 0
        self._enable_pooling = enable_pooling
        self._debug = debug
        self._stats = {
            'allocations': 0,
            'pool_hits': 0,
            'pool_misses': 0,
            'bytes_allocated': 0
        }
        self._lock = threading.Lock()
    
    def _make_key(self, shape: Tuple[int, ...], dtype: torch.dtype) -> str:
        """Create lookup key from shape and dtype"""
        return f"{shape}_{dtype}"
    
    def _get_tensor_bytes(self, tensor: torch.Tensor) -> int:
        """Calculate tensor memory footprint"""
        return tensor.numel() * tensor.element_size()
    
    def allocate(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        fill_value: Optional[float] = None
    ) -> torch.Tensor:
        """
        Allocate a pinned memory tensor.
        
        Attempts to reuse pooled buffers when possible.
        Falls back to regular CPU tensor if pinned allocation fails.
        
        Args:
            shape: Tensor shape
            dtype: Tensor dtype
            fill_value: Optional value to fill tensor with
            
        Returns:
            Pinned memory tensor (or regular CPU tensor on failure)
        """
        if not self._enabled:
            if fill_value is not None:
                return torch.full(shape, fill_value, dtype=dtype)
            return torch.empty(shape, dtype=dtype)
        
        key = self._make_key(shape, dtype)
        
        with self._lock:
            # Check pool for reusable buffer
            if self._enable_pooling and key in self._pool:
                tensor = self._pool.pop(key)
                self._current_pool_bytes -= self._get_tensor_bytes(tensor)
                self._stats['pool_hits'] += 1
                
                if fill_value is not None:
                    tensor.fill_(fill_value)
                return tensor
            
            self._stats['pool_misses'] += 1
        
        # Allocate new pinned tensor
        try:
            if fill_value is not None:
                tensor = torch.full(shape, fill_value, dtype=dtype, pin_memory=True)
            else:
                tensor = torch.empty(shape, dtype=dtype, pin_memory=True)
            
            self._stats['allocations'] += 1
            self._stats['bytes_allocated'] += self._get_tensor_bytes(tensor)
            return tensor
            
        except RuntimeError as e:
            # Pinned allocation failed (OOM or limit reached)
            if self._debug:
                self._debug.log(
                    f"Pinned memory allocation failed: {e}. Using regular CPU tensor.",
                    level="WARNING", category="memory"
                )
            
            if fill_value is not None:
                return torch.full(shape, fill_value, dtype=dtype)
            return torch.empty(shape, dtype=dtype)
    
    def release(self, tensor: torch.Tensor) -> None:
        """
        Return tensor to pool for reuse.
        
        Args:
            tensor: Pinned tensor to return to pool
        """
        if not self._enable_pooling or not tensor.is_pinned():
            return
        
        key = self._make_key(tuple(tensor.shape), tensor.dtype)
        tensor_bytes = self._get_tensor_bytes(tensor)
        
        with self._lock:
            # Check if we have room in pool
            if self._current_pool_bytes + tensor_bytes <= self._max_pool_bytes:
                self._pool[key] = tensor
                self._current_pool_bytes += tensor_bytes
    
    def copy_to_pinned(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Copy tensor to pinned memory.
        
        If tensor is already pinned, returns it directly.
        Otherwise allocates pinned buffer and copies data.
        
        Args:
            tensor: Source tensor
            
        Returns:
            Tensor in pinned memory
        """
        if tensor.is_pinned():
            return tensor
        
        if tensor.device.type != 'cpu':
            tensor = tensor.cpu()
        
        pinned = self.allocate(tuple(tensor.shape), tensor.dtype)
        pinned.copy_(tensor)
        return pinned
    
    def get_stats(self) -> Dict[str, Any]:
        """Get allocation statistics"""
        with self._lock:
            stats = self._stats.copy()
            stats['pool_size_mb'] = self._current_pool_bytes / (1024 * 1024)
            stats['pool_max_mb'] = self._max_pool_bytes / (1024 * 1024)
            stats['pool_buffers'] = len(self._pool)
            if stats['pool_hits'] + stats['pool_misses'] > 0:
                stats['hit_rate'] = stats['pool_hits'] / (stats['pool_hits'] + stats['pool_misses'])
            else:
                stats['hit_rate'] = 0.0
            return stats
    
    def clear(self) -> None:
        """Clear all pooled buffers"""
        with self._lock:
            self._pool.clear()
            self._current_pool_bytes = 0


class StreamedTransferManager:
    """
    CUDA stream manager for overlapped compute and data transfer.
    
    Provides dedicated streams for:
    - Host-to-Device (H2D) transfers - loading next batch while computing
    - Device-to-Host (D2H) transfers - offloading results while computing next
    - Compute operations - main inference stream
    - Prefetch operations - high-priority stream for layer prefetching
    
    This enables the "double-buffering" pattern:
    While GPU computes on batch N, CPU prepares batch N+1 in pinned memory,
    and GPU loads batch N+1 via DMA on a separate stream.
    """
    
    def __init__(
        self,
        device: Optional[torch.device] = None,
        debug: Optional[DebugLogger] = None
    ):
        """
        Initialize stream manager.
        
        Args:
            device: CUDA device (defaults to current device)
            debug: Debug instance for logging
        """
        self._enabled = torch.cuda.is_available()
        self._debug = debug
        
        if device is None:
            device = torch.device('cuda:0' if self._enabled else 'cpu')
        self._device = device
        
        if self._enabled and device.type == 'cuda':
            # Create dedicated streams
            self._h2d_stream = torch.cuda.Stream(device=device)
            self._d2h_stream = torch.cuda.Stream(device=device)
            self._compute_stream = torch.cuda.Stream(device=device)
            
            # Events for synchronization
            self._h2d_events: Dict[str, torch.cuda.Event] = {}
            self._compute_events: Dict[str, torch.cuda.Event] = {}
            
            # High-priority stream for prefetching (if available)
            try:
                self._prefetch_stream = torch.cuda.Stream(device=device, priority=-1)
            except Exception:
                self._prefetch_stream = self._h2d_stream
        else:
            self._h2d_stream = None
            self._d2h_stream = None
            self._compute_stream = None
            self._prefetch_stream = None
            self._h2d_events = {}
            self._compute_events = {}
    
    @property
    def h2d_stream(self) -> Optional[torch.cuda.Stream]:
        """Host-to-Device transfer stream"""
        return self._h2d_stream
    
    @property
    def d2h_stream(self) -> Optional[torch.cuda.Stream]:
        """Device-to-Host transfer stream"""
        return self._d2h_stream
    
    @property
    def compute_stream(self) -> Optional[torch.cuda.Stream]:
        """Compute stream"""
        return self._compute_stream
    
    @property
    def prefetch_stream(self) -> Optional[torch.cuda.Stream]:
        """High-priority prefetch stream"""
        return self._prefetch_stream
    
    def transfer_to_device_async(
        self,
        tensor: torch.Tensor,
        name: str = "tensor",
        record_event: bool = True
    ) -> torch.Tensor:
        """
        Asynchronously transfer tensor from CPU to GPU.
        
        Uses dedicated H2D stream for overlap with compute.
        Tensor should be in pinned memory for best performance.
        
        Args:
            tensor: CPU tensor to transfer (ideally pinned)
            name: Identifier for synchronization
            record_event: Whether to record event for later sync
            
        Returns:
            GPU tensor (transfer may still be in progress)
        """
        if not self._enabled or self._device.type != 'cuda':
            return tensor.to(self._device)
        
        if tensor.device == self._device:
            return tensor
        
        with torch.cuda.stream(self._h2d_stream):
            result = tensor.to(self._device, non_blocking=True)
            
            if record_event:
                event = torch.cuda.Event()
                event.record(self._h2d_stream)
                self._h2d_events[name] = event
        
        return result
    
    def transfer_to_host_async(
        self,
        tensor: torch.Tensor,
        name: str = "tensor"
    ) -> torch.Tensor:
        """
        Asynchronously transfer tensor from GPU to CPU.
        
        Uses dedicated D2H stream for overlap with compute.
        
        Args:
            tensor: GPU tensor to transfer
            name: Identifier for tracking
            
        Returns:
            CPU tensor (transfer may still be in progress)
        """
        if not self._enabled or tensor.device.type != 'cuda':
            return tensor.cpu()
        
        with torch.cuda.stream(self._d2h_stream):
            result = tensor.cpu()
        
        return result
    
    def wait_for_transfer(self, name: str) -> None:
        """
        Wait for specific transfer to complete.
        
        Args:
            name: Transfer identifier
        """
        if name in self._h2d_events:
            self._h2d_events[name].synchronize()
            del self._h2d_events[name]
    
    def synchronize_transfers(self) -> None:
        """Wait for all pending transfers to complete"""
        if self._h2d_stream:
            self._h2d_stream.synchronize()
        if self._d2h_stream:
            self._d2h_stream.synchronize()
        
        self._h2d_events.clear()
    
    def synchronize_all(self) -> None:
        """Wait for all stream operations to complete"""
        if self._enabled:
            if self._h2d_stream:
                self._h2d_stream.synchronize()
            if self._d2h_stream:
                self._d2h_stream.synchronize()
            if self._compute_stream:
                self._compute_stream.synchronize()
            if self._prefetch_stream and self._prefetch_stream != self._h2d_stream:
                self._prefetch_stream.synchronize()
        
        self._h2d_events.clear()
        self._compute_events.clear()


class AsyncDataLoader:
    """
    Async data loader with prefetching for optimal GPU utilization.
    
    Implements the "double-buffering" pattern:
    - Batch N is being processed on GPU (compute stream)
    - Batch N+1 is being loaded to GPU via DMA (H2D stream)
    - Batch N+2 is being prepared in pinned memory (CPU)
    
    This eliminates GPU idle time from data loading.
    
    Usage:
        loader = AsyncDataLoader(data_iterator, device)
        for batch in loader:
            result = model(batch)
            # During this compute, next batch is already loading
    """
    
    def __init__(
        self,
        data_iterator: Iterator[torch.Tensor],
        device: torch.device,
        prefetch_count: int = 2,
        use_pinned_memory: bool = True,
        debug: Optional[DebugLogger] = None
    ):
        """
        Initialize async data loader.
        
        Args:
            data_iterator: Source data iterator
            device: Target device for data
            prefetch_count: Number of batches to prefetch
            use_pinned_memory: Whether to use pinned memory for CPU tensors
            debug: Debug instance for logging
        """
        self._iterator = data_iterator
        self._device = device
        self._prefetch_count = prefetch_count
        self._use_pinned = use_pinned_memory and torch.cuda.is_available()
        self._debug = debug
        
        # Initialize components
        self._allocator = PinnedTensorAllocator(debug=debug) if self._use_pinned else None
        self._stream_manager = StreamedTransferManager(device=device, debug=debug)
        
        # Prefetch queue
        self._prefetch_queue: List[Tuple[torch.Tensor, str]] = []
        self._batch_idx = 0
        self._exhausted = False
    
    def _prefetch_next(self) -> bool:
        """
        Prefetch next batch into the queue.
        
        Returns:
            True if batch was prefetched, False if iterator exhausted
        """
        if self._exhausted:
            return False
        
        try:
            batch = next(self._iterator)
            batch_name = f"batch_{self._batch_idx}"
            
            # Copy to pinned memory if available
            if self._allocator and not batch.is_pinned():
                batch = self._allocator.copy_to_pinned(batch.cpu())
            elif batch.device.type != 'cpu':
                batch = batch.cpu()
            
            # Start async transfer to GPU
            gpu_batch = self._stream_manager.transfer_to_device_async(
                batch, name=batch_name, record_event=True
            )
            
            self._prefetch_queue.append((gpu_batch, batch_name))
            self._batch_idx += 1
            return True
            
        except StopIteration:
            self._exhausted = True
            return False
    
    def __iter__(self) -> Generator[torch.Tensor, None, None]:
        """
        Iterate over batches with prefetching.
        
        Yields:
            GPU tensors ready for computation
        """
        # Initial prefetch
        for _ in range(self._prefetch_count):
            if not self._prefetch_next():
                break
        
        while self._prefetch_queue:
            # Get next ready batch
            gpu_batch, batch_name = self._prefetch_queue.pop(0)
            
            # Wait for this batch's transfer to complete
            self._stream_manager.wait_for_transfer(batch_name)
            
            # Start prefetching next batch
            self._prefetch_next()
            
            yield gpu_batch
    
    def cleanup(self) -> None:
        """Release resources"""
        self._stream_manager.synchronize_all()
        if self._allocator:
            self._allocator.clear()


def prefetch_next_batch(
    current_batch: torch.Tensor,
    next_batch_cpu: Optional[torch.Tensor],
    stream_manager: StreamedTransferManager,
    allocator: Optional[PinnedTensorAllocator] = None
) -> Optional[torch.Tensor]:
    """
    Prefetch next batch while current batch is being processed.
    
    This is the core pattern for overlapped data loading:
    1. Current batch starts compute on GPU
    2. Next batch starts transfer to GPU on separate stream
    3. By the time compute finishes, next batch is ready
    
    Args:
        current_batch: Current batch being processed (for reference)
        next_batch_cpu: Next batch on CPU to prefetch
        stream_manager: Stream manager for async transfers
        allocator: Optional pinned memory allocator
        
    Returns:
        GPU tensor for next batch (transfer in progress)
    """
    if next_batch_cpu is None:
        return None
    
    # Ensure next batch is in pinned memory
    if allocator and not next_batch_cpu.is_pinned():
        next_batch_cpu = allocator.copy_to_pinned(next_batch_cpu)
    
    # Start async transfer on H2D stream
    return stream_manager.transfer_to_device_async(
        next_batch_cpu, name="next_batch", record_event=True
    )


def create_async_transfer_context(
    device: Optional[torch.device] = None,
    max_pool_mb: float = 2048.0,
    debug: Optional[DebugLogger] = None
) -> Dict[str, Any]:
    """
    Create a context with all async transfer components.
    
    Convenience function to set up all components needed for
    async data loading in a processing pipeline.
    
    Args:
        device: Target CUDA device
        max_pool_mb: Maximum pinned memory pool size
        debug: Debug instance
        
    Returns:
        Dictionary with 'allocator', 'stream_manager', 'device' keys
    """
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    return {
        'device': device,
        'allocator': PinnedTensorAllocator(max_pool_mb=max_pool_mb, debug=debug),
        'stream_manager': StreamedTransferManager(device=device, debug=debug)
    }


def cleanup_async_context(context: Dict[str, Any]) -> None:
    """
    Cleanup async transfer context.
    
    Args:
        context: Context dict from create_async_transfer_context
    """
    if 'stream_manager' in context:
        context['stream_manager'].synchronize_all()
    if 'allocator' in context:
        context['allocator'].clear()


# Module exports
__all__ = [
    'PinnedTensorAllocator',
    'StreamedTransferManager',
    'AsyncDataLoader',
    'prefetch_next_batch',
    'create_async_transfer_context',
    'cleanup_async_context',
]
