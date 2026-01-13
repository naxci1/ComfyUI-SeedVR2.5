#!/usr/bin/env python3
"""
SeedVR2 Optimization Benchmark Script

This script validates and benchmarks the NVIDIA GPU optimizations for SeedVR2,
including async offloading, pinned memory, CUDA streams, and torch.compile.

Metrics Measured:
- Iterations Per Second (it/s) - throughput performance
- Peak VRAM Usage (GB) - maximum GPU memory allocated
- Latency (ms) - time per operation
- CPU-GPU Transfer Bandwidth (GB/s)

Comparison Modes:
- Eager Mode: Standard PyTorch execution
- Compiled + Async Mode: Full optimizations enabled

Usage:
    python scripts/optimization_benchmark.py
    python scripts/optimization_benchmark.py --quick          # Fast benchmark
    python scripts/optimization_benchmark.py --full           # Comprehensive benchmark
    python scripts/optimization_benchmark.py --compare        # Side-by-side comparison

Requirements:
    - NVIDIA GPU with CUDA support
    - PyTorch 2.0+ (2.6+ recommended for torch.compile)
    - Optional: Triton for inductor backend

Author: SeedVR2 Team
"""

import os
import sys
import time
import argparse
import gc
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

# Add parent directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

import torch
import torch.nn as nn


# ============================================================================
# Constants for unit conversions
# ============================================================================
MS_TO_SECONDS = 1000.0  # Convert milliseconds to seconds (divide by this)
MB_TO_GB = 1024.0  # Convert MB to GB (divide by this)
BYTES_PER_FLOAT32 = 4  # Size of float32 in bytes
BYTES_TO_GB = 1024 ** 3  # Convert bytes to GB (divide by this)
BYTES_TO_MB = 1024 ** 2  # Convert bytes to MB (divide by this)


@dataclass
class BenchmarkResult:
    """Container for benchmark results"""
    name: str
    iterations: int = 0
    total_time_s: float = 0.0
    iterations_per_sec: float = 0.0
    avg_latency_ms: float = 0.0
    min_latency_ms: float = float('inf')
    max_latency_ms: float = 0.0
    peak_vram_gb: float = 0.0
    avg_vram_gb: float = 0.0
    transfer_bandwidth_gbs: float = 0.0
    extra_metrics: Dict[str, Any] = field(default_factory=dict)


def print_header(title: str) -> None:
    """Print formatted section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_subheader(title: str) -> None:
    """Print formatted subsection header"""
    print(f"\n--- {title} ---")


def get_gpu_info() -> Dict[str, Any]:
    """Get GPU information"""
    if not torch.cuda.is_available():
        return {'available': False}
    
    device = torch.device('cuda:0')
    props = torch.cuda.get_device_properties(device)
    compute_cap = torch.cuda.get_device_capability(device)
    
    return {
        'available': True,
        'name': props.name,
        'total_memory_gb': props.total_memory / BYTES_TO_GB,
        'compute_capability': f"{compute_cap[0]}.{compute_cap[1]}",
        'is_blackwell': compute_cap[0] >= 10,
        'is_hopper': compute_cap[0] == 9,
        'is_ampere': compute_cap[0] == 8,
        'cuda_version': torch.version.cuda,
        'torch_version': torch.__version__,
        'cudnn_version': torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
    }


def clear_gpu_memory() -> None:
    """Clear GPU memory and caches"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def get_vram_usage() -> Tuple[float, float]:
    """Get current and peak VRAM usage in GB"""
    if not torch.cuda.is_available():
        return (0.0, 0.0)
    
    current = torch.cuda.memory_allocated() / BYTES_TO_GB
    peak = torch.cuda.max_memory_allocated() / BYTES_TO_GB
    return (current, peak)


# =============================================================================
# Benchmark: Pinned Memory Transfer
# =============================================================================

def benchmark_pinned_memory(
    size_mb: int = 256,
    iterations: int = 10,
    warmup: int = 3
) -> BenchmarkResult:
    """
    Benchmark pinned vs pageable memory transfer speed.
    
    Args:
        size_mb: Size of tensor to transfer (MB)
        iterations: Number of benchmark iterations
        warmup: Number of warmup iterations
    """
    result = BenchmarkResult(name="Pinned Memory Transfer")
    
    if not torch.cuda.is_available():
        result.extra_metrics['status'] = 'CUDA not available'
        return result
    
    tensor_size = (size_mb * int(BYTES_TO_MB)) // BYTES_PER_FLOAT32  # float32 = 4 bytes
    
    # Create tensors
    pageable_tensor = torch.randn(tensor_size, dtype=torch.float32)
    pinned_tensor = torch.randn(tensor_size, dtype=torch.float32, pin_memory=True)
    
    # Warmup
    for _ in range(warmup):
        _ = pageable_tensor.to('cuda')
        torch.cuda.synchronize()
        _ = pinned_tensor.to('cuda', non_blocking=True)
        torch.cuda.synchronize()
    
    clear_gpu_memory()
    
    # Benchmark pageable
    pageable_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        gpu_tensor = pageable_tensor.to('cuda')
        torch.cuda.synchronize()
        pageable_times.append((time.perf_counter() - start) * 1000)
        del gpu_tensor
    
    clear_gpu_memory()
    
    # Benchmark pinned
    pinned_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        gpu_tensor = pinned_tensor.to('cuda', non_blocking=True)
        torch.cuda.synchronize()
        pinned_times.append((time.perf_counter() - start) * 1000)
        del gpu_tensor
    
    # Calculate results
    avg_pageable = sum(pageable_times) / len(pageable_times)
    avg_pinned = sum(pinned_times) / len(pinned_times)
    
    result.iterations = iterations
    result.avg_latency_ms = avg_pinned
    result.min_latency_ms = min(pinned_times)
    result.max_latency_ms = max(pinned_times)
    # Calculate bandwidth: size_mb / (time_in_seconds) / MB_TO_GB = GB/s
    time_in_seconds = avg_pinned / MS_TO_SECONDS
    result.transfer_bandwidth_gbs = size_mb / time_in_seconds / MB_TO_GB
    
    result.extra_metrics = {
        'size_mb': size_mb,
        'pageable_avg_ms': avg_pageable,
        'pinned_avg_ms': avg_pinned,
        'speedup': avg_pageable / avg_pinned if avg_pinned > 0 else 0,
        'bandwidth_gbs': result.transfer_bandwidth_gbs,
    }
    
    # Cleanup
    del pageable_tensor, pinned_tensor
    clear_gpu_memory()
    
    return result


# =============================================================================
# Benchmark: CUDA Stream Overlap
# =============================================================================

def benchmark_stream_overlap(
    size_mb: int = 128,
    iterations: int = 10,
    warmup: int = 3
) -> BenchmarkResult:
    """
    Benchmark CUDA stream overlap for concurrent transfers.
    
    Tests:
    - Sequential transfers (no overlap)
    - Parallel transfers (stream overlap)
    """
    result = BenchmarkResult(name="CUDA Stream Overlap")
    
    if not torch.cuda.is_available():
        result.extra_metrics['status'] = 'CUDA not available'
        return result
    
    tensor_size = (size_mb * 1024 * 1024) // 4
    
    # Create streams
    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()
    
    # Create pinned tensors
    tensor1 = torch.randn(tensor_size, dtype=torch.float32, pin_memory=True)
    tensor2 = torch.randn(tensor_size, dtype=torch.float32, pin_memory=True)
    
    # Warmup
    for _ in range(warmup):
        _ = tensor1.to('cuda')
        _ = tensor2.to('cuda')
        torch.cuda.synchronize()
    
    clear_gpu_memory()
    
    # Benchmark sequential
    sequential_times = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        gpu1 = tensor1.to('cuda')
        torch.cuda.synchronize()
        gpu2 = tensor2.to('cuda')
        torch.cuda.synchronize()
        
        sequential_times.append((time.perf_counter() - start) * 1000)
        del gpu1, gpu2
    
    clear_gpu_memory()
    
    # Benchmark parallel (stream overlap)
    parallel_times = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        with torch.cuda.stream(stream1):
            gpu1 = tensor1.to('cuda', non_blocking=True)
        
        with torch.cuda.stream(stream2):
            gpu2 = tensor2.to('cuda', non_blocking=True)
        
        stream1.synchronize()
        stream2.synchronize()
        
        parallel_times.append((time.perf_counter() - start) * 1000)
        del gpu1, gpu2
    
    # Calculate results
    avg_sequential = sum(sequential_times) / len(sequential_times)
    avg_parallel = sum(parallel_times) / len(parallel_times)
    
    result.iterations = iterations
    result.avg_latency_ms = avg_parallel
    result.min_latency_ms = min(parallel_times)
    result.max_latency_ms = max(parallel_times)
    
    result.extra_metrics = {
        'size_mb_each': size_mb,
        'sequential_avg_ms': avg_sequential,
        'parallel_avg_ms': avg_parallel,
        'speedup': avg_sequential / avg_parallel if avg_parallel > 0 else 0,
        'overlap_efficiency': (avg_sequential - avg_parallel) / avg_sequential * 100 if avg_sequential > 0 else 0,
    }
    
    # Cleanup
    del tensor1, tensor2
    clear_gpu_memory()
    
    return result


# =============================================================================
# Benchmark: torch.compile
# =============================================================================

class SimpleMLP(nn.Module):
    """Simple MLP for compile benchmarking"""
    
    def __init__(self, hidden_size: int = 4096, num_layers: int = 4):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.GELU())
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def benchmark_torch_compile(
    batch_size: int = 32,
    hidden_size: int = 4096,
    num_layers: int = 4,
    iterations: int = 50,
    warmup: int = 10
) -> BenchmarkResult:
    """
    Benchmark torch.compile speedup.
    
    Compares:
    - Eager mode: Standard PyTorch execution
    - Compiled mode: torch.compile with inductor backend
    """
    result = BenchmarkResult(name="torch.compile")
    
    if not torch.cuda.is_available():
        result.extra_metrics['status'] = 'CUDA not available'
        return result
    
    device = torch.device('cuda:0')
    
    # Check torch version
    torch_version = tuple(int(x) for x in torch.__version__.split('.')[:2])
    if torch_version < (2, 0):
        result.extra_metrics['status'] = f'torch.compile requires PyTorch 2.0+ (found {torch.__version__})'
        return result
    
    # Create model and input
    model_eager = SimpleMLP(hidden_size=hidden_size, num_layers=num_layers).to(device)
    x = torch.randn(batch_size, hidden_size, device=device, dtype=torch.float32)
    
    # Warmup eager
    model_eager.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model_eager(x)
        torch.cuda.synchronize()
    
    clear_gpu_memory()
    
    # Benchmark eager
    eager_times = []
    with torch.no_grad():
        for _ in range(iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model_eager(x)
            torch.cuda.synchronize()
            eager_times.append((time.perf_counter() - start) * 1000)
    
    _, eager_peak = get_vram_usage()
    clear_gpu_memory()
    
    # Try to compile model
    try:
        # Import Windows Triton compat for setup
        try:
            from src.optimization.windows_triton_compat import (
                ensure_windows_triton_compat,
                get_torch_compile_backend
            )
            ensure_windows_triton_compat()
            backend = get_torch_compile_backend("inductor")
        except ImportError:
            backend = "inductor"
        
        if backend is None:
            backend = "cudagraphs"  # Fallback
        
        model_compiled = torch.compile(
            SimpleMLP(hidden_size=hidden_size, num_layers=num_layers).to(device),
            backend=backend,
            mode="reduce-overhead"
        )
        model_compiled.eval()
        
        # Warmup compiled (includes compilation time)
        compile_start = time.perf_counter()
        with torch.no_grad():
            for _ in range(warmup):
                _ = model_compiled(x)
            torch.cuda.synchronize()
        compile_time = (time.perf_counter() - compile_start) * 1000
        
        clear_gpu_memory()
        
        # Benchmark compiled
        compiled_times = []
        with torch.no_grad():
            for _ in range(iterations):
                torch.cuda.synchronize()
                start = time.perf_counter()
                _ = model_compiled(x)
                torch.cuda.synchronize()
                compiled_times.append((time.perf_counter() - start) * 1000)
        
        _, compiled_peak = get_vram_usage()
        
        # Calculate results
        avg_eager = sum(eager_times) / len(eager_times)
        avg_compiled = sum(compiled_times) / len(compiled_times)
        
        result.iterations = iterations
        result.avg_latency_ms = avg_compiled
        result.min_latency_ms = min(compiled_times)
        result.max_latency_ms = max(compiled_times)
        result.peak_vram_gb = max(eager_peak, compiled_peak)
        result.iterations_per_sec = 1000 / avg_compiled if avg_compiled > 0 else 0
        
        result.extra_metrics = {
            'backend': backend,
            'eager_avg_ms': avg_eager,
            'compiled_avg_ms': avg_compiled,
            'speedup': avg_eager / avg_compiled if avg_compiled > 0 else 0,
            'compile_time_ms': compile_time,
            'eager_peak_vram_gb': eager_peak,
            'compiled_peak_vram_gb': compiled_peak,
        }
        
        del model_compiled
        
    except Exception as e:
        result.extra_metrics = {
            'status': f'Compilation failed: {str(e)}',
            'eager_avg_ms': sum(eager_times) / len(eager_times),
        }
    
    # Cleanup
    del model_eager, x
    clear_gpu_memory()
    
    return result


# =============================================================================
# Benchmark: Full Pipeline Comparison
# =============================================================================

def benchmark_full_comparison(
    tensor_size_mb: int = 256,
    compute_iterations: int = 100,
    warmup: int = 10
) -> Dict[str, BenchmarkResult]:
    """
    Full comparison of Eager vs Compiled+Async modes.
    
    Simulates a realistic workload with:
    - Data loading from CPU
    - GPU computation
    - Result retrieval
    """
    results = {}
    
    if not torch.cuda.is_available():
        return {'error': BenchmarkResult(name='Error', extra_metrics={'status': 'CUDA not available'})}
    
    device = torch.device('cuda:0')
    tensor_size = (tensor_size_mb * 1024 * 1024) // 4
    
    # Create model
    model = SimpleMLP(hidden_size=4096, num_layers=4).to(device)
    model.eval()
    
    # ===== EAGER MODE =====
    print_subheader("Eager Mode")
    
    # Create data
    cpu_data = [torch.randn(32, 4096, dtype=torch.float32) for _ in range(compute_iterations)]
    
    # Warmup
    with torch.no_grad():
        for i in range(min(warmup, len(cpu_data))):
            x = cpu_data[i].to(device)
            _ = model(x)
            torch.cuda.synchronize()
    
    clear_gpu_memory()
    
    # Benchmark eager
    eager_times = []
    with torch.no_grad():
        for data in cpu_data:
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            x = data.to(device)  # Blocking transfer
            output = model(x)
            torch.cuda.synchronize()
            
            eager_times.append((time.perf_counter() - start) * 1000)
    
    _, eager_peak = get_vram_usage()
    
    eager_result = BenchmarkResult(name="Eager Mode")
    eager_result.iterations = len(eager_times)
    eager_result.total_time_s = sum(eager_times) / 1000
    eager_result.avg_latency_ms = sum(eager_times) / len(eager_times)
    eager_result.min_latency_ms = min(eager_times)
    eager_result.max_latency_ms = max(eager_times)
    eager_result.iterations_per_sec = len(eager_times) / eager_result.total_time_s
    eager_result.peak_vram_gb = eager_peak
    results['eager'] = eager_result
    
    clear_gpu_memory()
    del cpu_data
    
    # ===== COMPILED + ASYNC MODE =====
    print_subheader("Compiled + Async Mode")
    
    # Recreate data with pinned memory
    cpu_data_pinned = [
        torch.randn(32, 4096, dtype=torch.float32, pin_memory=True)
        for _ in range(compute_iterations)
    ]
    
    # Try to compile model
    try:
        model_compiled = torch.compile(model, mode="reduce-overhead")
    except Exception:
        model_compiled = model  # Fallback to eager
    
    # Create async stream
    async_stream = torch.cuda.Stream()
    
    # Warmup with prefetching
    with torch.no_grad():
        for i in range(min(warmup, len(cpu_data_pinned))):
            x = cpu_data_pinned[i].to(device, non_blocking=True)
            torch.cuda.synchronize()
            _ = model_compiled(x)
            torch.cuda.synchronize()
    
    clear_gpu_memory()
    
    # Benchmark with prefetching
    async_times = []
    next_batch = None
    
    with torch.no_grad():
        for i, data in enumerate(cpu_data_pinned):
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            # Use prefetched batch if available
            if next_batch is not None:
                x = next_batch
            else:
                x = data.to(device, non_blocking=True)
            
            # Start prefetching next batch
            if i + 1 < len(cpu_data_pinned):
                with torch.cuda.stream(async_stream):
                    next_batch = cpu_data_pinned[i + 1].to(device, non_blocking=True)
            
            # Compute
            output = model_compiled(x)
            torch.cuda.synchronize()
            
            async_times.append((time.perf_counter() - start) * 1000)
    
    _, async_peak = get_vram_usage()
    
    async_result = BenchmarkResult(name="Compiled + Async Mode")
    async_result.iterations = len(async_times)
    async_result.total_time_s = sum(async_times) / 1000
    async_result.avg_latency_ms = sum(async_times) / len(async_times)
    async_result.min_latency_ms = min(async_times)
    async_result.max_latency_ms = max(async_times)
    async_result.iterations_per_sec = len(async_times) / async_result.total_time_s
    async_result.peak_vram_gb = async_peak
    results['async'] = async_result
    
    # Calculate speedup
    if eager_result.avg_latency_ms > 0:
        async_result.extra_metrics['speedup'] = eager_result.avg_latency_ms / async_result.avg_latency_ms
        async_result.extra_metrics['improvement_percent'] = (
            (eager_result.avg_latency_ms - async_result.avg_latency_ms) / eager_result.avg_latency_ms * 100
        )
    
    # Cleanup
    del model, cpu_data_pinned
    if model_compiled is not model:
        del model_compiled
    clear_gpu_memory()
    
    return results


# =============================================================================
# Result Printing
# =============================================================================

def print_result(result: BenchmarkResult) -> None:
    """Print benchmark result"""
    print(f"\n  {result.name}")
    print(f"  " + "-" * 50)
    
    if result.iterations > 0:
        print(f"    Iterations:        {result.iterations}")
        
        if result.iterations_per_sec > 0:
            print(f"    Throughput:        {result.iterations_per_sec:.2f} it/s")
        
        if result.avg_latency_ms > 0:
            print(f"    Avg Latency:       {result.avg_latency_ms:.3f} ms")
            print(f"    Min Latency:       {result.min_latency_ms:.3f} ms")
            print(f"    Max Latency:       {result.max_latency_ms:.3f} ms")
        
        if result.peak_vram_gb > 0:
            print(f"    Peak VRAM:         {result.peak_vram_gb:.3f} GB")
        
        if result.transfer_bandwidth_gbs > 0:
            print(f"    Bandwidth:         {result.transfer_bandwidth_gbs:.2f} GB/s")
    
    # Extra metrics
    for key, value in result.extra_metrics.items():
        if isinstance(value, float):
            print(f"    {key:18} {value:.3f}")
        else:
            print(f"    {key:18} {value}")


def print_comparison(results: Dict[str, BenchmarkResult]) -> None:
    """Print comparison between eager and async modes"""
    if 'eager' not in results or 'async' not in results:
        return
    
    eager = results['eager']
    async_result = results['async']
    
    print("\n" + "=" * 70)
    print("  COMPARISON: Eager Mode vs Compiled + Async Mode")
    print("=" * 70)
    
    print("\n  Metric                    Eager       Compiled+Async   Improvement")
    print("  " + "-" * 64)
    
    # Throughput
    if eager.iterations_per_sec > 0 and async_result.iterations_per_sec > 0:
        improvement = (async_result.iterations_per_sec - eager.iterations_per_sec) / eager.iterations_per_sec * 100
        print(f"  Throughput (it/s)         {eager.iterations_per_sec:>8.2f}    {async_result.iterations_per_sec:>12.2f}     {improvement:>+.1f}%")
    
    # Latency
    if eager.avg_latency_ms > 0 and async_result.avg_latency_ms > 0:
        improvement = (eager.avg_latency_ms - async_result.avg_latency_ms) / eager.avg_latency_ms * 100
        print(f"  Avg Latency (ms)          {eager.avg_latency_ms:>8.3f}    {async_result.avg_latency_ms:>12.3f}     {improvement:>+.1f}%")
    
    # VRAM
    if eager.peak_vram_gb > 0 and async_result.peak_vram_gb > 0:
        improvement = (eager.peak_vram_gb - async_result.peak_vram_gb) / eager.peak_vram_gb * 100
        print(f"  Peak VRAM (GB)            {eager.peak_vram_gb:>8.3f}    {async_result.peak_vram_gb:>12.3f}     {improvement:>+.1f}%")
    
    # Overall speedup
    if 'speedup' in async_result.extra_metrics:
        speedup = async_result.extra_metrics['speedup']
        print(f"\n  Overall Speedup: {speedup:.2f}x")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Run benchmarks"""
    parser = argparse.ArgumentParser(
        description="SeedVR2 Optimization Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/optimization_benchmark.py                  # Full benchmark
  python scripts/optimization_benchmark.py --quick          # Quick test
  python scripts/optimization_benchmark.py --compare        # Mode comparison
  python scripts/optimization_benchmark.py --pinned-only    # Test pinned memory
  python scripts/optimization_benchmark.py --compile-only   # Test torch.compile
        """
    )
    
    parser.add_argument('--quick', action='store_true',
                       help='Quick benchmark with fewer iterations')
    parser.add_argument('--full', action='store_true',
                       help='Full comprehensive benchmark')
    parser.add_argument('--compare', action='store_true',
                       help='Run eager vs compiled+async comparison')
    parser.add_argument('--pinned-only', action='store_true',
                       help='Only test pinned memory transfer')
    parser.add_argument('--streams-only', action='store_true',
                       help='Only test CUDA stream overlap')
    parser.add_argument('--compile-only', action='store_true',
                       help='Only test torch.compile')
    
    args = parser.parse_args()
    
    # Determine test parameters
    if args.quick:
        iterations = 5
        warmup = 2
        size_mb = 128
    elif args.full:
        iterations = 100
        warmup = 20
        size_mb = 512
    else:
        iterations = 20
        warmup = 5
        size_mb = 256
    
    # Print header
    print("\n" + "=" * 70)
    print("  SeedVR2 Optimization Benchmark")
    print("  GPU Memory Management & Compile Performance Analysis")
    print("=" * 70)
    
    # Print GPU info
    print_header("System Information")
    gpu_info = get_gpu_info()
    
    if gpu_info['available']:
        print(f"  GPU:                   {gpu_info['name']}")
        print(f"  Memory:                {gpu_info['total_memory_gb']:.1f} GB")
        print(f"  Compute Capability:    {gpu_info['compute_capability']}")
        print(f"  Architecture:          ", end="")
        if gpu_info['is_blackwell']:
            print("Blackwell (RTX 50-series)")
        elif gpu_info['is_hopper']:
            print("Hopper (H100)")
        elif gpu_info['is_ampere']:
            print("Ampere (RTX 30/A-series)")
        else:
            print("Unknown/Legacy")
        print(f"  CUDA Version:          {gpu_info['cuda_version']}")
        print(f"  PyTorch Version:       {gpu_info['torch_version']}")
        if gpu_info['cudnn_version']:
            print(f"  cuDNN Version:         {gpu_info['cudnn_version']}")
    else:
        print("  No CUDA GPU available!")
        return 1
    
    results = {}
    
    # Run selected benchmarks
    run_all = not any([args.pinned_only, args.streams_only, args.compile_only, args.compare])
    
    if run_all or args.pinned_only:
        print_header("Benchmark 1: Pinned Memory Transfer")
        results['pinned'] = benchmark_pinned_memory(
            size_mb=size_mb,
            iterations=iterations,
            warmup=warmup
        )
        print_result(results['pinned'])
    
    if run_all or args.streams_only:
        print_header("Benchmark 2: CUDA Stream Overlap")
        results['streams'] = benchmark_stream_overlap(
            size_mb=size_mb // 2,
            iterations=iterations,
            warmup=warmup
        )
        print_result(results['streams'])
    
    if run_all or args.compile_only:
        print_header("Benchmark 3: torch.compile")
        results['compile'] = benchmark_torch_compile(
            iterations=iterations * 2,
            warmup=warmup * 2
        )
        print_result(results['compile'])
    
    if run_all or args.compare:
        print_header("Benchmark 4: Full Pipeline Comparison")
        comparison_results = benchmark_full_comparison(
            tensor_size_mb=size_mb,
            compute_iterations=iterations * 2,
            warmup=warmup
        )
        
        for name, result in comparison_results.items():
            print_result(result)
        
        print_comparison(comparison_results)
    
    # Summary
    print_header("Summary")
    
    if 'pinned' in results and 'speedup' in results['pinned'].extra_metrics:
        speedup = results['pinned'].extra_metrics['speedup']
        print(f"  Pinned Memory Speedup:     {speedup:.2f}x")
    
    if 'streams' in results and 'speedup' in results['streams'].extra_metrics:
        speedup = results['streams'].extra_metrics['speedup']
        efficiency = results['streams'].extra_metrics['overlap_efficiency']
        print(f"  Stream Overlap Speedup:    {speedup:.2f}x ({efficiency:.1f}% overlap)")
    
    if 'compile' in results and 'speedup' in results['compile'].extra_metrics:
        speedup = results['compile'].extra_metrics['speedup']
        backend = results['compile'].extra_metrics.get('backend', 'unknown')
        print(f"  torch.compile Speedup:     {speedup:.2f}x (backend: {backend})")
    
    print("\n" + "=" * 70)
    print("  Benchmark Complete")
    print("=" * 70 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
