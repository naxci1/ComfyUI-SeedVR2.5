#!/usr/bin/env python3
"""
Attention Performance Benchmark for SeedVR2

This script benchmarks attention backends to measure:
- VRAM Consumption (Peak MB)
- Throughput (Tokens/Second)
- Inference Latency (ms)

Usage:
    python scripts/attention_benchmark.py
    python scripts/attention_benchmark.py --backends sdpa sageattn_2 --seq_len 256

Backends tested:
    - sdpa: PyTorch scaled_dot_product_attention
    - flash_attn_2: Flash Attention 2 (Ampere+)
    - flash_attn_3: Flash Attention 3 (Hopper+)
    - sageattn_2: SageAttention 2
    - sageattn_3: SageAttention 3 (Blackwell/RTX 50xx)

Author: SeedVR2 Team
"""

import sys
import os
import time
import argparse
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

# Add parent directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run"""
    backend: str
    batch_size: int
    num_heads: int
    seq_len: int
    head_dim: int
    dtype: str
    
    # Performance metrics
    latency_ms: float
    throughput_tokens_per_sec: float
    
    # Memory metrics
    vram_peak_mb: float
    vram_allocated_mb: float
    
    # Additional info
    warmup_iterations: int
    benchmark_iterations: int
    is_available: bool = True
    error: Optional[str] = None


def print_header(title: str):
    """Print a formatted header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def get_available_backends() -> Dict[str, bool]:
    """Check which attention backends are available"""
    from src.optimization.compatibility import (
        FLASH_ATTN_2_AVAILABLE,
        FLASH_ATTN_3_AVAILABLE,
        SAGE_ATTN_2_AVAILABLE,
        SAGE_ATTN_3_AVAILABLE,
    )
    
    return {
        'sdpa': True,  # Always available
        'flash_attn_2': FLASH_ATTN_2_AVAILABLE,
        'flash_attn_3': FLASH_ATTN_3_AVAILABLE,
        'sageattn_2': SAGE_ATTN_2_AVAILABLE,
        'sageattn_3': SAGE_ATTN_3_AVAILABLE,
    }


def get_gpu_info() -> Dict[str, Any]:
    """Get GPU information"""
    import torch
    
    if not torch.cuda.is_available():
        return {'available': False}
    
    return {
        'available': True,
        'name': torch.cuda.get_device_name(0),
        'compute_capability': torch.cuda.get_device_capability(0),
        'total_memory_gb': torch.cuda.get_device_properties(0).total_memory / 1024**3,
        'cuda_version': torch.version.cuda,
        'pytorch_version': torch.__version__,
    }


def reset_memory_stats():
    """Reset CUDA memory statistics"""
    import torch
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_memory_stats() -> Dict[str, float]:
    """Get current CUDA memory statistics"""
    import torch
    
    if not torch.cuda.is_available():
        return {'peak_mb': 0, 'allocated_mb': 0}
    
    torch.cuda.synchronize()
    
    return {
        'peak_mb': torch.cuda.max_memory_allocated() / 1024**2,
        'allocated_mb': torch.cuda.memory_allocated() / 1024**2,
    }


def create_benchmark_tensors(batch_size: int, num_heads: int, seq_len: int, head_dim: int,
                             dtype, device):
    """Create tensors for benchmarking variable-length attention"""
    import torch
    
    # Uniform sequence lengths (required for fair comparison across backends)
    total_tokens = batch_size * seq_len
    
    # Varlen format: (total_tokens, heads, head_dim)
    q = torch.randn(total_tokens, num_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(total_tokens, num_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(total_tokens, num_heads, head_dim, dtype=dtype, device=device)
    
    # Cumulative sequence lengths
    cu_seqlens = torch.arange(0, (batch_size + 1) * seq_len, seq_len,
                              dtype=torch.int32, device=device)
    
    return q, k, v, cu_seqlens, seq_len


def benchmark_backend(backend: str, batch_size: int, num_heads: int, seq_len: int,
                      head_dim: int, dtype, device, warmup_iters: int = 10,
                      benchmark_iters: int = 100) -> BenchmarkResult:
    """Benchmark a single attention backend"""
    import torch
    from src.models.dit_3b.attention import FlashAttentionVarlen
    
    dtype_str = str(dtype).split('.')[-1]
    
    # Check if backend is available
    available_backends = get_available_backends()
    if not available_backends.get(backend, False):
        return BenchmarkResult(
            backend=backend,
            batch_size=batch_size,
            num_heads=num_heads,
            seq_len=seq_len,
            head_dim=head_dim,
            dtype=dtype_str,
            latency_ms=0,
            throughput_tokens_per_sec=0,
            vram_peak_mb=0,
            vram_allocated_mb=0,
            warmup_iterations=warmup_iters,
            benchmark_iterations=benchmark_iters,
            is_available=False,
            error="Backend not installed"
        )
    
    try:
        # Create tensors
        q, k, v, cu_seqlens, max_seqlen = create_benchmark_tensors(
            batch_size, num_heads, seq_len, head_dim, dtype, device
        )
        
        # Create attention module
        attn = FlashAttentionVarlen(attention_mode=backend)
        
        # Reset memory stats before warmup
        reset_memory_stats()
        
        # Warmup
        for _ in range(warmup_iters):
            _ = attn(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen)
        
        torch.cuda.synchronize()
        
        # Reset memory stats before benchmark
        reset_memory_stats()
        
        # Benchmark
        start_time = time.perf_counter()
        
        for _ in range(benchmark_iters):
            _ = attn(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen)
        
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        # Calculate metrics
        total_time = end_time - start_time
        latency_ms = (total_time / benchmark_iters) * 1000
        
        total_tokens = batch_size * seq_len
        throughput = (total_tokens * benchmark_iters) / total_time
        
        # Get memory stats
        mem_stats = get_memory_stats()
        
        # Cleanup
        del q, k, v, cu_seqlens
        torch.cuda.empty_cache()
        
        return BenchmarkResult(
            backend=backend,
            batch_size=batch_size,
            num_heads=num_heads,
            seq_len=seq_len,
            head_dim=head_dim,
            dtype=dtype_str,
            latency_ms=latency_ms,
            throughput_tokens_per_sec=throughput,
            vram_peak_mb=mem_stats['peak_mb'],
            vram_allocated_mb=mem_stats['allocated_mb'],
            warmup_iterations=warmup_iters,
            benchmark_iterations=benchmark_iters,
        )
        
    except Exception as e:
        return BenchmarkResult(
            backend=backend,
            batch_size=batch_size,
            num_heads=num_heads,
            seq_len=seq_len,
            head_dim=head_dim,
            dtype=dtype_str,
            latency_ms=0,
            throughput_tokens_per_sec=0,
            vram_peak_mb=0,
            vram_allocated_mb=0,
            warmup_iterations=warmup_iters,
            benchmark_iterations=benchmark_iters,
            is_available=True,
            error=str(e)
        )


def run_benchmark_suite(backends: List[str], configs: List[Dict], dtype, device,
                        warmup_iters: int, benchmark_iters: int) -> List[BenchmarkResult]:
    """Run benchmarks for all backends and configurations"""
    results = []
    
    total = len(backends) * len(configs)
    current = 0
    
    for config in configs:
        for backend in backends:
            current += 1
            print(f"\r  Progress: {current}/{total} - {backend} @ {config['seq_len']} seq...", end='', flush=True)
            
            result = benchmark_backend(
                backend=backend,
                batch_size=config['batch_size'],
                num_heads=config['num_heads'],
                seq_len=config['seq_len'],
                head_dim=config['head_dim'],
                dtype=dtype,
                device=device,
                warmup_iters=warmup_iters,
                benchmark_iters=benchmark_iters,
            )
            results.append(result)
    
    print()  # Newline after progress
    return results


def print_results_table(results: List[BenchmarkResult], reference_backend: str = 'sdpa'):
    """Print benchmark results as a formatted table"""
    print_header("Benchmark Results")
    
    # Group by configuration
    configs = {}
    for r in results:
        key = (r.batch_size, r.seq_len)
        if key not in configs:
            configs[key] = {}
        configs[key][r.backend] = r
    
    # Print table for each configuration
    for (batch_size, seq_len), backend_results in configs.items():
        print(f"\n  Configuration: batch={batch_size}, seq_len={seq_len}")
        print(f"  {'Backend':15s} {'Latency (ms)':>15s} {'Throughput':>18s} {'VRAM (MB)':>12s} {'Speedup':>10s}")
        print(f"  {'-'*15} {'-'*15} {'-'*18} {'-'*12} {'-'*10}")
        
        # Get reference latency for speedup calculation
        ref_latency = None
        if reference_backend in backend_results:
            ref_result = backend_results[reference_backend]
            if ref_result.is_available and ref_result.error is None:
                ref_latency = ref_result.latency_ms
        
        for backend in ['sdpa', 'flash_attn_2', 'flash_attn_3', 'sageattn_2', 'sageattn_3']:
            if backend not in backend_results:
                continue
            
            r = backend_results[backend]
            
            if not r.is_available:
                print(f"  {backend:15s} {'N/A (not installed)':>45s}")
                continue
            
            if r.error:
                print(f"  {backend:15s} Error: {r.error[:40]}")
                continue
            
            latency_str = f"{r.latency_ms:.3f}"
            throughput_str = f"{r.throughput_tokens_per_sec:,.0f} tok/s"
            vram_str = f"{r.vram_peak_mb:.1f}"
            
            if ref_latency and r.latency_ms > 0:
                speedup = ref_latency / r.latency_ms
                speedup_str = f"{speedup:.2f}x"
            else:
                speedup_str = "-"
            
            print(f"  {backend:15s} {latency_str:>15s} {throughput_str:>18s} {vram_str:>12s} {speedup_str:>10s}")


def print_performance_summary(results: List[BenchmarkResult]):
    """Print a summary of performance across backends"""
    print_header("Performance Summary")
    
    # Filter successful results
    successful = [r for r in results if r.is_available and r.error is None]
    
    if not successful:
        print("  No successful benchmark results!")
        return
    
    # Group by backend
    by_backend = {}
    for r in successful:
        if r.backend not in by_backend:
            by_backend[r.backend] = []
        by_backend[r.backend].append(r)
    
    # Calculate average metrics per backend
    print(f"\n  Average metrics across all configurations:")
    print(f"  {'Backend':15s} {'Avg Latency':>15s} {'Avg Throughput':>18s} {'Avg VRAM':>12s}")
    print(f"  {'-'*15} {'-'*15} {'-'*18} {'-'*12}")
    
    for backend in ['sdpa', 'flash_attn_2', 'flash_attn_3', 'sageattn_2', 'sageattn_3']:
        if backend not in by_backend:
            continue
        
        backend_results = by_backend[backend]
        avg_latency = sum(r.latency_ms for r in backend_results) / len(backend_results)
        avg_throughput = sum(r.throughput_tokens_per_sec for r in backend_results) / len(backend_results)
        avg_vram = sum(r.vram_peak_mb for r in backend_results) / len(backend_results)
        
        print(f"  {backend:15s} {avg_latency:>12.3f} ms {avg_throughput:>14,.0f} tok/s {avg_vram:>9.1f} MB")
    
    # Best performer per metric
    print(f"\n  Best performers:")
    
    # Lowest latency
    lowest_latency = min(successful, key=lambda r: r.latency_ms)
    print(f"    Lowest latency: {lowest_latency.backend} ({lowest_latency.latency_ms:.3f} ms)")
    
    # Highest throughput
    highest_throughput = max(successful, key=lambda r: r.throughput_tokens_per_sec)
    print(f"    Highest throughput: {highest_throughput.backend} ({highest_throughput.throughput_tokens_per_sec:,.0f} tok/s)")
    
    # Lowest VRAM
    lowest_vram = min(successful, key=lambda r: r.vram_peak_mb)
    print(f"    Lowest VRAM: {lowest_vram.backend} ({lowest_vram.vram_peak_mb:.1f} MB)")


def save_results(results: List[BenchmarkResult], output_path: str, gpu_info: Dict):
    """Save benchmark results to JSON file"""
    data = {
        'gpu_info': gpu_info,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'results': [asdict(r) for r in results],
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n  Results saved to: {output_path}")


def main():
    """Run the benchmark suite"""
    parser = argparse.ArgumentParser(description='Attention Performance Benchmark for SeedVR2')
    parser.add_argument('--backends', nargs='+',
                        default=['sdpa', 'flash_attn_2', 'flash_attn_3', 'sageattn_2', 'sageattn_3'],
                        help='Backends to benchmark')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size (default: 2)')
    parser.add_argument('--num_heads', type=int, default=24,
                        help='Number of attention heads (default: 24)')
    parser.add_argument('--seq_len', type=int, nargs='+', default=[64, 256, 1024],
                        help='Sequence lengths to test (default: 64 256 1024)')
    parser.add_argument('--head_dim', type=int, default=64,
                        help='Head dimension (default: 64)')
    parser.add_argument('--dtype', type=str, default='bfloat16',
                        choices=['float16', 'bfloat16'],
                        help='Data type (default: bfloat16)')
    parser.add_argument('--warmup', type=int, default=10,
                        help='Warmup iterations (default: 10)')
    parser.add_argument('--iterations', type=int, default=100,
                        help='Benchmark iterations (default: 100)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file for results')
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("  Attention Performance Benchmark for SeedVR2")
    print("  Comparing VRAM, Throughput, and Latency across backends")
    print("=" * 70)
    
    # Check CUDA availability
    import torch
    if not torch.cuda.is_available():
        print("\n❌ CUDA not available. Benchmark requires a GPU.")
        return 1
    
    # Get GPU info
    gpu_info = get_gpu_info()
    print(f"\n  GPU: {gpu_info['name']}")
    print(f"  Compute Capability: SM{gpu_info['compute_capability'][0]}{gpu_info['compute_capability'][1]}")
    print(f"  VRAM: {gpu_info['total_memory_gb']:.1f} GB")
    print(f"  CUDA: {gpu_info['cuda_version']}")
    print(f"  PyTorch: {gpu_info['pytorch_version']}")
    
    # Check backend availability
    print_header("Backend Availability")
    available = get_available_backends()
    for backend in args.backends:
        status = "✅ Available" if available.get(backend, False) else "❌ Not installed"
        print(f"  {backend:15s}: {status}")
    
    # Setup
    dtype_map = {'float16': torch.float16, 'bfloat16': torch.bfloat16}
    dtype = dtype_map[args.dtype]
    device = torch.device('cuda')
    
    # Create configurations to test
    configs = [
        {
            'batch_size': args.batch_size,
            'num_heads': args.num_heads,
            'seq_len': seq_len,
            'head_dim': args.head_dim,
        }
        for seq_len in args.seq_len
    ]
    
    print_header("Running Benchmarks")
    print(f"  Configurations: {len(configs)}")
    print(f"  Backends: {len(args.backends)}")
    print(f"  Warmup iterations: {args.warmup}")
    print(f"  Benchmark iterations: {args.iterations}")
    print(f"  Dtype: {args.dtype}")
    print()
    
    # Run benchmarks
    results = run_benchmark_suite(
        backends=args.backends,
        configs=configs,
        dtype=dtype,
        device=device,
        warmup_iters=args.warmup,
        benchmark_iters=args.iterations,
    )
    
    # Print results
    print_results_table(results)
    print_performance_summary(results)
    
    # Save results if requested
    if args.output:
        save_results(results, args.output, gpu_info)
    
    print("\n" + "=" * 70)
    print("  Benchmark Complete")
    print("=" * 70 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
