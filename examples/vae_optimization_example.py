#!/usr/bin/env python3
"""
Example script demonstrating VAE optimizations for Windows + RTX 50xx (Blackwell).
This script shows how to:
1. Load and optimize a VAE model
2. Use automatic mixed precision (AMP)
3. Benchmark performance

Usage:
    python examples/vae_optimization_example.py
"""

import torch
import time
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from src.models.video_vae_v3.modules.video_vae import VideoAutoencoderKL
    from src.optimization.vae_optimizer import (
        optimize_for_windows_blackwell,
        configure_amp_context,
        is_fp8_available,
    )
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you run this from the repository root.")
    sys.exit(1)


def create_dummy_vae():
    """Create a small VAE model for testing."""
    vae = VideoAutoencoderKL(
        in_channels=3,
        out_channels=3,
        block_out_channels=(64, 128, 256),
        layers_per_block=1,
        latent_channels=4,
        use_quant_conv=False,
        use_post_quant_conv=False,
        temporal_scale_num=2,
        spatial_downsample_factor=8,
        temporal_downsample_factor=4,
    )
    return vae


def benchmark_vae(vae, input_tensor, num_runs=10, warmup=3):
    """
    Benchmark VAE inference performance.
    
    Args:
        vae: VAE model
        input_tensor: Input tensor
        num_runs: Number of benchmark runs
        warmup: Number of warmup runs
    
    Returns:
        Average time per run in milliseconds
    """
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = vae.encode(input_tensor)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = vae.encode(input_tensor)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms
    
    return sum(times) / len(times)


def main():
    print("=" * 70)
    print("VAE Optimization Demo - Windows + RTX 50xx (Blackwell)")
    print("=" * 70)
    print()
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("⚠ CUDA not available. Running on CPU (optimizations are for GPU).")
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
        print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  PyTorch Version: {torch.__version__}")
        print()
    
    # Create test input (small for quick testing)
    # Shape: (batch, channels, frames, height, width)
    batch_size = 1
    channels = 3
    frames = 8
    height = 256
    width = 256
    
    print(f"Creating test input: [{batch_size}, {channels}, {frames}, {height}, {width}]")
    input_tensor = torch.randn(batch_size, channels, frames, height, width)
    input_tensor = input_tensor.to(device)
    print()
    
    # Test 1: Baseline (no optimizations)
    print("Test 1: Baseline (no optimizations)")
    print("-" * 70)
    vae_baseline = create_dummy_vae()
    vae_baseline = vae_baseline.to(device)
    vae_baseline.eval()
    
    baseline_time = benchmark_vae(vae_baseline, input_tensor)
    print(f"  Average time: {baseline_time:.2f} ms")
    
    if torch.cuda.is_available():
        baseline_memory = torch.cuda.max_memory_allocated() / 1024**3
        print(f"  Peak GPU memory: {baseline_memory:.2f} GB")
        torch.cuda.reset_peak_memory_stats()
    print()
    
    # Test 2: With optimizations
    print("Test 2: With Windows + Blackwell optimizations")
    print("-" * 70)
    vae_optimized = create_dummy_vae()
    vae_optimized = vae_optimized.to(device)
    vae_optimized.enable_windows_blackwell_optimizations(enable_channels_last=True)
    vae_optimized.eval()
    
    optimized_time = benchmark_vae(vae_optimized, input_tensor)
    print(f"  Average time: {optimized_time:.2f} ms")
    
    if torch.cuda.is_available():
        optimized_memory = torch.cuda.max_memory_allocated() / 1024**3
        print(f"  Peak GPU memory: {optimized_memory:.2f} GB")
        torch.cuda.reset_peak_memory_stats()
    print()
    
    # Test 3: With AMP (Automatic Mixed Precision)
    print("Test 3: With AMP (FP16 mixed precision)")
    print("-" * 70)
    
    def benchmark_with_amp(vae, input_tensor, num_runs=10, warmup=3):
        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                    _ = vae.encode(input_tensor)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.perf_counter()
                with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                    _ = vae.encode(input_tensor)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end = time.perf_counter()
                times.append((end - start) * 1000)
        
        return sum(times) / len(times)
    
    amp_time = benchmark_with_amp(vae_optimized, input_tensor)
    print(f"  Average time: {amp_time:.2f} ms")
    
    if torch.cuda.is_available():
        amp_memory = torch.cuda.max_memory_allocated() / 1024**3
        print(f"  Peak GPU memory: {amp_memory:.2f} GB")
    print()
    
    # Summary
    print("=" * 70)
    print("Performance Summary")
    print("=" * 70)
    print(f"  Baseline:           {baseline_time:.2f} ms")
    print(f"  Optimized:          {optimized_time:.2f} ms  ({(baseline_time/optimized_time - 1)*100:+.1f}%)")
    print(f"  Optimized + AMP:    {amp_time:.2f} ms  ({(baseline_time/amp_time - 1)*100:+.1f}%)")
    print()
    
    if torch.cuda.is_available():
        print("Memory Usage:")
        print(f"  Baseline:           {baseline_memory:.2f} GB")
        print(f"  Optimized:          {optimized_memory:.2f} GB  ({(baseline_memory - optimized_memory):.2f} GB saved)")
        print(f"  Optimized + AMP:    {amp_memory:.2f} GB  ({(baseline_memory - amp_memory):.2f} GB saved)")
        print()
    
    # FP8 check
    if is_fp8_available():
        print("✓ FP8 support available (experimental)")
    else:
        print("✗ FP8 not available on this system")
    print()
    
    print("Optimizations applied:")
    print("  ✓ cuDNN benchmark mode")
    print("  ✓ Fused activation functions (F.silu)")
    print("  ✓ Optimized reparameterization (torch.addcmul)")
    print("  ✓ Channels-last memory format (Conv2d)")
    print("  ✓ Automatic mixed precision (FP16)")
    print()
    
    print("For production use:")
    print("  vae.enable_windows_blackwell_optimizations(enable_channels_last=True)")
    print("  with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):")
    print("      output = vae.encode(input)")
    print()
    
    print("See docs/VAE_OPTIMIZATION_WINDOWS_BLACKWELL.md for full details.")


if __name__ == "__main__":
    main()
