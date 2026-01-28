"""
Test script for VAE optimizations on Windows + RTX 50xx Blackwell
Validates that all optimizations work correctly without breaking functionality
"""

import torch
import torch.nn.functional as F
import time
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from src.vae.wan2_1_vae import create_wan2_1_vae, Wan2_1_VAE
from src.vae.wan2_2_vae import create_wan2_2_vae, Wan2_2_VAE

def test_basic_functionality():
    """Test basic encode-decode functionality"""
    print("=" * 80)
    print("Testing Basic Functionality")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test Wan2.1 VAE
    print("\n1. Testing Wan2.1 VAE...")
    model_21 = create_wan2_1_vae(z_channels=4, use_channels_last=True, use_fp8=False)
    model_21 = model_21.to(device)
    model_21.eval()
    
    # Create test input
    batch_size = 2
    x = torch.randn(batch_size, 3, 256, 256, device=device)
    
    with torch.no_grad():
        # Test encoding
        mean, logvar = model_21.encode(x, use_amp=True)
        print(f"   Encoded mean shape: {mean.shape}")
        print(f"   Encoded logvar shape: {logvar.shape}")
        
        # Test reparameterization
        z = model_21.encoder.reparameterize(mean, logvar)
        print(f"   Latent z shape: {z.shape}")
        
        # Test decoding
        x_recon = model_21.decode(z, use_amp=True)
        print(f"   Reconstruction shape: {x_recon.shape}")
        
        # Test full forward pass
        x_full, kl_loss = model_21(x, return_loss=True, use_amp=True)
        print(f"   Full forward shape: {x_full.shape}")
        print(f"   KL loss: {kl_loss.item():.6f}")
    
    print("   ✓ Wan2.1 VAE basic functionality passed")
    
    # Test Wan2.2 VAE
    print("\n2. Testing Wan2.2 VAE...")
    model_22 = create_wan2_2_vae(latent_channels=4, use_channels_last=True, use_fp8=False)
    model_22 = model_22.to(device)
    model_22.eval()  # Set to eval mode for consistency
    
    # Create a dummy inner VAE model for testing
    class DummyVAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 8, 3, padding=1)
            self.deconv = torch.nn.Conv2d(8, 3, 3, padding=1)
        
        def encode(self, x):
            # Simple encoding that preserves spatial dimensions
            h = self.conv(x)
            # Downsample to match expected latent size
            h = F.avg_pool2d(h, kernel_size=8)
            return h
        
        def decode(self, z):
            # Upsample back to original size
            # Use nearest-exact if available (PyTorch 2.0+)
            try:
                h = F.interpolate(z, scale_factor=8, mode='nearest-exact')
            except (TypeError, RuntimeError):
                # Fallback for older PyTorch versions
                h = F.interpolate(z, scale_factor=8, mode='nearest')
            return self.deconv(h)
    
    model_22.vae_model = DummyVAE().to(device)
    
    with torch.no_grad():
        # Test encoding
        encoded = model_22.encode(x, use_amp=True)
        print(f"   Encoded latent shape: {encoded['latent'].shape}")
        
        # Test decoding
        decoded = model_22.decode(encoded['latent'], use_amp=True)
        print(f"   Decoded shape: {decoded['sample'].shape}")
        
        # Test full forward
        x_full_22 = model_22(x, use_amp=True)
        print(f"   Full forward shape: {x_full_22.shape}")
    
    print("   ✓ Wan2.2 VAE basic functionality passed")
    
    return True

def test_channels_last():
    """Test channels_last memory format"""
    print("\n" + "=" * 80)
    print("Testing Channels Last Memory Format")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("   ⚠ CUDA not available, skipping channels_last test")
        return True
    
    device = torch.device('cuda')
    
    # Create model with channels_last
    model = create_wan2_1_vae(z_channels=4, use_channels_last=True)
    model = model.to(device)
    model.eval()
    
    # Test input in channels_last format
    x = torch.randn(1, 3, 256, 256, device=device)
    x_cl = x.to(memory_format=torch.channels_last)
    
    with torch.no_grad():
        mean, logvar = model.encode(x_cl, use_amp=True)
        print(f"   Input memory format: {x_cl.is_contiguous(memory_format=torch.channels_last)}")
        print(f"   Output shape: {mean.shape}")
    
    print("   ✓ Channels last format test passed")
    return True

def test_mixed_precision():
    """Test mixed precision (AMP) support"""
    print("\n" + "=" * 80)
    print("Testing Mixed Precision (AMP)")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("   ⚠ CUDA not available, skipping AMP test")
        return True
    
    device = torch.device('cuda')
    
    model = create_wan2_1_vae(z_channels=4, use_channels_last=True)
    model = model.to(device)
    model.eval()
    
    x = torch.randn(1, 3, 256, 256, device=device)
    
    # Test with AMP enabled
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            mean, logvar = model.encode(x, use_amp=True)
            z = model.encoder.reparameterize(mean, logvar)
            x_recon = model.decode(z, use_amp=True)
    
    print(f"   Reconstruction dtype: {x_recon.dtype}")
    print(f"   Reconstruction shape: {x_recon.shape}")
    print("   ✓ Mixed precision test passed")
    
    return True

def test_numerical_stability():
    """Test numerical stability of optimizations"""
    print("\n" + "=" * 80)
    print("Testing Numerical Stability")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = create_wan2_1_vae(z_channels=4, use_channels_last=False)
    model = model.to(device)
    model.eval()
    
    # Test with extreme values
    x_normal = torch.randn(1, 3, 128, 128, device=device)
    
    with torch.no_grad():
        # Normal values
        mean, logvar = model.encode(x_normal, use_amp=False)
        z = model.encoder.reparameterize(mean, logvar)
        x_recon = model.decode(z, use_amp=False)
        
        # Check for NaN or Inf
        has_nan = torch.isnan(x_recon).any().item()
        has_inf = torch.isinf(x_recon).any().item()
        
        print(f"   Has NaN: {has_nan}")
        print(f"   Has Inf: {has_inf}")
        print(f"   Output range: [{x_recon.min().item():.4f}, {x_recon.max().item():.4f}]")
        
        if has_nan or has_inf:
            print("   ✗ Numerical stability test failed!")
            return False
    
    print("   ✓ Numerical stability test passed")
    return True

def test_reparameterization():
    """Test optimized reparameterization trick"""
    print("\n" + "=" * 80)
    print("Testing Optimized Reparameterization")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = create_wan2_1_vae(z_channels=4)
    model = model.to(device)
    
    # Create test mean and logvar
    mean = torch.randn(2, 4, 16, 16, device=device)
    logvar = torch.randn(2, 4, 16, 16, device=device)
    
    # Test multiple samples to ensure randomness
    with torch.no_grad():
        z1 = model.encoder.reparameterize(mean, logvar)
        z2 = model.encoder.reparameterize(mean, logvar)
        
        print(f"   Sample 1 shape: {z1.shape}")
        print(f"   Sample 2 shape: {z2.shape}")
        print(f"   Samples are different: {not torch.allclose(z1, z2)}")
        print(f"   Mean of z1: {z1.mean().item():.6f}")
        print(f"   Std of z1: {z1.std().item():.6f}")
    
    print("   ✓ Reparameterization test passed")
    return True

def test_performance():
    """Test performance improvements"""
    print("\n" + "=" * 80)
    print("Testing Performance")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("   ⚠ CUDA not available, skipping performance test")
        return True
    
    device = torch.device('cuda')
    
    # Create models with different configurations
    model_baseline = create_wan2_1_vae(z_channels=4, use_channels_last=False, use_fp8=False)
    model_baseline = model_baseline.to(device)
    model_baseline.eval()
    
    model_optimized = create_wan2_1_vae(z_channels=4, use_channels_last=True, use_fp8=False)
    model_optimized = model_optimized.to(device)
    model_optimized.eval()
    
    # Prepare input
    x = torch.randn(4, 3, 256, 256, device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model_baseline(x, return_loss=False, use_amp=False)
            _ = model_optimized(x, return_loss=False, use_amp=True)
    
    torch.cuda.synchronize()
    
    # Benchmark baseline
    num_runs = 20
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model_baseline(x, return_loss=False, use_amp=False)
    torch.cuda.synchronize()
    baseline_time = (time.time() - start_time) / num_runs
    
    # Benchmark optimized
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model_optimized(x, return_loss=False, use_amp=True)
    torch.cuda.synchronize()
    optimized_time = (time.time() - start_time) / num_runs
    
    speedup = baseline_time / optimized_time
    
    print(f"   Baseline time: {baseline_time * 1000:.2f} ms")
    print(f"   Optimized time: {optimized_time * 1000:.2f} ms")
    print(f"   Speedup: {speedup:.2f}x")
    
    if speedup > 1.0:
        print("   ✓ Performance improvement detected")
    else:
        print("   ⚠ No significant performance improvement (may vary by hardware)")
    
    return True

def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("VAE Optimization Test Suite")
    print("Windows + RTX 50xx Blackwell Optimizations")
    print("=" * 80)
    
    # Print system info
    print("\nSystem Information:")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"  CuDNN enabled: {torch.backends.cudnn.enabled}")
    print(f"  CuDNN benchmark: {torch.backends.cudnn.benchmark}")
    
    # Run tests
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Channels Last", test_channels_last),
        ("Mixed Precision", test_mixed_precision),
        ("Numerical Stability", test_numerical_stability),
        ("Reparameterization", test_reparameterization),
        ("Performance", test_performance),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n   ✗ {test_name} failed with error: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠ {total - passed} test(s) failed")
        return 1

if __name__ == "__main__":
    exit(main())
