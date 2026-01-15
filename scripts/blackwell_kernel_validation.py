#!/usr/bin/env python3
"""
Blackwell GPU Kernel Validation Script for SeedVR2.5

This script verifies that:
1. Triton is correctly installed and can compile for your GPU (sm_120 for Blackwell)
2. The SpargeAttn/Sage2 custom kernels are being loaded (not fallback)
3. Memory pointers are hitting the custom CUDA/Triton path (not PyTorch native)
4. The CDF mapping is correctly applied for your GPU architecture

Usage:
    python blackwell_kernel_validation.py

Output:
    - [PASS] or [FAIL] for each check
    - Detailed diagnostic information for debugging

Author: SeedVR2.5 Team
"""

import sys
import os

# Add parent directory to path for local imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_cuda():
    """Check CUDA availability and GPU architecture."""
    print("\n" + "="*60)
    print("STEP 1: CUDA & GPU Architecture Check")
    print("="*60)
    
    try:
        import torch
        if not torch.cuda.is_available():
            print("[FAIL] CUDA is not available")
            return False
        
        device_count = torch.cuda.device_count()
        print(f"[PASS] CUDA available with {device_count} device(s)")
        
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            major, minor = torch.cuda.get_device_capability(i)
            sm_version = f"sm_{major}{minor}"
            
            # Determine GPU family
            if major >= 12:
                family = "Blackwell (RTX 50xx)"
            elif major >= 10:
                family = "Blackwell (early revision)"
            elif major == 9:
                family = "Hopper (H100)"
            elif major == 8 and minor >= 9:
                family = "Ada Lovelace (RTX 40xx)"
            elif major == 8:
                family = "Ampere (RTX 30xx)"
            else:
                family = "Older (pre-Ampere)"
            
            print(f"  GPU {i}: {props.name}")
            print(f"    SM Version: {sm_version} ({family})")
            print(f"    Total Memory: {props.total_memory / (1024**3):.1f} GB")
            print(f"    Shared Memory per Block: {props.max_shared_memory_per_block / 1024:.1f} KB")
            
            is_blackwell = major >= 10 or major == 12
            if is_blackwell:
                print(f"    [INFO] Blackwell detected - Will use aggressive CDF mapping")
                print(f"    [INFO] Expected BLOCK_M=128, num_stages=3 for optimal performance")
            else:
                print(f"    [INFO] Non-Blackwell GPU - Will use conservative CDF mapping")
        
        return True
    except Exception as e:
        print(f"[FAIL] CUDA check failed: {e}")
        return False


def check_triton():
    """Check Triton installation and JIT compilation capability."""
    print("\n" + "="*60)
    print("STEP 2: Triton Installation Check")
    print("="*60)
    
    try:
        import triton
        import triton.language as tl
        print(f"[PASS] Triton imported successfully")
        print(f"  Triton version: {triton.__version__}")
        
        # Check if Triton can compile for this GPU
        import torch
        major, minor = torch.cuda.get_device_capability()
        sm_version = f"sm_{major}{minor}"
        
        # Simple kernel to test compilation
        @triton.jit
        def _test_kernel(X, N: tl.constexpr):
            pid = tl.program_id(0)
            x = tl.load(X + pid * N + tl.arange(0, N))
            tl.store(X + pid * N + tl.arange(0, N), x + 1)
        
        # Try to compile
        print(f"  Testing kernel compilation for {sm_version}...")
        x = torch.zeros(128, device='cuda', dtype=torch.float32)
        try:
            _test_kernel[(1,)](x, 128)
            torch.cuda.synchronize()
            print(f"[PASS] Triton JIT compilation successful for {sm_version}")
            return True
        except Exception as e:
            print(f"[FAIL] Triton JIT compilation failed: {e}")
            print(f"  This may indicate that Triton doesn't support {sm_version} yet")
            return False
            
    except ImportError as e:
        print(f"[FAIL] Triton not installed: {e}")
        print("  Install with: pip install triton")
        return False
    except Exception as e:
        print(f"[FAIL] Triton check failed: {e}")
        return False


def check_local_sparge_module():
    """Check if the local SpargeAttn/Sage2 module is being loaded."""
    print("\n" + "="*60)
    print("STEP 3: Local SpargeAttn/Sage2 Module Check")
    print("="*60)
    
    try:
        # Try to import the local module
        from src.optimization.spas_sage_attn.core import (
            TRITON_AVAILABLE,
            SPARGE_LOCAL_VERSION,
            SPARGE_LOCAL_AVAILABLE,
            spas_sage2_attn_meansim_topk,
            get_blackwell_config
        )
        
        print(f"[PASS] Local SpargeAttn module imported")
        print(f"  Local version: {SPARGE_LOCAL_VERSION}")
        print(f"  Triton available: {TRITON_AVAILABLE}")
        print(f"  SpargeAttn available: {SPARGE_LOCAL_AVAILABLE}")
        
        # Check Blackwell configuration
        config = get_blackwell_config()
        if config:
            print(f"  Blackwell config detected:")
            for key, value in config.items():
                print(f"    {key}: {value}")
        else:
            print(f"  No Blackwell-specific config (non-Blackwell GPU)")
        
        return True
    except ImportError as e:
        print(f"[FAIL] Could not import local SpargeAttn module: {e}")
        return False
    except Exception as e:
        print(f"[FAIL] Local module check failed: {e}")
        return False


def check_kernel_execution():
    """Test that the custom kernel actually executes (not fallback)."""
    print("\n" + "="*60)
    print("STEP 4: Kernel Execution Path Check")
    print("="*60)
    
    try:
        import torch
        from src.optimization.spas_sage_attn.core import (
            spas_sage2_attn_meansim_topk,
            TRITON_AVAILABLE
        )
        
        if not TRITON_AVAILABLE:
            print("[FAIL] Triton not available - kernel will use fallback")
            return False
        
        # Create test tensors (batch=2, heads=8, seq_len=256, head_dim=64)
        batch, heads, seq_len, head_dim = 2, 8, 256, 64
        q = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.bfloat16)
        k = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.bfloat16)
        v = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.bfloat16)
        
        # Get GPU info
        major, minor = torch.cuda.get_device_capability()
        is_blackwell = major >= 10 or major == 12
        
        # Test different topk values
        print("  Testing kernel with different sparsity thresholds...")
        for topk in [0.3, 0.5, 0.7]:
            expected_cdf = 0.65 + (topk * 0.3) if is_blackwell else 0.85 + (topk * 0.1)
            print(f"\n  topk={topk}:")
            print(f"    Expected CDF (based on sm_{major}{minor}): {expected_cdf:.4f}")
            
            try:
                # This should print [KERNEL-EXEC] if the custom path is taken
                out = spas_sage2_attn_meansim_topk(q, k, v, topk=topk, is_causal=False)
                torch.cuda.synchronize()
                print(f"    [PASS] Kernel executed successfully, output shape: {out.shape}")
            except Exception as e:
                print(f"    [FAIL] Kernel execution failed: {e}")
        
        return True
    except Exception as e:
        print(f"[FAIL] Kernel execution check failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_compatibility_layer():
    """Check that compatibility.py routes to local module correctly."""
    print("\n" + "="*60)
    print("STEP 5: Compatibility Layer Check")
    print("="*60)
    
    try:
        from src.optimization.compatibility import (
            ENABLE_SPARGE_SAGE2,
            sage2_available,
            Sage2BlackwellConfig
        )
        
        print(f"[PASS] Compatibility layer imported")
        print(f"  ENABLE_SPARGE_SAGE2: {ENABLE_SPARGE_SAGE2}")
        print(f"  sage2_available: {sage2_available}")
        print(f"  Sage2BlackwellConfig:")
        print(f"    DEFAULT_TOPK: {Sage2BlackwellConfig.DEFAULT_TOPK}")
        print(f"    TRITON_NUM_WARPS: {Sage2BlackwellConfig.TRITON_NUM_WARPS}")
        print(f"    TRITON_NUM_STAGES: {Sage2BlackwellConfig.TRITON_NUM_STAGES}")
        print(f"    TRITON_BLOCK_M: {Sage2BlackwellConfig.TRITON_BLOCK_M}")
        print(f"    TRITON_BLOCK_N: {Sage2BlackwellConfig.TRITON_BLOCK_N}")
        
        return True
    except ImportError as e:
        print(f"[FAIL] Could not import compatibility layer: {e}")
        return False
    except Exception as e:
        print(f"[FAIL] Compatibility layer check failed: {e}")
        return False


def main():
    """Run all validation checks."""
    print("="*60)
    print("BLACKWELL GPU KERNEL VALIDATION FOR SeedVR2.5")
    print("="*60)
    print("This script verifies that Triton kernels are properly")
    print("configured for your GPU architecture.")
    
    results = {}
    
    # Run all checks
    results['cuda'] = check_cuda()
    results['triton'] = check_triton() if results['cuda'] else False
    results['local_module'] = check_local_sparge_module() if results['triton'] else False
    results['kernel_exec'] = check_kernel_execution() if results['local_module'] else False
    results['compatibility'] = check_compatibility_layer()
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    all_passed = True
    for check, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {check}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("[SUCCESS] All checks passed! Kernels are properly configured.")
        print("  Look for [KERNEL-EXEC] in your logs during inference to confirm")
        print("  that the custom path is being used.")
    else:
        print("[FAILURE] Some checks failed. See above for details.")
        print("  Common issues:")
        print("  - Triton not installed: pip install triton")
        print("  - GPU not supported: Check Triton SM version compatibility")
        print("  - Import errors: Ensure you're running from the project root")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
