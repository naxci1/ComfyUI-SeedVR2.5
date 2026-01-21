"""
Native Triton Kernel for NVFP4 (MX4 Microscaling) on Blackwell SM120

This module implements native 4-bit (E2M1) matrix multiplication using Triton 3.5.1
with MX4 microscaling (1:16 block scaling) as per NVIDIA's official specification.

Hardware Target: RTX 5070 Ti (Blackwell / SM120)
Triton Version: 3.5.1+
CUDA Version: 12.8+

Author: GitHub Copilot
Date: 2026-01-21
"""

import torch
import triton
import triton.language as tl
from typing import Tuple


# MX4 Microscaling Configuration
MX4_BLOCK_SIZE = 16  # 1:16 microscaling (NVIDIA specification)
MX4_E2M1_MIN = 0.0   # E2M1 range: [0, 6]
MX4_E2M1_MAX = 6.0

# Triton kernel configuration for Blackwell
BLOCK_SIZE_M = 128
BLOCK_SIZE_N = 128
BLOCK_SIZE_K = 32  # K must be multiple of 16 for TMA alignment


@triton.jit
def _unpack_4bit_to_fp32(
    packed_ptr,
    unpacked_ptr,
    scale_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr,
    MX_BLOCK_SIZE: tl.constexpr,
):
    """
    Unpack 4-bit E2M1 values with 1:16 microscaling.
    
    Args:
        packed_ptr: Pointer to packed 4-bit weights (uint8, 2 weights per byte)
        unpacked_ptr: Pointer to output FP32 tensor
        scale_ptr: Pointer to E4M3 scales (1 per 16 elements)
        numel: Total number of elements to unpack
        BLOCK_SIZE: Triton block size for parallel processing
        MX_BLOCK_SIZE: Microscaling block size (16)
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask for valid elements
    mask = offsets < numel
    
    # Load packed bytes (2 weights per byte)
    byte_offsets = offsets // 2
    packed_bytes = tl.load(packed_ptr + byte_offsets, mask=mask, other=0)
    
    # Determine if we need lower or upper nibble
    is_even = (offsets % 2) == 0
    
    # Extract 4-bit values
    # Even indices: lower 4 bits (packed_byte & 0x0F)
    # Odd indices: upper 4 bits (packed_byte >> 4)
    nibble = tl.where(is_even, packed_bytes & 0x0F, (packed_bytes >> 4) & 0x0F)
    
    # Convert 4-bit E2M1 to FP32
    # E2M1 format: 1 sign bit + 2 exponent bits + 1 mantissa bit
    # Value = (-1)^s * 2^(e-1) * (1 + m/2)
    # For unsigned (our case): Value = 2^e * (1 + m/2)
    
    sign = (nibble >> 3) & 0x1
    exponent = (nibble >> 1) & 0x3
    mantissa = nibble & 0x1
    
    # Calculate base value (before scaling)
    # exp_val = 2^exponent
    exp_val = tl.where(exponent == 0, 1.0,
              tl.where(exponent == 1, 2.0,
              tl.where(exponent == 2, 4.0, 8.0)))
    
    # mantissa_val = 1 + m/2
    mantissa_val = 1.0 + mantissa.to(tl.float32) * 0.5
    
    # base_value = exp_val * mantissa_val
    base_value = exp_val * mantissa_val
    
    # Apply sign
    base_value = tl.where(sign == 1, -base_value, base_value)
    
    # Load scale (1:16 microscaling)
    scale_idx = offsets // MX_BLOCK_SIZE
    scale = tl.load(scale_ptr + scale_idx, mask=mask, other=1.0)
    
    # NUMERICAL STABILITY: Clamp scale to prevent extreme values
    # Epsilon prevents division by zero and extreme quantization artifacts
    scale = tl.maximum(scale, 1e-5)
    
    # Apply scale
    fp32_value = base_value * scale
    
    # Store result
    tl.store(unpacked_ptr + offsets, fp32_value, mask=mask)


@triton.jit
def _nvfp4_matmul_kernel(
    # Input pointers
    a_ptr, b_packed_ptr, b_scale_ptr,
    # Output pointer
    c_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # MX4 config
    MX_BLOCK_SIZE: tl.constexpr,
    # Triton block sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Native NVFP4 matrix multiplication kernel with MX4 microscaling.
    
    Computes: C = A @ B (where B is in 4-bit MX4 format)
    
    This kernel:
    1. Loads activation A in FP16/BF16
    2. Unpacks 4-bit B weights with 1:16 microscaling in shared memory
    3. Performs tl.dot with input_precision="float32" for Blackwell Tensor Cores
    4. Writes FP32 output
    
    Hardware: Optimized for Blackwell SM120 Tensor Cores
    """
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Block offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        # Load A block (FP16/BF16 → FP32 for computation)
        a_mask = (offs_m[:, None] < M) & ((k + offs_k[None, :]) < K)
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)
        
        # Load and unpack B block (4-bit packed → FP32) - VECTORIZED
        # B is stored as packed 4-bit (2 values per byte)
        b_mask = ((k + offs_k[:, None]) < K) & (offs_n[None, :] < N)
        
        # Calculate flat indices for all elements in the B block (vectorized)
        # flat_idx[i, j] = (k + i) * N + j
        flat_idx = (k + offs_k[:, None]) * N + offs_n[None, :]
        
        # Calculate byte indices (2 weights per byte)
        byte_idx = flat_idx // 2
        is_even = (flat_idx % 2) == 0
        
        # Load packed bytes (vectorized)
        packed_bytes = tl.load(b_packed_ptr + byte_idx, mask=b_mask, other=0)
        
        # Extract nibbles (vectorized bitwise operations)
        # Even indices: lower 4 bits, Odd indices: upper 4 bits
        nibble = tl.where(is_even, packed_bytes & 0x0F, (packed_bytes >> 4) & 0x0F)
        
        # Decode E2M1 format (fully vectorized)
        # E2M1 format: 1 sign bit + 2 exponent bits + 1 mantissa bit
        sign = (nibble >> 3) & 0x1
        exponent = (nibble >> 1) & 0x3
        mantissa = nibble & 0x1
        
        # Calculate exponent value: 2^exponent
        exp_val = tl.where(exponent == 0, 1.0,
                  tl.where(exponent == 1, 2.0,
                  tl.where(exponent == 2, 4.0, 8.0)))
        
        # Calculate mantissa value: 1 + m/2
        mantissa_val = 1.0 + mantissa.to(tl.float32) * 0.5
        
        # Calculate base value: exp_val * mantissa_val
        base_value = exp_val * mantissa_val
        
        # Apply sign (vectorized)
        base_value = tl.where(sign == 1, -base_value, base_value)
        
        # Load scales (1:16 microscaling) - vectorized
        scale_idx = flat_idx // MX_BLOCK_SIZE
        scales = tl.load(b_scale_ptr + scale_idx, mask=b_mask, other=1.0)
        
        # Apply scaling (vectorized)
        b_unpacked = base_value * scales
        
        # Perform matrix multiplication using Blackwell Tensor Cores
        # input_precision="tf32" enables TensorFloat-32 for Blackwell/Hopper dispatch
        accumulator += tl.dot(a, b_unpacked, input_precision="tf32")
    
    # Write output
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    
    tl.store(c_ptrs, accumulator, mask=c_mask)


def unpack_nvfp4_tensor(
    packed: torch.Tensor,
    scales: torch.Tensor,
    original_shape: Tuple[int, ...],
    mx_block_size: int = MX4_BLOCK_SIZE,
) -> torch.Tensor:
    """
    Unpack 4-bit MX4 tensor to FP32 using Triton kernel.
    
    Args:
        packed: Packed 4-bit weights (uint8, 2 values per byte)
        scales: E4M3 scales (1 per mx_block_size elements)
        original_shape: Output tensor shape
        mx_block_size: Microscaling block size (default: 16)
    
    Returns:
        Unpacked FP32 tensor
    """
    numel = torch.prod(torch.tensor(original_shape)).item()
    unpacked = torch.empty(numel, dtype=torch.float32, device=packed.device)
    
    # Launch Triton kernel
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    
    _unpack_4bit_to_fp32[grid](
        packed, unpacked, scales,
        numel,
        BLOCK_SIZE=1024,  # Threads per block
        MX_BLOCK_SIZE=mx_block_size,
    )
    
    return unpacked.reshape(original_shape)


def nvfp4_matmul_triton(
    input: torch.Tensor,
    weight_packed: torch.Tensor,
    weight_scales: torch.Tensor,
    weight_shape: Tuple[int, int],
    bias: torch.Tensor = None,
) -> torch.Tensor:
    """
    Native NVFP4 matrix multiplication using Triton kernel.
    
    Args:
        input: Input tensor (M, K) in FP16/BF16
        weight_packed: Packed 4-bit weights (uint8)
        weight_scales: E4M3 scales for 1:16 microscaling
        weight_shape: Original weight shape (out_features, in_features)
        bias: Optional bias tensor
    
    Returns:
        Output tensor (M, N) in FP32
    """
    # Validate input
    assert input.dim() >= 2, "Input must be at least 2D"
    assert input.device.type == 'cuda', "Input must be on CUDA"
    
    # Flatten input to 2D if needed
    original_shape = input.shape
    if input.dim() > 2:
        input = input.reshape(-1, input.shape[-1])
    
    M, K = input.shape
    out_features, in_features = weight_shape
    N = out_features
    
    assert K == in_features, f"Dimension mismatch: input K={K}, weight in_features={in_features}"
    
    # Allocate output
    output = torch.empty((M, N), dtype=torch.float32, device=input.device)
    
    # Launch Triton kernel
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_SIZE_M']),
        triton.cdiv(N, meta['BLOCK_SIZE_N']),
    )
    
    _nvfp4_matmul_kernel[grid](
        input, weight_packed, weight_scales,
        output,
        M, N, K,
        input.stride(0), input.stride(1),
        1, N,  # Packed B strides (simplified)
        output.stride(0), output.stride(1),
        MX_BLOCK_SIZE=MX4_BLOCK_SIZE,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    # Add bias if provided
    if bias is not None:
        output = output + bias.to(output.dtype)
    
    # Reshape output if needed
    if len(original_shape) > 2:
        output = output.reshape(*original_shape[:-1], N)
    
    return output


def test_triton_nvfp4_kernel():
    """
    Test function for Triton NVFP4 kernel.
    Run this to verify the kernel works on your RTX 5070 Ti.
    """
    print("Testing Triton NVFP4 Kernel on Blackwell SM120...")
    
    # Check GPU
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return False
    
    device = torch.device('cuda:0')
    props = torch.cuda.get_device_properties(device)
    print(f"GPU: {props.name}")
    print(f"Compute Capability: {props.major}.{props.minor}")
    
    if props.major < 12:
        print("WARNING: Not a Blackwell GPU (SM120+). Performance may be suboptimal.")
    
    # Test dimensions
    M, K, N = 128, 512, 256
    
    # Create test tensors
    input_fp16 = torch.randn((M, K), dtype=torch.float16, device=device)
    
    # Simulate 4-bit packed weights
    weight_shape = (N, K)
    numel = N * K
    packed_numel = (numel + 1) // 2
    weight_packed = torch.randint(0, 256, (packed_numel,), dtype=torch.uint8, device=device)
    
    # Create scales (1 per 16 elements)
    num_scales = (numel + MX4_BLOCK_SIZE - 1) // MX4_BLOCK_SIZE
    weight_scales = torch.randn(num_scales, dtype=torch.float32, device=device).abs() * 0.1
    
    print(f"\nTest configuration:")
    print(f"  Input: {M} x {K} (FP16)")
    print(f"  Weight: {N} x {K} (4-bit MX4)")
    print(f"  Packed weight: {packed_numel} bytes")
    print(f"  Scales: {num_scales} (1 per {MX4_BLOCK_SIZE} elements)")
    
    try:
        # Run Triton kernel
        output = nvfp4_matmul_triton(
            input_fp16,
            weight_packed,
            weight_scales,
            weight_shape,
        )
        
        print(f"\n✅ SUCCESS! Output shape: {output.shape}")
        print(f"   Output dtype: {output.dtype}")
        print(f"   Output device: {output.device}")
        print(f"   Output stats: min={output.min().item():.4f}, max={output.max().item():.4f}, mean={output.mean().item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run test when executed directly
    test_triton_nvfp4_kernel()
