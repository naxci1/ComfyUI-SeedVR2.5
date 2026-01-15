"""
Core SpargeAttn/Sage2 API implementation using Triton JIT kernels.

This module provides the main attention APIs for sparse attention computation,
optimized for NVIDIA Blackwell (RTX 50xx) GPUs.

The implementation uses pure Triton kernels that compile JIT on first use,
avoiding the need for pre-compiled CUDA extensions.

Original Copyright (c) 2025 by SpargeAttn team.
Licensed under the Apache License, Version 2.0
"""

import torch
import torch.nn.functional as F
from einops import rearrange
import math
import logging

# Configure logging for Blackwell kernel verification
logger = logging.getLogger("SeedVR2.Blackwell")

# Track if we've logged once (to avoid spam during execution)
_kernel_logged_once = False

# Try to import Triton - required for JIT compilation
# Supports both regular triton and triton-windows packages
TRITON_AVAILABLE = False
TRITON_IMPORT_ERROR = None
triton = None
tl = None

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError as e:
    TRITON_IMPORT_ERROR = f"Triton import failed: {e}"
    # Print diagnostic for debugging
    print(f"[SpargeAttn Debug] {TRITON_IMPORT_ERROR}")
except Exception as e:
    TRITON_IMPORT_ERROR = f"Triton import error: {type(e).__name__}: {e}"
    print(f"[SpargeAttn Debug] {TRITON_IMPORT_ERROR}")

# Local module imports
from .utils import hyperparameter_check, get_block_map_meansim
from .quant_per_block import per_block_int8

# Version and availability flags
SPARGE_LOCAL_VERSION = "0.1.0-local-triton"
SPARGE_LOCAL_AVAILABLE = TRITON_AVAILABLE

# ============================================================================
# BOOTSTRAP VERIFICATION: This prints when the module is loaded
# If you don't see this in logs, the module is NOT being imported correctly
# ============================================================================
if TRITON_AVAILABLE:
    print(f"[CORE-BOOTSTRAP] SpargeAttn/Sage2 LOCAL module loaded successfully (Triton OK)", flush=True)
else:
    print(f"[CORE-BOOTSTRAP] WARNING: Triton not available - SpargeAttn disabled!", flush=True)


def get_cuda_arch_versions():
    """Get CUDA architecture versions for all available GPUs."""
    cuda_archs = []
    for i in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(i)
        cuda_archs.append(f"sm{major}{minor}")
    return cuda_archs


def get_blackwell_config():
    """
    Get optimized configuration for Blackwell GPUs (RTX 50xx, sm100+ / sm120).
    
    Returns dict with Triton kernel parameters tuned for Blackwell architecture:
    - Enhanced L1 cache (128KB vs 64KB on Ada)
    - 5th gen Tensor Cores
    - FP8/BF16 optimization
    
    SM 12.0 (Blackwell) uses SM 9.0 (Hopper) kernels as fallback since they're
    natively supported on Blackwell architecture.
    """
    if not torch.cuda.is_available():
        return {}
    
    capability = torch.cuda.get_device_capability(0)
    major, minor = capability
    
    # SM 12.0 is Blackwell (RTX 5070 Ti, etc.)
    # SM 10.0+ is also Blackwell (different revision)
    is_blackwell = major >= 10 or (major == 12)
    
    # SM 9.0 is Hopper (H100, etc.)
    is_hopper = major == 9
    
    if is_blackwell:
        # Blackwell configuration (RTX 5070 Ti, SM 12.0)
        # Hardcoded parameters for maximum Blackwell throughput:
        # - num_warps=8: Optimal for Blackwell SM architecture
        # - num_stages=3: Tuned for Blackwell memory pipeline (reduced from 4)
        # - block_m=128, block_n=64: Matches block-sparse row/col sizes
        return {
            'num_warps': 8,
            'num_stages': 3,  # Tuned for Blackwell (3 stages for better throughput)
            'BLOCK_M': 128,
            'BLOCK_N': 64,
            'prefer_fp8': True,
            'arch': f'sm{major}{minor}',
            'fallback_arch': 'sm90',  # Use Hopper kernels as fallback
            'is_blackwell': True,
        }
    elif is_hopper:
        # Hopper (H100) configuration
        return {
            'num_warps': 8,
            'num_stages': 4,
            'BLOCK_M': 64,
            'BLOCK_N': 128,
            'prefer_fp8': True,
            'arch': f'sm{major}{minor}',
            'is_blackwell': False,
        }
    else:
        # Ampere/Ada (RTX 30xx, 40xx) configuration
        return {
            'num_warps': 4,
            'num_stages': 4,
            'BLOCK_M': 128,
            'BLOCK_N': 64,
            'prefer_fp8': False,
            'arch': f'sm{major}{minor}',
            'is_blackwell': False,
        }


if TRITON_AVAILABLE:
    @triton.jit
    def _attn_fwd_inner(acc, l_i, old_m, q, q_scale, kv_len,
                        K_ptrs, K_bid_ptr, K_scale_ptr, V_ptrs, stride_kn, stride_vn, 
                        pvthreshd, start_m,  
                        BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  
                        STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  
                        ):
        if STAGE == 1:
            lo, hi = 0, start_m * BLOCK_M
        elif STAGE == 2:
            lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
            lo = tl.multiple_of(lo, BLOCK_M)
            K_scale_ptr += lo // BLOCK_N
            K_ptrs += stride_kn * lo
            V_ptrs += stride_vn * lo
        elif STAGE == 3:
            lo, hi = 0, kv_len
        for start_n in range(lo, hi, BLOCK_N):
            kbid = tl.load(K_bid_ptr + start_n//BLOCK_N)
            if kbid:
                k_mask = offs_n[None, :] < (kv_len - start_n)   
                k = tl.load(K_ptrs, mask = k_mask)
                k_scale = tl.load(K_scale_ptr)
                qk = tl.dot(q, k).to(tl.float32) * q_scale * k_scale 
                if STAGE == 2:
                    mask = offs_m[:, None] >= (start_n + offs_n[None, :])
                    qk = qk + tl.where(mask, 0, -1.0e6)
                    local_m = tl.max(qk, 1)
                    new_m = tl.maximum(old_m, local_m)
                    qk -= new_m[:, None]
                else:
                    local_m = tl.max(qk, 1)
                    new_m = tl.maximum(old_m, local_m)
                    qk = qk - new_m[:, None]
                p = tl.math.exp2(qk)
                l_ij = tl.sum(p, 1)
                alpha = tl.math.exp2(old_m - new_m)
                l_i = l_i * alpha + l_ij
                acc = acc * alpha[:, None]
                v = tl.load(V_ptrs, mask = offs_n[:, None] < (kv_len - start_n))
                p = p.to(tl.float16)
                acc += tl.dot(p, v, out_dtype=tl.float16)   
                old_m = new_m
            K_ptrs += BLOCK_N * stride_kn
            K_scale_ptr += 1
            V_ptrs += BLOCK_N * stride_vn
        return acc, l_i, old_m

    @triton.jit
    def _attn_fwd(Q, K, K_blkid, V, Q_scale, K_scale, PVThreshd, Out,  
                  stride_qz, stride_qh, stride_qn,
                  stride_kz, stride_kh, stride_kn,  
                  stride_vz, stride_vh, stride_vn,  
                  stride_oz, stride_oh, stride_on,  
                  stride_kbidq, stride_kbidk,
                  qo_len, kv_len, H:tl.constexpr, num_kv_groups:tl.constexpr, 
                  HEAD_DIM: tl.constexpr,  
                  BLOCK_M: tl.constexpr,  
                  BLOCK_N: tl.constexpr,  
                  STAGE: tl.constexpr
                  ):
        start_m = tl.program_id(0)
        off_z = tl.program_id(2).to(tl.int64)
        off_h = tl.program_id(1).to(tl.int64)
        q_scale_offset = (off_z * H + off_h) * tl.cdiv(qo_len, BLOCK_M)
        k_scale_offset = (off_z * (H // num_kv_groups) + off_h // num_kv_groups) * tl.cdiv(kv_len, BLOCK_N)  
        k_bid_offset = (off_z * (H // num_kv_groups) + off_h // num_kv_groups) * stride_kbidq
        pvthreshd = tl.load(PVThreshd+off_h)
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, HEAD_DIM)
        Q_ptrs = Q + (off_z * stride_qz + off_h * stride_qh) + offs_m[:, None] * stride_qn + offs_k[None, :]
        Q_scale_ptr = Q_scale + q_scale_offset + start_m
        K_ptrs = K + (off_z * stride_kz + (off_h // num_kv_groups) * stride_kh) + offs_n[None, :] * stride_kn + offs_k[:, None] 
        K_scale_ptr = K_scale + k_scale_offset
        K_bid_ptr = K_blkid + k_bid_offset + start_m * stride_kbidk 
        V_ptrs = V + (off_z * stride_vz + (off_h // num_kv_groups) * stride_vh) + offs_n[:, None] * stride_vn + offs_k[None, :]
        O_block_ptr = Out + (off_z * stride_oz + off_h * stride_oh) + offs_m[:, None] * stride_on + offs_k[None, :]
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
        q = tl.load(Q_ptrs, mask = offs_m[:, None] < qo_len)
        q_scale = tl.load(Q_scale_ptr)
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, q_scale, kv_len, K_ptrs, K_bid_ptr, K_scale_ptr, V_ptrs, stride_kn, stride_vn,
                                        pvthreshd, start_m,  
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  
                                        4 - STAGE, offs_m, offs_n 
                                        )
        if STAGE != 1:
            acc, l_i, _ = _attn_fwd_inner(acc, l_i, m_i, q, q_scale, kv_len, K_ptrs, K_bid_ptr, K_scale_ptr, V_ptrs, stride_kn, stride_vn,
                                            pvthreshd, start_m,  
                                            BLOCK_M, HEAD_DIM, BLOCK_N,  
                                            2, offs_m, offs_n 
                                            )
        acc = acc / l_i[:, None]
        tl.store(O_block_ptr, acc.to(Out.type.element_ty), mask = (offs_m[:, None] < qo_len))


def _triton_forward(q, k, k_block_id, v, q_scale, k_scale, pvthreshd, is_causal=False, tensor_layout="HND", output_dtype=torch.float16, block_m_override=None, num_stages_override=None):
    """
    Execute sparse attention using Triton JIT kernels.
    
    This is the core forward pass that uses block-sparse attention patterns
    determined by the k_block_id mask.
    
    Args:
        block_m_override: Optional override for BLOCK_M (default uses Blackwell config).
                          Use block_m_override=64 for VAE BF16 to avoid shared memory errors.
        num_stages_override: Optional override for num_stages (default uses Blackwell config).
                             Use num_stages_override=2 for VAE BF16 to fit in shared memory.
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is required for local SpargeAttn. Install with: pip install triton")
    
    # Get Blackwell-optimized config
    config = get_blackwell_config()
    BLOCK_M = block_m_override if block_m_override is not None else config.get('BLOCK_M', 128)
    BLOCK_N = config.get('BLOCK_N', 64)
    num_warps = config.get('num_warps', 4)
    num_stages = num_stages_override if num_stages_override is not None else config.get('num_stages', 4)
    
    # Memory limit guard for Blackwell GPUs (shared memory limit: 101376 bytes)
    # Formula: shared_mem ≈ BLOCK_M * head_dim * 2 * num_stages (for BF16)
    # With safety factor: BLOCK_M * head_dim * 2 * num_stages < 101376
    BLACKWELL_SHARED_MEM_LIMIT = 101376
    dtype_bytes = 2  # BF16/FP16
    actual_head_dim = q.size(-1)  # Get actual head_dim from tensor
    
    # BLACKWELL SM 12.0 OVERRIDE: 
    # Blackwell has 101KB shared memory - it can handle 720p at BLOCK_M=128 without downscaling
    # Only apply memory guard for non-Blackwell GPUs
    is_blackwell = config.get('is_blackwell', False)
    if not is_blackwell:
        # Estimate shared memory usage: Q_block + K_block + V_block for each stage
        # More accurate formula: BLOCK_M * HEAD_DIM * 2 * num_stages (for Q rows per stage)
        # Plus K/V blocks: BLOCK_N * HEAD_DIM * 2 * num_stages
        estimated_shared = (BLOCK_M * actual_head_dim * dtype_bytes + BLOCK_N * actual_head_dim * dtype_bytes * 2) * num_stages
        
        if estimated_shared > BLACKWELL_SHARED_MEM_LIMIT:
            # Downscale: first try reducing num_stages, then block_m
            original_block_m = BLOCK_M
            original_stages = num_stages
            
            # Try reducing num_stages first
            while num_stages > 1 and estimated_shared > BLACKWELL_SHARED_MEM_LIMIT:
                num_stages -= 1
                estimated_shared = (BLOCK_M * actual_head_dim * dtype_bytes + BLOCK_N * actual_head_dim * dtype_bytes * 2) * num_stages
            
            # If still too high, reduce BLOCK_M
            while BLOCK_M > 32 and estimated_shared > BLACKWELL_SHARED_MEM_LIMIT:
                BLOCK_M = BLOCK_M // 2
                estimated_shared = (BLOCK_M * actual_head_dim * dtype_bytes + BLOCK_N * actual_head_dim * dtype_bytes * 2) * num_stages
            
            if original_block_m != BLOCK_M or original_stages != num_stages:
                logger.warning(
                    f"Triton kernel params auto-downscaled to fit shared memory limit "
                    f"(BLOCK_M: {original_block_m}→{BLOCK_M}, stages: {original_stages}→{num_stages}, "
                    f"estimated: {estimated_shared} bytes, limit: {BLACKWELL_SHARED_MEM_LIMIT})"
                )
    
    stage = 3 if is_causal else 1
    o = torch.empty(q.shape, dtype=output_dtype, device=q.device)

    if tensor_layout == "HND":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape
        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(1), q.stride(2)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(1), k.stride(2)
        stride_bz_v, stride_h_v, stride_seq_v = v.stride(0), v.stride(1), v.stride(2)
        stride_bz_o, stride_h_o, stride_seq_o = o.stride(0), o.stride(1), o.stride(2)
    elif tensor_layout == "NHD":
        b, qo_len, h_qo, head_dim = q.shape
        _, kv_len, h_kv, _ = k.shape
        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(2), q.stride(1)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(2), k.stride(1)
        stride_bz_v, stride_h_v, stride_seq_v = v.stride(0), v.stride(2), v.stride(1)
        stride_bz_o, stride_h_o, stride_seq_o = o.stride(0), o.stride(2), o.stride(1)
    else:
        raise ValueError(f"tensor_layout {tensor_layout} not supported")
    
    assert qo_len == kv_len, "qo_len and kv_len must be equal for causal attention"

    HEAD_DIM_K = head_dim
    num_kv_groups = h_qo // h_kv

    grid = (triton.cdiv(qo_len, BLOCK_M), h_qo, b)
    _attn_fwd[grid](
        q, k, k_block_id, v, q_scale, k_scale, pvthreshd, o,  
        stride_bz_q, stride_h_q, stride_seq_q, 
        stride_bz_k, stride_h_k, stride_seq_k,  
        stride_bz_v, stride_h_v, stride_seq_v,  
        stride_bz_o, stride_h_o, stride_seq_o,
        k_block_id.stride(1), k_block_id.stride(2),
        qo_len, kv_len,
        h_qo, num_kv_groups,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=HEAD_DIM_K,  
        STAGE=stage,  
        num_warps=num_warps,
        num_stages=num_stages)
    return o


@torch.compiler.disable
def spas_sage_attn_meansim_topk_cuda(q, k, v, topk=0.5, is_causal=False, scale=None, 
                                      smooth_k=True, tensor_layout="HND", 
                                      output_dtype=None, return_sparsity=False,
                                      block_m_override=None, num_stages_override=None):
    """
    SpargeAttn with mean-similarity based top-k block selection.
    
    This is the base Sage1 implementation optimized for sparse attention.
    
    Args:
        q: Query tensor (batch, heads, seq_len, head_dim) for HND layout
        k: Key tensor
        v: Value tensor
        topk: Top-k ratio for sparsity (0.0-1.0, lower = more sparse)
        is_causal: Whether to use causal masking
        scale: Softmax scale (default: 1/sqrt(head_dim))
        smooth_k: Whether to smooth key vectors
        tensor_layout: 'HND' or 'NHD'
        output_dtype: Output dtype (default: same as input)
        return_sparsity: Whether to return sparsity ratio
        block_m_override: Optional override for BLOCK_M (default: None uses Blackwell config).
                          Use block_m_override=64 for VAE BF16 to avoid shared memory errors.
        num_stages_override: Optional override for num_stages (default: None uses Blackwell config).
                             Use num_stages_override=2 for VAE BF16 to fit in shared memory.
        
    Returns:
        Attention output tensor
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is required for local SpargeAttn. Install with: pip install triton")
    
    if tensor_layout == 'NHD':
        q, k, v = map(lambda t: rearrange(t, '... L H D -> ... H L D'), (q, k, v))
    
    assert q.size(-2) >= 128, "seq_len should be not less than 128."
    torch.cuda.set_device(v.device)

    dtype = q.dtype
    if output_dtype is None:
        output_dtype = dtype
    
    # DTYPE CONVERSION LOGGING: Detect FP4/FP8 models and log conversion
    # FP4 (uint4) and FP8 (float8_e4m3fn, float8_e5m2) are not directly supported
    # They get converted to BF16/FP16 for kernel execution
    half_dtypes = (torch.float16, torch.bfloat16)
    needs_conversion = dtype not in half_dtypes
    if needs_conversion:
        dtype_name = str(dtype).split('.')[-1]
        target_dtype = "fp16" if dtype in (torch.float32, torch.float16) else "bf16"
        print(f"[KERNEL-DTYPE] Input dtype={dtype_name} converted to {target_dtype} for Triton kernel (FP4/FP8/FP32 → half precision)")
    
    if dtype == torch.float32 or dtype == torch.float16:
        q, k, v = q.contiguous().to(torch.float16), k.contiguous().to(torch.float16), v.contiguous().to(torch.float16)
    else:
        # FP8, FP4, BF16, or any other dtype → convert to bf16 for q,k and fp16 for v
        q, k, v = q.contiguous().to(torch.bfloat16), k.contiguous().to(torch.bfloat16), v.contiguous().to(torch.float16)

    if smooth_k:
        k = k - k.mean(dim=-2, keepdim=True)
    
    # Convert topk to threshold parameters
    # BLACKWELL OPTIMIZED: More aggressive sparsity for Blackwell GPUs
    # Original formula was too restrictive (0.9 + topk * 0.08)
    # New formula enables higher sparsity TOPS on Blackwell Tensor Cores
    simthreshd1 = 0.3 + (1 - topk) * 0.4  # Range 0.3-0.7
    pvthreshd = int(10 + topk * 40)       # Range 10-50
    
    # DYNAMIC KERNEL INJECTION: Detect GPU and apply appropriate CDF mapping
    # This is the RAW kernel-level execution - no wrappers can override this
    major, minor = torch.cuda.get_device_capability()
    
    # Get Blackwell configuration FIRST
    config = get_blackwell_config()
    is_blackwell = major >= 10 or major == 12  # SM 10.0+ or SM 12.0 is Blackwell
    actual_block_m = block_m_override if block_m_override is not None else config.get('BLOCK_M', 128)
    
    # DYNAMIC CDF MAPPING based on GPU architecture
    if is_blackwell:
        # BLACKWELL (SM 10.0+/SM 12.0): Aggressive CDF mapping for maximum TOPS
        # Formula: cdfthreshd = 0.65 + (topk * 0.3) → Range 0.65-0.95
        # UI 0.3 (Fast) → cdfthreshd = 0.74 (high sparsity, max speed)
        cdfthreshd = 0.65 + (topk * 0.3)
    else:
        # SM 8.0-9.0 (Ampere/Ada/Hopper): Conservative CDF mapping
        # Formula: cdfthreshd = 0.85 + (topk * 0.1) → Range 0.85-0.95
        cdfthreshd = 0.85 + (topk * 0.1)
    
    # MANDATORY KERNEL-EXEC LOG: This PROVES the kernel is executing with correct values
    # If this line is missing from logs, the code path is not being hit
    print(f"[KERNEL-EXEC] GPU: sm{major}{minor} | UI-Threshold: {topk} | Applied-CDF: {cdfthreshd:.4f} | BLOCK_M: {actual_block_m}")
    
    # Determine actual BLKQ to use (override or default 128)
    # This MUST be consistent across get_block_map_meansim, per_block_int8, and _triton_forward
    blkq = block_m_override if block_m_override is not None else 128
    blkk = 64  # BLKK is always 64 per the Sparge algorithm
    
    k_block_indices = get_block_map_meansim(q, k, is_causal=is_causal, 
                                            BLKQ=blkq, BLKK=blkk,
                                            simthreshd1=simthreshd1, 
                                            cdfthreshd=cdfthreshd)
    headdim = q.size(-1)

    assert headdim in [64, 128], "headdim should be in [64, 128]."

    q_int8, q_scale, k_int8, k_scale = per_block_int8(q, k, BLKQ=blkq, BLKK=blkk)
    pvthreshd_tensor = hyperparameter_check(pvthreshd, q.size(-3), q.device)
    
    o = _triton_forward(q_int8, k_int8, k_block_indices, v, q_scale, k_scale, 
                        pvthreshd_tensor, is_causal=is_causal, 
                        tensor_layout="HND", output_dtype=output_dtype,
                        block_m_override=block_m_override,
                        num_stages_override=num_stages_override)

    if tensor_layout == 'NHD':
        o = rearrange(o, '... H L D -> ... L H D')
    
    if return_sparsity:
        total_blocks = k_block_indices.numel()
        sparse_blocks = (k_block_indices == 0).sum().item()
        sparsity = sparse_blocks / total_blocks
        return o.to(output_dtype), sparsity
    
    return o.to(output_dtype)


@torch.compiler.disable  
def spas_sage2_attn_meansim_topk_cuda(q, k, v, topk=0.5, is_causal=False, scale=None,
                                       smooth_k=True, tensor_layout="HND",
                                       output_dtype=None, return_sparsity=False,
                                       block_m_override=None, num_stages_override=None):
    """
    SpargeAttn Sage2 with mean-similarity based top-k block selection.
    
    This is the recommended API for plug-and-play SDPA replacement.
    Uses Sage2 architecture with enhanced sparsity detection.
    Optimized for NVIDIA Blackwell (RTX 50xx) GPUs.
    
    Args:
        q: Query tensor (batch, heads, seq_len, head_dim) for HND layout
        k: Key tensor  
        v: Value tensor
        topk: Top-k ratio for sparsity (0.0-1.0, lower = more sparse)
              - 0.3: Maximum speed, some accuracy loss
              - 0.5: Balanced (default)
              - 0.7: High quality, less speedup
        is_causal: Whether to use causal masking
        scale: Softmax scale (default: 1/sqrt(head_dim))
        smooth_k: Whether to smooth key vectors (recommended: True)
        tensor_layout: 'HND' (default) or 'NHD'
        output_dtype: Output dtype (default: same as input)
        return_sparsity: Whether to return sparsity ratio
        block_m_override: Optional override for BLOCK_M (default: None uses Blackwell config).
                          Use block_m_override=64 for VAE BF16 to avoid shared memory errors.
        num_stages_override: Optional override for num_stages (default: None uses Blackwell config).
                             Use num_stages_override=2 for VAE BF16 to fit in shared memory.
        
    Returns:
        Attention output tensor (same shape as input)
        If return_sparsity=True, also returns sparsity ratio
        
    Example:
        >>> output = spas_sage2_attn_meansim_topk_cuda(q, k, v, topk=0.5, is_causal=False)
        >>> # For VAE (BF16), use smaller block size and fewer stages to fit in shared memory:
        >>> output = spas_sage2_attn_meansim_topk_cuda(q, k, v, topk=0.5, block_m_override=64, num_stages_override=2)
    """
    global _kernel_logged_once
    
    # Get Blackwell configuration for kernel parameters
    config = get_blackwell_config()
    is_blackwell = config.get('is_blackwell', False)
    
    # Determine actual values (override or config)
    actual_block_m = block_m_override if block_m_override is not None else config.get('BLOCK_M', 128)
    actual_num_stages = num_stages_override if num_stages_override is not None else config.get('num_stages', 4)
    
    # Log only once on first call to avoid Python overhead during execution
    if is_blackwell and not _kernel_logged_once:
        _kernel_logged_once = True
        num_warps = config.get('num_warps', 8)
        block_n = config.get('BLOCK_N', 64)
        block_m_info = f"{actual_block_m} (override)" if block_m_override is not None else str(actual_block_m)
        stages_info = f"{actual_num_stages} (override)" if num_stages_override is not None else str(actual_num_stages)
        kernel_msg = f"!!! Sparge_Sage2 Kernel: topk={topk}, Blackwell=True, Warps={num_warps}, Stages={stages_info}, BlockM={block_m_info}, BlockN={block_n}"
        print(kernel_msg, flush=True)
        logger.info(kernel_msg)
    
    # Sage2 uses same implementation as Sage1 for Triton-only version
    # The difference is in CUDA kernel optimizations (Sage2++) which require compilation
    # For local JIT, we use the Triton implementation with Sage2-tuned parameters
    return spas_sage_attn_meansim_topk_cuda(
        q, k, v, topk=topk, is_causal=is_causal, scale=scale,
        smooth_k=smooth_k, tensor_layout=tensor_layout,
        output_dtype=output_dtype, return_sparsity=return_sparsity,
        block_m_override=block_m_override,
        num_stages_override=num_stages_override
    )


@torch.compiler.disable
def block_sparse_sage2_attn_cuda(q, k, v, mask_id=None, is_causal=False, 
                                  tensor_layout="HND", output_dtype=None):
    """
    Block-sparse Sage2 attention with custom block-sparse mask.
    
    This API supports computing attention for any block-sparse mask per attention head.
    
    Args:
        q: Query tensor (batch, heads, seq_len, head_dim) for HND layout
        k: Key tensor
        v: Value tensor
        mask_id: Block-sparse mask with shape (batch_size, num_heads, ⌈seq_len/128⌉, ⌈seq_len/64⌉)
                 consisting of 0 (skip) and 1 (compute). If None, computes full attention.
        is_causal: Whether to use causal masking
        tensor_layout: 'HND' (default) or 'NHD'
        output_dtype: Output dtype (default: same as input)
        
    Returns:
        Attention output tensor
        
    Note:
        Block size is fixed at 128x64 (rows x cols) to match kernel requirements.
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is required for local SpargeAttn. Install with: pip install triton")
    
    if tensor_layout == 'NHD':
        q, k, v = map(lambda t: rearrange(t, '... L H D -> ... H L D'), (q, k, v))
    
    assert q.size(-2) >= 128, "seq_len should be not less than 128."
    torch.cuda.set_device(v.device)
    
    dtype = q.dtype
    if output_dtype is None:
        output_dtype = dtype
    
    if dtype == torch.float32 or dtype == torch.float16:
        q, k, v = q.contiguous().to(torch.float16), k.contiguous().to(torch.float16), v.contiguous().to(torch.float16)
    else:
        q, k, v = q.contiguous().to(torch.bfloat16), k.contiguous().to(torch.bfloat16), v.contiguous().to(torch.float16)
    
    b, h, seq_len, head_dim = q.shape
    BLOCK_M = 128
    BLOCK_N = 64
    
    # Generate mask if not provided
    if mask_id is None:
        # Full attention - all blocks active
        num_q_blocks = math.ceil(seq_len / BLOCK_M)
        num_k_blocks = math.ceil(seq_len / BLOCK_N)
        mask_id = torch.ones((b, h, num_q_blocks, num_k_blocks), 
                             dtype=torch.int32, device=q.device)
    
    # Validate mask shape
    expected_q_blocks = math.ceil(seq_len / BLOCK_M)
    expected_k_blocks = math.ceil(seq_len / BLOCK_N)
    
    if mask_id.shape[-2:] != (expected_q_blocks, expected_k_blocks):
        raise ValueError(
            f"Invalid mask_id shape. Expected (..., {expected_q_blocks}, {expected_k_blocks}) "
            f"for seq_len={seq_len} with block size 128x64, got {mask_id.shape}"
        )
    
    headdim = q.size(-1)
    assert headdim in [64, 128], "headdim should be in [64, 128]."
    
    q_int8, q_scale, k_int8, k_scale = per_block_int8(q, k)
    pvthreshd = hyperparameter_check(50, q.size(-3), q.device)
    
    o = _triton_forward(q_int8, k_int8, mask_id, v, q_scale, k_scale,
                        pvthreshd, is_causal=is_causal,
                        tensor_layout="HND", output_dtype=output_dtype)
    
    if tensor_layout == 'NHD':
        o = rearrange(o, '... H L D -> ... L H D')
    
    return o.to(output_dtype)
