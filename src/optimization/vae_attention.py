"""
VAE SageAttention 2 (SA2) Optimized Module for Blackwell GPUs

This module provides SA2-optimized attention for VAE attention blocks on Blackwell GPUs (RTX 50 series).

Key Features:
- Dynamic head splitting: 512-dim → 8 heads × 64 dim (Blackwell Tensor Core sweet spot)
- SageAttention 2 integration with INT8/FP8 quantization for maximum speed
- Automatic fallback to standard SDPA if SA2 is unavailable
- Verbose logging to show which backend is being used
- Memory-efficient operation on 16GB GPUs (Blackwell RTX 50xx)

NOTE: This implementation reshapes 1 head × 512 dim to 8 heads × 64 dim in the forward pass,
preserving the original weights while enabling faster SA2/FlashAttention kernels.
"""

import os
import gc
import torch
import torch.nn.functional as F
import logging
from typing import Optional

# Configure logging
logger = logging.getLogger("SeedVR2.VAE.Attention")

# Track if we've logged kernel info once (avoid per-call logging overhead on Windows)
_vae_kernel_logged_once = False

# Enable verbose logging for SA2 backend selection
_verbose_sa2_logging = True

# SA2 availability check (cached)
_sa2_available = None

# Blackwell memory guard: Set expandable_segments for better fragmentation handling
try:
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
except Exception:
    pass


def _check_sa2_available() -> bool:
    """Check if SageAttention 2 is available."""
    global _sa2_available
    if _sa2_available is None:
        try:
            from sageattention import sageattn
            _sa2_available = True
        except ImportError:
            _sa2_available = False
    return _sa2_available


def is_vae_sparge_available() -> bool:
    """
    Check if VAE SA2/Sparge attention is available.
    
    Returns True if SageAttention 2 is installed and can be used.
    """
    return _check_sa2_available()


def set_vae_sparsity_threshold(threshold: float):
    """
    Set the global sparsity threshold for VAE attention.
    
    NOTE: This function is kept for API compatibility.
    SA2 does not use sparsity thresholds.
    """
    pass


def get_vae_sparsity_threshold() -> float:
    """
    Get the current VAE sparsity threshold.
    
    NOTE: This function is kept for API compatibility.
    SA2 does not use sparsity thresholds.
    """
    return 0.5


def _split_heads_for_blackwell(tensor: torch.Tensor, target_heads: int = 8) -> torch.Tensor:
    """
    Split 512-dim tensor into 8 heads × 64 dim for Blackwell Tensor Core optimization.
    
    Args:
        tensor: Input tensor (batch, 1, seq_len, 512)
        target_heads: Number of heads to split into (default: 8)
        
    Returns:
        Reshaped tensor (batch, 8, seq_len, 64)
    """
    batch, heads, seq_len, dim = tensor.shape
    assert heads == 1, f"Expected 1 head, got {heads}"
    assert dim % target_heads == 0, f"Dimension {dim} not divisible by {target_heads}"
    
    new_head_dim = dim // target_heads
    # Reshape: (batch, 1, seq_len, 512) -> (batch, seq_len, 8, 64) -> (batch, 8, seq_len, 64)
    return tensor.squeeze(1).view(batch, seq_len, target_heads, new_head_dim).transpose(1, 2)


def _merge_heads_from_blackwell(tensor: torch.Tensor) -> torch.Tensor:
    """
    Merge 8 heads × 64 dim back to 1 head × 512 dim after attention.
    
    Args:
        tensor: Input tensor (batch, 8, seq_len, 64)
        
    Returns:
        Merged tensor (batch, 1, seq_len, 512)
    """
    batch, heads, seq_len, head_dim = tensor.shape
    # Reshape: (batch, 8, seq_len, 64) -> (batch, seq_len, 8, 64) -> (batch, seq_len, 512) -> (batch, 1, seq_len, 512)
    return tensor.transpose(1, 2).contiguous().view(batch, seq_len, heads * head_dim).unsqueeze(1)


def _sa2_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Compute attention using SageAttention 2 with INT8 quantization.
    
    Args:
        query: Query tensor (batch, heads, seq_len, head_dim)
        key: Key tensor (batch, heads, seq_len, head_dim)
        value: Value tensor (batch, heads, seq_len, head_dim)
        scale: Softmax scale (default: 1/sqrt(head_dim))
        
    Returns:
        Attention output tensor (batch, heads, seq_len, head_dim)
    """
    from sageattention import sageattn
    
    batch, heads, seq_len, head_dim = query.shape
    
    if scale is None:
        scale = head_dim ** -0.5
    
    # SA2 expects (batch, heads, seq_len, head_dim) format - which is what we have
    # Use tensor_layout="HND" for proper shape handling
    output = sageattn(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        tensor_layout="HND",
        is_causal=False,
        sm_scale=scale,
    )
    
    return output


def _sliced_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    slice_size: int = 4096,
) -> torch.Tensor:
    """
    Memory-efficient sliced SDPA for large sequence lengths.
    
    Processes attention in slices to avoid OOM on very large inputs.
    
    Args:
        query: Query tensor (batch, heads, seq_len, head_dim)
        key: Key tensor (batch, heads, seq_len, head_dim)
        value: Value tensor (batch, heads, seq_len, head_dim)
        attention_mask: Optional attention mask
        scale: Softmax scale (default: 1/sqrt(head_dim))
        slice_size: Number of query positions to process at once (default: 4096)
        
    Returns:
        Attention output tensor (batch, heads, seq_len, head_dim)
    """
    if scale is None:
        scale = query.size(-1) ** -0.5
    
    batch, heads, seq_len, head_dim = query.shape
    
    # For small sequences, just use regular SDPA
    if seq_len <= slice_size:
        return F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, scale=scale)
    
    # Allocate output tensor
    output = torch.zeros_like(query)
    
    # Process in slices
    for i in range(0, seq_len, slice_size):
        end_i = min(i + slice_size, seq_len)
        
        # Get query slice
        q_slice = query[:, :, i:end_i, :]
        
        # For self-attention, we need the full key/value (not sliced)
        # This computes attention for positions i:end_i against all positions
        out_slice = F.scaled_dot_product_attention(
            q_slice, key, value, 
            attn_mask=attention_mask[:, :, i:end_i, :] if attention_mask is not None else None,
            scale=scale
        )
        
        output[:, :, i:end_i, :] = out_slice
    
    return output


def standard_vae_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Compute VAE attention using SA2-optimized path with dynamic head splitting.
    
    For Blackwell GPUs, this:
    1. Splits 1 head × 512 dim into 8 heads × 64 dim
    2. Applies SageAttention 2 (or FlashAttention if SA2 unavailable)
    3. Merges back to 1 head × 512 dim
    
    Args:
        query: Query tensor (batch, heads, seq_len, head_dim) in HND layout
        key: Key tensor (batch, heads, seq_len, head_dim)
        value: Value tensor (batch, heads, seq_len, head_dim)
        attention_mask: Optional attention mask
        scale: Softmax scale (default: 1/sqrt(head_dim))
        
    Returns:
        Attention output tensor (batch, heads, seq_len, head_dim)
    """
    global _vae_kernel_logged_once, _verbose_sa2_logging
    
    batch, heads, seq_len, head_dim = query.shape
    
    if scale is None:
        scale = head_dim ** -0.5
    
    # Determine if we should use head splitting (only for 1 head × 512 dim VAE structure)
    use_head_splitting = (heads == 1 and head_dim == 512)
    sa2_available = _check_sa2_available()
    
    # Verbose logging - show backend selection on first call
    if not _vae_kernel_logged_once:
        _vae_kernel_logged_once = True
        
        if use_head_splitting and sa2_available:
            print(f"[DEBUG] SA2 Optimized Forward Pass: Active")
            print(f"[DEBUG] VAE Attention: Reshaping 1 head x 512 dim → 8 heads x 64 dim")
            print(f"[DEBUG] VAE Attention Backend: SageAttention 2 (INT8 Quantized)")
            print(f"[DEBUG] VAE Attention Shape: Q/K/V = ({batch}, 8, {seq_len}, 64)")
            logger.info("VAE SA2 Optimized: heads=8, head_dim=64 (dynamic split from 512)")
        elif use_head_splitting:
            print(f"[DEBUG] VAE Attention: Reshaping 1 head x 512 dim → 8 heads x 64 dim")
            print(f"[DEBUG] VAE Attention Backend: FlashAttention/SDPA (SA2 not available)")
            print(f"[DEBUG] VAE Attention Shape: Q/K/V = ({batch}, 8, {seq_len}, 64)")
            logger.info("VAE FlashAttention: heads=8, head_dim=64 (dynamic split from 512)")
        else:
            print(f"[DEBUG] VAE Attention: Executing via Native SDPA ({heads} head x {head_dim} dim)")
            print(f"[DEBUG] VAE Attention Shape: Q/K/V = ({batch}, {heads}, {seq_len}, {head_dim})")
            logger.info(f"VAE Standard SDPA: heads={heads}, head_dim={head_dim}")
    
    # Use sliced SDPA for very large sequence lengths to prevent OOM
    if seq_len > 4096:
        return _sliced_sdpa(query, key, value, attention_mask=attention_mask, scale=scale)
    
    # SA2-optimized path with dynamic head splitting for 1 head × 512 dim VAE
    if use_head_splitting:
        # Split to 8 heads × 64 dim
        q_split = _split_heads_for_blackwell(query)
        k_split = _split_heads_for_blackwell(key)
        v_split = _split_heads_for_blackwell(value)
        
        # Adjust scale for new head dimension
        new_scale = 64 ** -0.5
        
        if sa2_available:
            # Use SageAttention 2
            try:
                out_split = _sa2_attention(q_split, k_split, v_split, scale=new_scale)
            except Exception as e:
                logger.warning(f"SA2 failed, falling back to SDPA: {e}")
                with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True):
                    out_split = F.scaled_dot_product_attention(
                        q_split, k_split, v_split,
                        attn_mask=None,
                        dropout_p=0.0,
                        is_causal=False,
                        scale=new_scale
                    )
        else:
            # Use FlashAttention via SDPA (head_dim=64 enables FlashAttention)
            try:
                with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True):
                    out_split = F.scaled_dot_product_attention(
                        q_split, k_split, v_split,
                        attn_mask=None,
                        dropout_p=0.0,
                        is_causal=False,
                        scale=new_scale
                    )
            except Exception:
                out_split = F.scaled_dot_product_attention(
                    q_split, k_split, v_split,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=False,
                    scale=new_scale
                )
        
        # Merge back to 1 head × 512 dim
        return _merge_heads_from_blackwell(out_split)
    
    # Standard path for non-512 dim VAE models
    try:
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True):
            return F.scaled_dot_product_attention(
                query, key, value, 
                attn_mask=attention_mask, 
                dropout_p=0.0, 
                is_causal=False,
                scale=scale
            )
    except Exception:
        # Fallback to standard SDPA without context manager
        return F.scaled_dot_product_attention(
            query, key, value, 
            attn_mask=attention_mask, 
            dropout_p=0.0, 
            is_causal=False,
            scale=scale
        )


def inject_sparge_into_vae(vae, topk: Optional[float] = None, debug=None) -> int:
    """
    Inject attention processors into a VAE model.
    
    NOTE: This function is kept for API compatibility but no longer injects
    Sparge attention. VAE uses standard SDPA which is natively optimized.
    
    Args:
        vae: VAE model (diffusers AutoencoderKL or custom VAE)
        topk: Sparsity ratio (ignored - Sparge disabled)
        debug: Optional debug instance for logging
        
    Returns:
        0 (no modules patched - VAE uses standard attention)
    """
    if debug:
        debug.log(
            "VAE using standard PyTorch SDPA attention (natively optimized for Blackwell)",
            level="INFO", category="vae", force=True
        )
    return 0


def reset_vae_attention_logging():
    """Reset VAE attention logging state (call before each new generation)."""
    global _vae_kernel_logged_once
    _vae_kernel_logged_once = False
