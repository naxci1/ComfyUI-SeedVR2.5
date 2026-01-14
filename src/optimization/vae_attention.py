"""
VAE Standard Attention Module for Blackwell GPUs (Speed Optimized)

This module provides standard PyTorch SDPA (Scaled Dot-Product Attention) for VAE attention blocks.
The experimental Sparge/SageAttention 2 kernels have been removed for stability.

Key Features:
- Uses standard torch.nn.functional.scaled_dot_product_attention (natively optimized for all GPUs)
- Fast-path SDPA with sdp_kernel context manager for maximum speed
- Sliced SDPA fallback for very large sequence lengths (> 4096)
- Verbose logging to show which backend is being used (FlashAttention, Memory Efficient, etc.)
- Stable operation on 16GB GPUs (Blackwell RTX 50xx)

NOTE: Sparge attention is no longer used for VAE. The original 1 head × 512 dim structure
is maintained. Standard SDPA is natively optimized by PyTorch for Blackwell GPUs.
"""

import torch
import torch.nn.functional as F
import logging
from typing import Optional

# Configure logging
logger = logging.getLogger("SeedVR2.VAE.Attention")

# Track if we've logged kernel info once (avoid per-call logging overhead on Windows)
_vae_kernel_logged_once = False

# Enable verbose logging for SDPA backend selection
_verbose_sdpa_logging = True


def is_vae_sparge_available() -> bool:
    """
    Check if VAE Sparge attention is available.
    
    Always returns False - Sparge attention has been disabled for VAE
    in favor of standard SDPA for stability.
    """
    return False


def set_vae_sparsity_threshold(threshold: float):
    """
    Set the global sparsity threshold for VAE attention.
    
    NOTE: This function is kept for API compatibility but has no effect.
    Sparge attention has been disabled for VAE.
    """
    pass


def get_vae_sparsity_threshold() -> float:
    """
    Get the current VAE sparsity threshold.
    
    NOTE: This function is kept for API compatibility.
    Sparge attention has been disabled for VAE.
    """
    return 0.5


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
    Compute VAE attention using standard PyTorch SDPA with Fast-Path optimization.
    
    This is the stable, production-ready path for VAE attention on all GPUs.
    PyTorch's SDPA is natively optimized for Blackwell GPUs.
    
    Uses sdp_kernel context manager to force the fastest available backend.
    
    Args:
        query: Query tensor (batch, heads, seq_len, head_dim) in HND layout
        key: Key tensor (batch, heads, seq_len, head_dim)
        value: Value tensor (batch, heads, seq_len, head_dim)
        attention_mask: Optional attention mask
        scale: Softmax scale (default: 1/sqrt(head_dim))
        
    Returns:
        Attention output tensor (batch, heads, seq_len, head_dim)
    """
    global _vae_kernel_logged_once, _verbose_sdpa_logging
    
    batch, heads, seq_len, head_dim = query.shape
    
    if scale is None:
        scale = head_dim ** -0.5
    
    # Verbose SDPA logging - show backend selection on first call
    if not _vae_kernel_logged_once:
        _vae_kernel_logged_once = True
        
        # Determine which backend will be used
        backend_info = "Unknown"
        try:
            # Check available backends
            flash_available = hasattr(torch.backends.cuda, 'flash_sdp_enabled') and torch.backends.cuda.flash_sdp_enabled()
            mem_efficient_available = hasattr(torch.backends.cuda, 'mem_efficient_sdp_enabled') and torch.backends.cuda.mem_efficient_sdp_enabled()
            math_available = hasattr(torch.backends.cuda, 'math_sdp_enabled') and torch.backends.cuda.math_sdp_enabled()
            
            # FlashAttention requires head_dim in [8, 128] and power of 2, or 64/128
            if flash_available and head_dim in [64, 128]:
                backend_info = "FlashAttention (Fastest)"
            elif mem_efficient_available:
                backend_info = "Memory Efficient"
            elif math_available:
                backend_info = "Math (Fallback)"
            else:
                backend_info = "CUDA Default"
        except Exception:
            backend_info = "PyTorch Default"
        
        # Print verbose debug info to terminal
        print(f"[DEBUG] VAE Attention: Executing via Native SDPA (1 head x {head_dim} dim)")
        print(f"[DEBUG] VAE Attention Backend: {backend_info}")
        print(f"[DEBUG] VAE Attention Shape: Q/K/V = ({batch}, {heads}, {seq_len}, {head_dim})")
        print(f"[DEBUG] VAE Attention Scale: {scale:.6f}")
        
        logger.info(
            f"VAE Standard SDPA: heads={heads}, head_dim={head_dim}, "
            f"seq_len={seq_len}, scale={scale:.4f}, backend={backend_info}"
        )
    
    # Use sliced SDPA for very large sequence lengths to prevent OOM
    if seq_len > 4096:
        return _sliced_sdpa(query, key, value, attention_mask=attention_mask, scale=scale)
    
    # Fast-Path SDPA with sdp_kernel context manager
    # Force the fastest available backend: Flash > Memory Efficient > Math
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
