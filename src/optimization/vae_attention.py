"""
VAE Sparse Attention Processor for Blackwell GPUs

This module provides a SpargeAttnProcessor that can be injected into VAE Attention blocks
to accelerate decoding using block-sparse attention patterns. Optimized for NVIDIA Blackwell
(RTX 50xx, SM 12.0) GPUs.

Key Features:
- Monkey-patches diffusers.models.attention_processor.Attention.forward
- Uses spas_sage2_attn_meansim_topk_cuda Triton kernel for VAE attention
- Maps VAE spatial attention (B, C, H, W) to sequence format (B, H, seq, D)
- Targets Attention blocks in MidBlock2D and UpDecoderBlock2D of the VAE
- VAE-specific tuning: num_warps=8, num_stages=2, block_m=64, block_n=64
  (num_stages=2 and block_m=64 for VAE BF16 to fit in Blackwell shared memory limit of 101376 bytes)
- DiT (Diffusion Transformer) keeps block_m=128, num_stages=3 for maximum speed with NVFP4/FP8
"""

import torch
import torch.nn.functional as F
import math
import logging
from typing import Optional, Tuple

# Configure logging
logger = logging.getLogger("SeedVR2.VAE.Blackwell")

# Track if we've logged kernel info once (avoid per-call logging overhead on Windows)
_vae_kernel_logged_once = False

# VAE-specific kernel parameters to fit within Blackwell shared memory limit (101376 bytes)
# DiT (Diffusion Transformer) uses block_m=128, num_stages=3 (from Blackwell config) with NVFP4/FP8
# VAE uses block_m=64, num_stages=2 to avoid shared memory "Out of Resource" errors on Blackwell with BF16
VAE_BLOCK_M = 64
VAE_NUM_STAGES = 2

# Import sparge attention functions
from .spas_sage_attn import (
    spas_sage2_attn_meansim_topk_cuda,
    SPARGE_LOCAL_AVAILABLE,
    get_blackwell_config,
)

# Default sparsity threshold for VAE attention
VAE_SPARSITY_THRESHOLD = 0.5


def set_vae_sparsity_threshold(threshold: float):
    """
    Set the global sparsity threshold for VAE attention.
    
    Args:
        threshold: Sparsity ratio (0.0, 1.0]. Lower = more sparsity = faster but less accurate.
                   Note: 0.0 (full sparsity) is not allowed as it would zero out attention.
                   - 0.3: Fast mode
                   - 0.5: Balanced mode (default)
                   - 0.7: Quality mode
    """
    global VAE_SPARSITY_THRESHOLD
    assert 0.0 < threshold <= 1.0, f"threshold must be in (0.0, 1.0], got {threshold}"
    VAE_SPARSITY_THRESHOLD = threshold


def get_vae_sparsity_threshold() -> float:
    """Get the current VAE sparsity threshold."""
    return VAE_SPARSITY_THRESHOLD


def is_vae_sparge_available() -> bool:
    """Check if VAE Sparge attention is available."""
    return SPARGE_LOCAL_AVAILABLE


def sparge_vae_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    topk: Optional[float] = None,
) -> torch.Tensor:
    """
    Compute VAE attention using Sparge Sage2 Triton kernels.
    
    This function is designed to be a drop-in replacement for standard attention
    in VAE decoder blocks. It maps the spatial attention tensors to the sequence
    format expected by the Sparge kernel.
    
    Uses block_m=64 and num_stages=2 specifically for VAE to fit within Blackwell
    shared memory limit (101376 bytes) when running in BF16/FP8.
    DiT uses block_m=128, num_stages=3 with NVFP4/FP8.
    
    Args:
        query: Query tensor (batch, heads, seq_len, head_dim) in HND layout
        key: Key tensor (batch, heads, seq_len, head_dim)
        value: Value tensor (batch, heads, seq_len, head_dim)
        attention_mask: Optional attention mask (not used with Sparge)
        scale: Softmax scale (default: 1/sqrt(head_dim))
        topk: Sparsity ratio (default: VAE_SPARSITY_THRESHOLD)
        
    Returns:
        Attention output tensor (batch, heads, seq_len, head_dim)
    """
    global _vae_kernel_logged_once
    
    if topk is None:
        topk = VAE_SPARSITY_THRESHOLD
    
    # Log kernel parameters only on first call
    if not _vae_kernel_logged_once:
        _vae_kernel_logged_once = True
        config = get_blackwell_config()
        if config.get('is_blackwell', False):
            logger.info(
                f"VAE Sparge Kernel: topk={topk}, Blackwell=True, "
                f"Warps={config.get('num_warps', 8)}, Stages={VAE_NUM_STAGES} (VAE-specific), "
                f"BlockM={VAE_BLOCK_M} (VAE-specific), BlockN={config.get('BLOCK_N', 64)}"
            )
    
    # Ensure tensors are contiguous for Triton kernels
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    
    # Check minimum sequence length for Sparge (requires >= 128)
    seq_len = query.size(-2)
    if seq_len < 128:
        # Fall back to standard attention for small sequences
        if scale is None:
            scale = query.size(-1) ** -0.5
        return F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, scale=scale)
    
    # Call Sparge Sage2 attention with VAE-specific parameters:
    # - block_m=64 and num_stages=2 to fit in Blackwell shared memory (101376 bytes limit)
    # - DiT uses block_m=128, num_stages=3 (from Blackwell config)
    output = spas_sage2_attn_meansim_topk_cuda(
        query, key, value,
        topk=topk,
        is_causal=False,
        scale=scale,
        smooth_k=True,
        tensor_layout="HND",
        output_dtype=query.dtype,
        block_m_override=VAE_BLOCK_M,       # Use block_m=64 for VAE (vs 128 for DiT)
        num_stages_override=VAE_NUM_STAGES,  # Use num_stages=2 for VAE (vs 3 for DiT)
    )
    
    return output


class SpargeVAEAttnProcessor:
    """
    Attention processor that uses Sparge Sage2 Triton kernels for VAE attention.
    
    This processor can be assigned to diffusers Attention modules to accelerate
    VAE decoding on Blackwell GPUs.
    
    Usage:
        from diffusers.models.attention_processor import Attention
        
        # Replace attention processor in VAE
        for module in vae.decoder.modules():
            if isinstance(module, Attention):
                module.set_processor(SpargeVAEAttnProcessor())
    """
    
    def __init__(self, topk: Optional[float] = None):
        """
        Initialize Sparge VAE attention processor.
        
        Args:
            topk: Sparsity ratio (0.0-1.0). If None, uses global VAE_SPARSITY_THRESHOLD
        """
        self.topk = topk
    
    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Process attention using Sparge Sage2 kernels.
        
        This method mirrors the signature of diffusers AttnProcessor2_0 but uses
        Sparge attention for accelerated computation.
        """
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        
        input_ndim = hidden_states.ndim
        
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])
        
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        
        query = attn.to_q(hidden_states)
        
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        
        topk = self.topk if self.topk is not None else VAE_SPARSITY_THRESHOLD
        
        # Use Sparge attention if available and sequence length is sufficient
        if is_vae_sparge_available() and sequence_length >= 128:
            hidden_states = sparge_vae_attention(
                query, key, value,
                attention_mask=attention_mask,
                topk=topk,
            )
        else:
            # Fall back to standard SDPA
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
        
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        
        # Linear projection
        hidden_states = attn.to_out[0](hidden_states)
        # Dropout
        hidden_states = attn.to_out[1](hidden_states)
        
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        
        hidden_states = hidden_states / attn.rescale_output_factor
        
        return hidden_states


def inject_sparge_into_vae(vae, topk: Optional[float] = None, debug=None) -> int:
    """
    Inject Sparge attention processors into a VAE model's decoder attention blocks.
    
    This function finds all Attention modules in the VAE decoder (MidBlock and UpBlocks)
    and replaces their attention processors with SpargeVAEAttnProcessor.
    
    Args:
        vae: VAE model (diffusers AutoencoderKL or custom VAE)
        topk: Sparsity ratio (0.0-1.0). If None, uses global VAE_SPARSITY_THRESHOLD
        debug: Optional debug instance for logging
        
    Returns:
        Number of attention modules that were patched
    """
    if not is_vae_sparge_available():
        if debug:
            debug.log(
                "Sparge attention not available for VAE - using standard attention",
                level="WARNING", category="vae", force=True
            )
        return 0
    
    patched_count = 0
    
    try:
        from diffusers.models.attention_processor import Attention
    except ImportError:
        if debug:
            debug.log(
                "diffusers.models.attention_processor not available",
                level="WARNING", category="vae", force=True
            )
        return 0
    
    processor = SpargeVAEAttnProcessor(topk=topk)
    
    # Patch decoder attention modules
    if hasattr(vae, 'decoder'):
        for name, module in vae.decoder.named_modules():
            if isinstance(module, Attention):
                try:
                    module.set_processor(processor)
                    patched_count += 1
                except Exception as e:
                    if debug:
                        debug.log(
                            f"Failed to patch attention module {name}: {e}",
                            level="WARNING", category="vae"
                        )
    
    # Also patch encoder attention modules (for symmetry)
    if hasattr(vae, 'encoder'):
        for name, module in vae.encoder.named_modules():
            if isinstance(module, Attention):
                try:
                    module.set_processor(processor)
                    patched_count += 1
                except Exception as e:
                    if debug:
                        debug.log(
                            f"Failed to patch encoder attention module {name}: {e}",
                            level="WARNING", category="vae"
                        )
    
    if patched_count > 0 and debug:
        threshold = topk if topk is not None else VAE_SPARSITY_THRESHOLD
        debug.log(
            f"Injected Sparge attention into {patched_count} VAE attention blocks (topk={threshold})",
            category="vae", force=True
        )
    
    return patched_count


def reset_vae_attention_logging():
    """Reset VAE attention logging state (call before each new generation)."""
    global _vae_kernel_logged_once
    _vae_kernel_logged_once = False
