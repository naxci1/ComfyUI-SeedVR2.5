"""
VAE Sparse Attention Processor for Blackwell GPUs

This module provides a SpargeAttnProcessor that can be injected into VAE Attention blocks
to accelerate decoding using block-sparse attention patterns. Optimized for NVIDIA Blackwell
(RTX 50xx, SM 12.0) GPUs.

Key Features:
- Monkey-patches diffusers.models.attention_processor.Attention.forward
- Uses Sparge+SageAttention 2 Triton kernels for VAE attention
- Dynamic head splitting: 512-dim → 8 heads × 64 dim in forward pass (preserves weights)
- Sparge Top-K selection (threshold=0.3) zeros out unimportant pixels to save VRAM
- SageAttention 2 with INT8/FP8 quantization for Blackwell Tensor Cores
- VAE-specific tuning: num_warps=8, num_stages=2, block_m=64, block_n=64
  (num_stages=2 and block_m=64 for VAE BF16 to fit in Blackwell shared memory limit of 101376 bytes)
- DiT (Diffusion Transformer) keeps block_m=128, num_stages=3 for maximum speed with NVFP4/FP8
- Blackwell sync: torch.cuda.synchronize() after Sparge mask calculation to prevent scheduler hang
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
# Track if we've logged the head_dim fallback warning (avoid spam)
_vae_head_dim_logged = False
# Track if we've logged the kernel error fallback warning (avoid spam)
_vae_kernel_fallback_logged = False
# Track if we've logged head splitting (avoid spam)
_vae_head_split_logged = False

# VAE-specific kernel parameters to fit within Blackwell shared memory limit (101376 bytes)
# DiT (Diffusion Transformer) uses block_m=128, num_stages=3 (from Blackwell config) with NVFP4/FP8
# VAE uses block_m=64, num_stages=2 to avoid shared memory "Out of Resource" errors on Blackwell with BF16
VAE_BLOCK_M = 64
VAE_NUM_STAGES = 2

# Target head dimension for Blackwell Tensor Cores (optimal for FlashAttention/Sparge/SageAttn)
BLACKWELL_TARGET_HEAD_DIM = 64

# Import sparge attention functions
from .spas_sage_attn import (
    spas_sage2_attn_meansim_topk_cuda,
    SPARGE_LOCAL_AVAILABLE,
    get_blackwell_config,
)

# Import SageAttention 2 if available
try:
    from .compatibility import sageattn_varlen, SAGE_ATTN_2_AVAILABLE
except ImportError:
    sageattn_varlen = None
    SAGE_ATTN_2_AVAILABLE = False

# Default sparsity threshold for VAE attention - 0.3 for fast mode (saves massive VRAM)
# Lower = more sparsity = faster but less accurate
# - 0.3: Fast mode (recommended for 16GB GPUs like RTX 5070 Ti)
# - 0.5: Balanced mode
# - 0.7: Quality mode
VAE_SPARSITY_THRESHOLD = 0.3


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
    
    Processes attention in slices to avoid OOM on large inputs.
    
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


def _split_heads_for_blackwell(
    tensor: torch.Tensor,
    original_head_dim: int,
    target_head_dim: int = BLACKWELL_TARGET_HEAD_DIM,
) -> Tuple[torch.Tensor, int, int]:
    """
    Split large head dimensions into multiple smaller heads for Blackwell compatibility.
    
    This preserves the original weight structure by reshaping in the forward pass.
    E.g., 1 head × 512 dim → 8 heads × 64 dim
    
    Args:
        tensor: Input tensor (batch, heads, seq_len, head_dim)
        original_head_dim: Original head dimension
        target_head_dim: Target head dimension for Blackwell (default: 64)
        
    Returns:
        Tuple of (reshaped tensor, new_heads, new_head_dim)
    """
    batch, heads, seq_len, head_dim = tensor.shape
    
    if head_dim <= target_head_dim or head_dim % target_head_dim != 0:
        return tensor, heads, head_dim
    
    split_factor = head_dim // target_head_dim
    new_heads = heads * split_factor
    
    # Reshape: (B, H, S, D) -> (B, H, S, split, target) -> (B, H*split, S, target)
    tensor = tensor.view(batch, heads, seq_len, split_factor, target_head_dim)
    tensor = tensor.permute(0, 1, 3, 2, 4).contiguous()  # (B, H, split, S, target)
    tensor = tensor.view(batch, new_heads, seq_len, target_head_dim)
    
    return tensor, new_heads, target_head_dim


def _merge_heads_from_blackwell(
    tensor: torch.Tensor,
    original_heads: int,
    original_head_dim: int,
) -> torch.Tensor:
    """
    Merge split heads back to original structure.
    
    E.g., 8 heads × 64 dim → 1 head × 512 dim
    
    Args:
        tensor: Input tensor (batch, new_heads, seq_len, target_head_dim)
        original_heads: Original number of heads
        original_head_dim: Original head dimension
        
    Returns:
        Reshaped tensor (batch, original_heads, seq_len, original_head_dim)
    """
    batch, new_heads, seq_len, target_head_dim = tensor.shape
    
    if new_heads == original_heads:
        return tensor
    
    split_factor = new_heads // original_heads
    
    # Reshape: (B, H*split, S, target) -> (B, H, split, S, target) -> (B, H, S, D)
    tensor = tensor.view(batch, original_heads, split_factor, seq_len, target_head_dim)
    tensor = tensor.permute(0, 1, 3, 2, 4).contiguous()  # (B, H, S, split, target)
    tensor = tensor.view(batch, original_heads, seq_len, original_head_dim)
    
    return tensor


def sparge_vae_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    topk: Optional[float] = None,
) -> torch.Tensor:
    """
    Compute VAE attention using Sparge + SageAttention 2 for Blackwell GPUs.
    
    This function implements:
    1. Dynamic head splitting: 512-dim → 8 heads × 64 dim (preserves weights)
    2. Sparge Top-K selection (threshold=0.3) to zero out unimportant pixels
    3. SageAttention 2 with INT8/FP8 quantization for Blackwell Tensor Cores
    4. torch.cuda.synchronize() after Sparge mask to prevent scheduler hang
    
    Args:
        query: Query tensor (batch, heads, seq_len, head_dim) in HND layout
        key: Key tensor (batch, heads, seq_len, head_dim)
        value: Value tensor (batch, heads, seq_len, head_dim)
        attention_mask: Optional attention mask (not used with Sparge)
        scale: Softmax scale (default: 1/sqrt(head_dim))
        topk: Sparsity ratio (default: VAE_SPARSITY_THRESHOLD = 0.3)
        
    Returns:
        Attention output tensor (batch, heads, seq_len, head_dim)
    """
    global _vae_kernel_logged_once
    global _vae_head_dim_logged
    global _vae_kernel_fallback_logged
    global _vae_head_split_logged
    
    if topk is None:
        topk = VAE_SPARSITY_THRESHOLD
    
    # Store original dimensions for later merging
    batch, original_heads, seq_len, original_head_dim = query.shape
    
    # Dynamic head splitting for Blackwell compatibility
    # Sparge/SageAttn requires head_dim in [64, 128]
    # SeedVR2 VAE has head_dim=512 → split to 8 heads × 64 dim
    if original_head_dim > 128 and original_head_dim % BLACKWELL_TARGET_HEAD_DIM == 0:
        if not _vae_head_split_logged:
            _vae_head_split_logged = True
            split_factor = original_head_dim // BLACKWELL_TARGET_HEAD_DIM
            new_heads = original_heads * split_factor
            logger.info(
                f"VAE Blackwell optimization: Splitting {original_heads} heads × {original_head_dim} dim "
                f"→ {new_heads} heads × {BLACKWELL_TARGET_HEAD_DIM} dim for Sparge/SageAttn compatibility"
            )
        
        query, new_heads, new_head_dim = _split_heads_for_blackwell(query, original_head_dim)
        key, _, _ = _split_heads_for_blackwell(key, original_head_dim)
        value, _, _ = _split_heads_for_blackwell(value, original_head_dim)
        
        # Update scale for new head_dim
        if scale is None:
            scale = new_head_dim ** -0.5
        
        head_dim_for_kernel = new_head_dim
    else:
        head_dim_for_kernel = original_head_dim
        if scale is None:
            scale = original_head_dim ** -0.5
    
    # Check head_dim compatibility with Sparge kernel
    if head_dim_for_kernel not in [64, 128]:
        if not _vae_head_dim_logged:
            _vae_head_dim_logged = True
            logger.info(
                f"VAE head_dim={head_dim_for_kernel} not in [64, 128]. Using memory-efficient sliced SDPA."
            )
        output = _sliced_sdpa(query, key, value, attention_mask=attention_mask, scale=scale)
        
        # Merge heads back if we split them
        if original_head_dim != head_dim_for_kernel:
            output = _merge_heads_from_blackwell(output, original_heads, original_head_dim)
        return output
    
    # Log kernel parameters only on first call
    if not _vae_kernel_logged_once:
        _vae_kernel_logged_once = True
        config = get_blackwell_config()
        if config.get('is_blackwell', False):
            logger.info(
                f"VAE Sparge+SA2 Kernel: topk={topk}, Blackwell=True, "
                f"Warps={config.get('num_warps', 8)}, Stages={VAE_NUM_STAGES}, "
                f"BlockM={VAE_BLOCK_M}, head_dim={head_dim_for_kernel}"
            )
    
    # Ensure tensors are contiguous for Triton kernels
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    
    # Check minimum sequence length for Sparge (requires >= 128)
    if seq_len < 128:
        output = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, scale=scale)
        if original_head_dim != head_dim_for_kernel:
            output = _merge_heads_from_blackwell(output, original_heads, original_head_dim)
        return output
    
    # Try Sparge + SageAttention 2 pipeline
    try:
        # Call Sparge Sage2 attention with VAE-specific parameters
        output = spas_sage2_attn_meansim_topk_cuda(
            query, key, value,
            topk=topk,
            is_causal=False,
            scale=scale,
            smooth_k=True,
            tensor_layout="HND",
            output_dtype=query.dtype,
            block_m_override=VAE_BLOCK_M,
            num_stages_override=VAE_NUM_STAGES,
        )
        
        # Blackwell sync: Force GPU to release locked memory after Sparge mask calculation
        # This prevents scheduler hang on RTX 5070 Ti at end of batch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
    except Exception as e:
        # Fall back to sliced SDPA on any error
        if not _vae_kernel_fallback_logged:
            _vae_kernel_fallback_logged = True
            logger.warning(
                f"Sparge VAE kernel failed, falling back to sliced SDPA: {e}. "
                f"This warning will not repeat."
            )
        output = _sliced_sdpa(query, key, value, attention_mask=attention_mask, scale=scale)
    
    # Merge heads back to original structure if we split them
    if original_head_dim != head_dim_for_kernel:
        output = _merge_heads_from_blackwell(output, original_heads, original_head_dim)
    
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
    
    NOTE: This function should only be called if enable_sparge_attention is True.
    For SeedVR2 VAE (head_dim=512), Sparge kernel is incompatible and will fall back
    to sliced SDPA, which may cause memory leaks. It is recommended to keep
    enable_sparge_attention=False for best stability.
    
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
                "Sparge attention not available for VAE - using standard SDPA (recommended)",
                level="INFO", category="vae", force=True
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
    global _vae_kernel_logged_once, _vae_head_dim_logged, _vae_kernel_fallback_logged, _vae_head_split_logged
    _vae_kernel_logged_once = False
    _vae_head_dim_logged = False
    _vae_kernel_fallback_logged = False
    _vae_head_split_logged = False
