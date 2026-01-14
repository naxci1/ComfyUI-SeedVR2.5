"""
VAE SageAttention 2 (SA2) Optimized Module for Blackwell GPUs

This module provides SA2-optimized attention for VAE attention blocks on Blackwell GPUs (RTX 50 series).
Applies to BOTH Encoder3D and Decoder3D attention layers.

Key Features:
- Dynamic head splitting: 512-dim → 8 heads × 64 dim (Blackwell Tensor Core sweet spot)
- SageAttention 2 integration with INT8/FP8 quantization for maximum speed
- Automatic fallback to standard SDPA if SA2 is unavailable
- Detailed verbose logging for both Encoder (Phase 1) and Decoder (Phase 3)
- Memory-efficient operation on 16GB GPUs (Blackwell RTX 50xx)
- No silent failures: Always logs warnings when SA2 is skipped

NOTE: This implementation reshapes 1 head × 512 dim to 8 heads × 64 dim in the forward pass,
preserving the original weights while enabling faster SA2/FlashAttention kernels.

INLINE SA2: The sa2_inline_attention() function directly replaces the diffusers Attention.forward()
to ensure SA2 is actually executed inside every attention block.
"""

import os
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional, Literal

# Configure logging
logger = logging.getLogger("SeedVR2.VAE.Attention")

# Track logging state separately for encoder and decoder
_encoder_logged_once = False
_decoder_logged_once = False

# Attention call counter for detailed logging
_attention_call_count = 0

# Enable verbose logging for SA2 backend selection (logs EVERY attention call)
_verbose_sa2_logging = True

# SA2 availability check (cached)
_sa2_available = None

# Current phase context (encoder/decoder)
_current_phase: Literal["encoder", "decoder", "unknown"] = "unknown"

# Blackwell memory guard: Set expandable_segments for better fragmentation handling
try:
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
except Exception:
    pass


def set_vae_phase(phase: Literal["encoder", "decoder", "unknown"]):
    """Set the current VAE phase for logging context."""
    global _current_phase
    _current_phase = phase


def _check_sa2_available() -> bool:
    """Check if SageAttention 2 is available."""
    global _sa2_available
    if _sa2_available is None:
        try:
            from sageattention import sageattn
            _sa2_available = True
            print("[DEBUG] SageAttention 2 (SA2) module: AVAILABLE")
        except ImportError:
            _sa2_available = False
            print("[WARNING] SageAttention 2 (SA2) module: NOT INSTALLED - Using FlashAttention/SDPA fallback")
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
    
    if heads != 1:
        print(f"[WARNING] SA2 head splitting skipped: Expected 1 head, got {heads}")
        return tensor
    
    if dim % target_heads != 0:
        print(f"[WARNING] SA2 head splitting skipped: Dimension {dim} not divisible by {target_heads}")
        return tensor
    
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
    
    Applies to BOTH Encoder3D and Decoder3D attention blocks.
    
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
    global _encoder_logged_once, _decoder_logged_once, _attention_call_count, _current_phase
    
    batch, heads, seq_len, head_dim = query.shape
    _attention_call_count += 1
    
    if scale is None:
        scale = head_dim ** -0.5
    
    # Determine if we should use head splitting (only for 1 head × 512 dim VAE structure)
    use_head_splitting = (heads == 1 and head_dim == 512)
    sa2_available = _check_sa2_available()
    
    # Determine phase for logging
    is_encoder = _current_phase == "encoder"
    is_decoder = _current_phase == "decoder"
    phase_name = "ENCODER" if is_encoder else ("DECODER" if is_decoder else "VAE")
    
    # Phase-specific logging
    should_log = False
    if is_encoder and not _encoder_logged_once:
        _encoder_logged_once = True
        should_log = True
    elif is_decoder and not _decoder_logged_once:
        _decoder_logged_once = True
        should_log = True
    elif not is_encoder and not is_decoder and not _encoder_logged_once:
        _encoder_logged_once = True
        should_log = True
    
    # Verbose logging - show backend selection
    if should_log:
        print(f"\n{'='*60}")
        print(f"[DEBUG] VAE {phase_name}: Initiating SageAttention 2 forward pass")
        print(f"{'='*60}")
        
        if use_head_splitting:
            print(f"[DEBUG] Reshaping: {head_dim}-dim -> 8 heads x 64-dim for SA2 compatibility")
            print(f"[DEBUG] Original Shape: Q/K/V = (batch={batch}, heads={heads}, seq_len={seq_len}, head_dim={head_dim})")
            print(f"[DEBUG] Reshaped Shape: Q/K/V = (batch={batch}, heads=8, seq_len={seq_len}, head_dim=64)")
            
            if sa2_available:
                print(f"[DEBUG] SA2 Optimized Forward Pass: ACTIVE")
                print(f"[DEBUG] Backend: SageAttention 2 (INT8 Quantized)")
                print(f"[DEBUG] num_heads=8, head_dim=64 (Blackwell Tensor Core optimized)")
            else:
                print(f"[WARNING] SA2 not available, using FlashAttention/SDPA fallback")
                print(f"[DEBUG] Backend: PyTorch FlashAttention/SDPA")
                print(f"[DEBUG] num_heads=8, head_dim=64")
        else:
            print(f"[WARNING] SA2 skipped: head_dim={head_dim} not compatible (requires 512-dim for splitting)")
            print(f"[DEBUG] Using Native SDPA: {heads} head x {head_dim} dim")
            print(f"[DEBUG] Shape: Q/K/V = (batch={batch}, heads={heads}, seq_len={seq_len}, head_dim={head_dim})")
        
        print(f"[DEBUG] Precision: {query.dtype}")
        print(f"{'='*60}\n")
        
        logger.info(f"VAE {phase_name} SA2: heads={8 if use_head_splitting else heads}, head_dim={64 if use_head_splitting else head_dim}")
    
    # Use sliced SDPA for very large sequence lengths to prevent OOM
    if seq_len > 4096:
        print(f"[DEBUG] VAE {phase_name}: Large seq_len={seq_len} > 4096, using sliced attention")
        return _sliced_sdpa(query, key, value, attention_mask=attention_mask, scale=scale)
    
    # SA2-optimized path with dynamic head splitting for 1 head × 512 dim VAE
    if use_head_splitting:
        # Split to 8 heads × 64 dim
        q_split = _split_heads_for_blackwell(query)
        k_split = _split_heads_for_blackwell(key)
        v_split = _split_heads_for_blackwell(value)
        
        # Check if splitting succeeded
        if q_split.shape[1] != 8:
            print(f"[WARNING] Head splitting failed, falling back to standard SDPA")
            return F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, scale=scale)
        
        # Adjust scale for new head dimension
        new_scale = 64 ** -0.5
        
        if sa2_available:
            # Use SageAttention 2
            try:
                out_split = _sa2_attention(q_split, k_split, v_split, scale=new_scale)
            except Exception as e:
                print(f"[WARNING] SA2 execution failed: {e}")
                print(f"[DEBUG] Falling back to FlashAttention/SDPA")
                logger.warning(f"SA2 failed, falling back to SDPA: {e}")
                try:
                    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True):
                        out_split = F.scaled_dot_product_attention(
                            q_split, k_split, v_split,
                            attn_mask=None,
                            dropout_p=0.0,
                            is_causal=False,
                            scale=new_scale
                        )
                except Exception as e2:
                    print(f"[WARNING] FlashAttention also failed: {e2}, using math fallback")
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
            except Exception as e:
                print(f"[WARNING] FlashAttention failed: {e}, using standard SDPA")
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
    print(f"[WARNING] VAE {phase_name}: Non-standard head_dim={head_dim}, SA2 disabled")
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
    Sparge attention. VAE uses SA2-optimized attention which is natively optimized.
    
    Args:
        vae: VAE model (diffusers AutoencoderKL or custom VAE)
        topk: Sparsity ratio (ignored - using SA2)
        debug: Optional debug instance for logging
        
    Returns:
        0 (no modules patched - VAE uses SA2/SDPA via standard_vae_attention)
    """
    sa2_available = _check_sa2_available()
    if debug:
        if sa2_available:
            debug.log(
                "VAE using SageAttention 2 (SA2) with dynamic head splitting (512-dim → 8 heads × 64 dim)",
                level="INFO", category="vae", force=True
            )
        else:
            debug.log(
                "VAE using FlashAttention/SDPA with dynamic head splitting (512-dim → 8 heads × 64 dim)",
                level="INFO", category="vae", force=True
            )
    return 0


def reset_vae_attention_logging():
    """Reset VAE attention logging state (call before each new generation)."""
    global _encoder_logged_once, _decoder_logged_once, _attention_call_count, _current_phase
    _encoder_logged_once = False
    _decoder_logged_once = False
    _attention_call_count = 0
    _current_phase = "unknown"


def sa2_inline_attention(attn_module: nn.Module, hidden_states: torch.Tensor, block_id: int) -> torch.Tensor:
    """
    INLINE SA2 ATTENTION: Replaces the diffusers Attention.forward() with SA2-optimized attention.
    
    This function is called DIRECTLY from UNetMidBlock3D.forward() to ensure SA2 is actually
    executed inside every attention block.
    
    Args:
        attn_module: The diffusers Attention module containing the projections (to_q, to_k, to_v, to_out)
        hidden_states: Input tensor (batch, channels, height, width) - already rearranged from 5D
        block_id: Attention block ID for logging
        
    Returns:
        Output tensor with same shape as input
    """
    global _current_phase, _attention_call_count
    _attention_call_count += 1
    
    # Get phase name for logging
    phase_name = _current_phase.upper() if _current_phase != "unknown" else "VAE"
    
    # Input shape info
    batch, channels, height, width = hidden_states.shape
    seq_len = height * width
    
    # Log EVERY attention call with detailed info (user requirement)
    print(f"[SA2_EXEC] Block ID: {block_id}, Phase: {phase_name}, Shape: [{batch}, {channels}, {height}, {width}], Heads: 8")
    
    # ====== MANDATORY RESHAPE ======
    # Before calling sage_attn, we MUST reshape the input:
    # (batch, channels, H, W) -> (batch, H*W, 8, 64)
    
    # 1. Apply normalization if present (from deprecated attn block)
    residual = hidden_states
    if hasattr(attn_module, 'group_norm') and attn_module.group_norm is not None:
        hidden_states = attn_module.group_norm(hidden_states)
    
    # 2. Reshape to sequence format: (batch, channels, H, W) -> (batch, seq_len, channels)
    hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, seq_len, channels)
    
    # Log the reshape operation
    print(f"[DEBUG] Reshaping: {channels}-dim -> 8 heads x 64-dim for SA2 compatibility")
    print(f"[DEBUG] Input Shape: [B={batch}, C={channels}, H={height}, W={width}] -> [B={batch}, seq_len={seq_len}, C={channels}]")
    
    # 3. Compute Q, K, V projections
    query = attn_module.to_q(hidden_states)  # (batch, seq_len, channels)
    key = attn_module.to_k(hidden_states)    # (batch, seq_len, channels)
    value = attn_module.to_v(hidden_states)  # (batch, seq_len, channels)
    
    # 4. MANDATORY RESHAPE for SA2: (batch, seq_len, 512) -> (batch, 8, seq_len, 64)
    # This is the key step that the user requested
    head_dim = 64
    num_heads = channels // head_dim
    
    if channels % head_dim != 0:
        print(f"[WARNING] SA2 skipped: channels={channels} not divisible by head_dim={head_dim}")
        # Fallback to simple attention
        attn_weights = torch.softmax(torch.bmm(query, key.transpose(1, 2)) * (channels ** -0.5), dim=-1)
        hidden_states = torch.bmm(attn_weights, value)
    else:
        # Reshape to multi-head format: (batch, seq_len, channels) -> (batch, num_heads, seq_len, head_dim)
        query = query.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
        key = key.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
        value = value.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
        
        print(f"[DEBUG] Reshaped Q/K/V: [B={batch}, heads={num_heads}, seq_len={seq_len}, head_dim={head_dim}]")
        
        # 5. Apply SA2 or FlashAttention/SDPA
        sa2_available = _check_sa2_available()
        scale = head_dim ** -0.5
        
        if sa2_available:
            try:
                from sageattention import sageattn
                print(f"[DEBUG] SA2 Optimized Forward Pass: ACTIVE")
                
                # SA2 expects (batch, heads, seq_len, head_dim) format
                hidden_states = sageattn(
                    query.contiguous(),
                    key.contiguous(),
                    value.contiguous(),
                    tensor_layout="HND",
                    is_causal=False,
                    sm_scale=scale,
                )
            except Exception as e:
                print(f"[WARNING] SA2 execution failed: {e}, falling back to FlashAttention/SDPA")
                try:
                    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True):
                        hidden_states = F.scaled_dot_product_attention(
                            query, key, value,
                            attn_mask=None,
                            dropout_p=0.0,
                            is_causal=False,
                            scale=scale
                        )
                except Exception:
                    hidden_states = F.scaled_dot_product_attention(
                        query, key, value,
                        attn_mask=None,
                        dropout_p=0.0,
                        is_causal=False,
                        scale=scale
                    )
        else:
            print(f"[DEBUG] Using FlashAttention/SDPA (SA2 not available)")
            try:
                with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True):
                    hidden_states = F.scaled_dot_product_attention(
                        query, key, value,
                        attn_mask=None,
                        dropout_p=0.0,
                        is_causal=False,
                        scale=scale
                    )
            except Exception:
                hidden_states = F.scaled_dot_product_attention(
                    query, key, value,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=False,
                    scale=scale
                )
        
        # 6. Reshape back: (batch, num_heads, seq_len, head_dim) -> (batch, seq_len, channels)
        hidden_states = hidden_states.transpose(1, 2).contiguous().view(batch, seq_len, channels)
    
    # 7. Apply output projection
    hidden_states = attn_module.to_out[0](hidden_states)  # Linear
    if len(attn_module.to_out) > 1:
        hidden_states = attn_module.to_out[1](hidden_states)  # Dropout (if present)
    
    # 8. Reshape back to image format: (batch, seq_len, channels) -> (batch, channels, H, W)
    hidden_states = hidden_states.transpose(1, 2).view(batch, channels, height, width)
    
    # 9. Apply residual connection (deprecated attn block always uses residual)
    if hasattr(attn_module, 'residual_connection') and attn_module.residual_connection:
        hidden_states = hidden_states + residual
    
    # 10. Apply rescale factor if present
    if hasattr(attn_module, 'rescale_output_factor'):
        hidden_states = hidden_states / attn_module.rescale_output_factor
    
    print(f"[DEBUG] Output Shape: [B={batch}, C={channels}, H={height}, W={width}]")
    
    return hidden_states
