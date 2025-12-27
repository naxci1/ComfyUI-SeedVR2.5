"""
VAE-optimized SageAttention 2.2.0 Integration

This module provides SageAttention-based attention processors specifically optimized
for VAE decoding operations. It includes:
- SageAttention 2.2.0 batched attention for VAE mid-block attention
- Memory-efficient attention with automatic dtype handling
- Fallback to PyTorch SDPA when SageAttention is unavailable

Target hardware: RTX 5070 Ti (16GB VRAM), PyTorch 2.7.1+cu128, CUDA 12.8
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# Import SageAttention availability flags
from .compatibility import SAGE_ATTN_2_AVAILABLE, SAGE_ATTN_3_AVAILABLE

# Import sageattention batched function if available
_sageattn = None
_sageattn_qk_int8_pv_fp16_cuda = None

try:
    from sageattention import sageattn as _sageattn_import
    _sageattn = _sageattn_import
except (ImportError, AttributeError, OSError):
    pass

# Try to import the optimized int8 kernel for even better performance
try:
    from sageattention import sageattn_qk_int8_pv_fp16_cuda as _sageattn_int8_import
    _sageattn_qk_int8_pv_fp16_cuda = _sageattn_int8_import
except (ImportError, AttributeError, OSError):
    pass


# Check if batched SageAttention is available (separate from varlen)
SAGE_ATTN_BATCHED_AVAILABLE = _sageattn is not None


@torch._dynamo.disable
def sage_attn_vae(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    SageAttention wrapper for VAE attention operations.
    
    Optimized for VAE decoder mid-block attention with:
    - Automatic dtype conversion to fp16/bf16 (SageAttention requirement)
    - Proper scale factor handling
    - Fallback to PyTorch SDPA when SageAttention unavailable
    
    Args:
        q: Query tensor (B, num_heads, seq_len, head_dim)
        k: Key tensor (B, num_heads, seq_len, head_dim)
        v: Value tensor (B, num_heads, seq_len, head_dim)
        is_causal: Whether to apply causal masking (default False for VAE)
        scale: Optional custom scale factor (default: 1/sqrt(head_dim))
        
    Returns:
        Attention output tensor (B, num_heads, seq_len, head_dim)
    """
    if not SAGE_ATTN_BATCHED_AVAILABLE:
        # Fallback to PyTorch scaled_dot_product_attention
        return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, scale=scale)
    
    # Store original dtype for output conversion
    out_dtype = q.dtype
    half_dtypes = (torch.float16, torch.bfloat16)
    
    # SageAttention requires half precision
    if q.dtype not in half_dtypes:
        q = q.to(torch.bfloat16)
        k = k.to(torch.bfloat16)
        v = v.to(torch.bfloat16)
    
    # Ensure matching dtypes
    if not (q.dtype == k.dtype == v.dtype):
        k = k.to(q.dtype)
        v = v.to(q.dtype)
    
    # Calculate scale if not provided
    if scale is None:
        scale = q.shape[-1] ** -0.5
    
    # Call SageAttention
    # sageattn expects (B, num_heads, seq_len, head_dim) layout
    out = _sageattn(q, k, v, is_causal=is_causal, scale=scale)
    
    # Convert back to original dtype if needed
    return out.to(out_dtype) if out.dtype != out_dtype else out


class SageAttentionVAEProcessor:
    """
    SageAttention processor for VAE attention layers.
    
    Drop-in replacement for diffusers' attention processors that uses
    SageAttention 2.2.0 kernels for improved performance.
    
    Memory optimization: Uses tiled processing for large spatial dimensions
    to prevent OOM on 16GB VRAM.
    """
    
    def __init__(
        self,
        tile_size: int = 64,
        use_tiling: bool = True,
        force_fp16: bool = True,
    ):
        """
        Initialize SageAttention VAE processor.
        
        Args:
            tile_size: Maximum spatial dimension before tiling (64 recommended for 16GB VRAM)
            use_tiling: Whether to use tiled processing for memory efficiency
            force_fp16: Force FP16 precision for attention (recommended for SageAttention)
        """
        self.tile_size = tile_size
        self.use_tiling = use_tiling
        self.force_fp16 = force_fp16
        self._use_sage = SAGE_ATTN_BATCHED_AVAILABLE
    
    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Process attention using SageAttention kernels.
        
        Args:
            attn: The attention module instance
            hidden_states: Input hidden states (B, C, H, W) for spatial attention
            encoder_hidden_states: Optional cross-attention input
            attention_mask: Optional attention mask
            temb: Optional time embedding
            **kwargs: Additional arguments
            
        Returns:
            Processed hidden states
        """
        residual = hidden_states
        
        # Handle input shape - VAE attention expects (B, C, H, W) -> (B, H*W, C)
        batch_size, channels, height, width = hidden_states.shape
        seq_len = height * width
        
        # Apply group norm if present
        if hasattr(attn, 'group_norm') and attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states)
        
        # Reshape for attention: (B, C, H, W) -> (B, H*W, C)
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch_size, seq_len, channels)
        
        # Project to Q, K, V
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states if encoder_hidden_states is None else encoder_hidden_states)
        value = attn.to_v(hidden_states if encoder_hidden_states is None else encoder_hidden_states)
        
        # Reshape for multi-head attention
        head_dim = attn.dim_head if hasattr(attn, 'dim_head') else channels // attn.heads
        
        query = query.view(batch_size, seq_len, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        
        # Apply tiled attention if sequence length is large
        if self.use_tiling and seq_len > self.tile_size * self.tile_size:
            hidden_states = self._tiled_attention(query, key, value, self.tile_size)
        else:
            # Standard attention with SageAttention
            hidden_states = sage_attn_vae(query, key, value, is_causal=False)
        
        # Reshape back: (B, heads, seq, dim) -> (B, seq, C)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, seq_len, channels)
        
        # Output projection
        hidden_states = attn.to_out[0](hidden_states)
        if len(attn.to_out) > 1:
            hidden_states = attn.to_out[1](hidden_states)  # dropout if present
        
        # Reshape back to spatial: (B, H*W, C) -> (B, C, H, W)
        hidden_states = hidden_states.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)
        
        # Add residual if configured
        if hasattr(attn, 'residual_connection') and attn.residual_connection:
            hidden_states = hidden_states + residual
        
        # Apply rescale factor if present
        if hasattr(attn, 'rescale_output_factor') and attn.rescale_output_factor != 1.0:
            hidden_states = hidden_states / attn.rescale_output_factor
        
        return hidden_states
    
    def _tiled_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        tile_size: int,
    ) -> torch.Tensor:
        """
        Memory-efficient tiled attention for large spatial dimensions.
        
        Splits the attention computation into tiles to prevent OOM errors
        on GPUs with limited VRAM (16GB target).
        
        Args:
            query: Query tensor (B, heads, seq, dim)
            key: Key tensor (B, heads, seq, dim)
            value: Value tensor (B, heads, seq, dim)
            tile_size: Size of each tile
            
        Returns:
            Attention output tensor (B, heads, seq, dim)
        """
        batch_size, num_heads, seq_len, head_dim = query.shape
        tile_seq = tile_size * tile_size
        
        # If sequence fits in one tile, process normally
        if seq_len <= tile_seq:
            return sage_attn_vae(query, key, value, is_causal=False)
        
        # Split into tiles
        num_tiles = (seq_len + tile_seq - 1) // tile_seq
        outputs = []
        
        for i in range(num_tiles):
            start_idx = i * tile_seq
            end_idx = min((i + 1) * tile_seq, seq_len)
            
            q_tile = query[:, :, start_idx:end_idx, :]
            
            # For self-attention, we need full K, V for accurate results
            # But this is memory-intensive. Use sliding window approximation
            # for memory efficiency on 16GB VRAM
            window_size = min(tile_seq * 2, seq_len)
            k_start = max(0, start_idx - tile_seq // 2)
            k_end = min(seq_len, end_idx + tile_seq // 2)
            
            k_tile = key[:, :, k_start:k_end, :]
            v_tile = value[:, :, k_start:k_end, :]
            
            # Compute tiled attention
            out_tile = sage_attn_vae(q_tile, k_tile, v_tile, is_causal=False)
            outputs.append(out_tile)
        
        return torch.cat(outputs, dim=2)


class SageAttentionBlock(nn.Module):
    """
    SageAttention-based attention block for VAE.
    
    Drop-in replacement for the standard AttentionBlock in wan2_1_vae.py
    that uses SageAttention 2.2.0 kernels for improved performance.
    """
    
    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        use_tiling: bool = True,
        tile_size: int = 64,
    ):
        super().__init__()
        
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.use_tiling = use_tiling
        self.tile_size = tile_size
        
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        
        self.norm = nn.GroupNorm(num_groups=min(32, channels), num_channels=channels)
        self.qkv = nn.Linear(channels, channels * 3, bias=True)
        self.proj = nn.Linear(channels, channels, bias=True)
        self.scale = self.head_dim ** -0.5
        
        self._use_sage = SAGE_ATTN_BATCHED_AVAILABLE
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        seq_len = height * width
        
        # Reshape and normalize
        x_norm = self.norm(x)
        x_flat = x_norm.permute(0, 2, 3, 1).reshape(batch_size, seq_len, channels)
        
        # Project to Q, K, V
        qkv = self.qkv(x_flat)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, seq, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Apply SageAttention or tiled attention based on sequence length
        if self.use_tiling and seq_len > self.tile_size * self.tile_size:
            out = self._tiled_sage_attention(q, k, v)
        else:
            out = sage_attn_vae(q, k, v, is_causal=False, scale=self.scale)
        
        # Reshape and project
        out = out.transpose(1, 2).reshape(batch_size, seq_len, channels)
        out = self.proj(out)
        
        # Reshape back and add residual
        out = out.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)
        out = out + x
        
        return out
    
    def _tiled_sage_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Memory-efficient tiled attention for large spatial dimensions."""
        batch_size, num_heads, seq_len, head_dim = q.shape
        tile_seq = self.tile_size * self.tile_size
        
        if seq_len <= tile_seq:
            return sage_attn_vae(q, k, v, is_causal=False, scale=self.scale)
        
        num_tiles = (seq_len + tile_seq - 1) // tile_seq
        outputs = []
        
        for i in range(num_tiles):
            start_idx = i * tile_seq
            end_idx = min((i + 1) * tile_seq, seq_len)
            
            q_tile = q[:, :, start_idx:end_idx, :]
            
            # Sliding window for K, V
            k_start = max(0, start_idx - tile_seq // 2)
            k_end = min(seq_len, end_idx + tile_seq // 2)
            
            k_tile = k[:, :, k_start:k_end, :]
            v_tile = v[:, :, k_start:k_end, :]
            
            out_tile = sage_attn_vae(q_tile, k_tile, v_tile, is_causal=False, scale=self.scale)
            outputs.append(out_tile)
        
        return torch.cat(outputs, dim=2)


def create_sage_attention_block(
    channels: int,
    num_heads: int = 4,
    use_tiling: bool = True,
    tile_size: int = 64,
) -> SageAttentionBlock:
    """
    Factory function to create a SageAttention block for VAE.
    
    Args:
        channels: Number of input/output channels
        num_heads: Number of attention heads
        use_tiling: Whether to use tiled processing for memory efficiency
        tile_size: Maximum spatial dimension before tiling
        
    Returns:
        SageAttentionBlock instance
    """
    return SageAttentionBlock(
        channels=channels,
        num_heads=num_heads,
        use_tiling=use_tiling,
        tile_size=tile_size,
    )


def get_sage_attention_status() -> dict:
    """
    Get status of SageAttention availability.
    
    Returns:
        Dictionary with availability flags and version info
    """
    return {
        "sage_attn_2_available": SAGE_ATTN_2_AVAILABLE,
        "sage_attn_3_available": SAGE_ATTN_3_AVAILABLE,
        "sage_attn_batched_available": SAGE_ATTN_BATCHED_AVAILABLE,
        "using_sage": SAGE_ATTN_BATCHED_AVAILABLE,
        "fallback": "pytorch_sdpa" if not SAGE_ATTN_BATCHED_AVAILABLE else None,
    }
