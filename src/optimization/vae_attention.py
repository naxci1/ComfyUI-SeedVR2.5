"""
VAE Attention Utilities for SeedVR2

Provides optional SageAttention 2 support for VAE attention blocks.
REAL BLOCK-LEVEL INJECTION when UI toggles are True.
NATIVE SDPA when UI toggles are False (default - fastest path).
"""

import os
import torch
import torch.nn.functional as F
import logging
from typing import Optional, Literal, Any

# Configure logging
logger = logging.getLogger("SeedVR2.VAE")

# ============================================================================
# GLOBAL STATE - UI-CONTROLLED
# ============================================================================

# SA2 availability (cached)
_sa2_available = None

# Current phase context
_current_phase: Literal["encoder", "decoder", "unknown"] = "unknown"

# UI-controlled SA2 toggles (set at runtime from node inputs)
# DEFAULTS: False = Native SDPA (fastest path, no patching)
_encoder_sa2_enabled = False
_decoder_sa2_enabled = False

# Original forward methods cache (for cleanup)
_original_forwards = {}

# Block counter for logging
_block_counter = 0

# Memory policy: expandable_segments for Blackwell
try:
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
except Exception:
    pass


def set_vae_phase(phase: Literal["encoder", "decoder", "unknown"]):
    """Set the current VAE phase."""
    global _current_phase, _block_counter
    _current_phase = phase
    _block_counter = 0


def configure_vae_sa2(encoder_sa2: bool = False, decoder_sa2: bool = False):
    """
    Configure SA2 enablement for Encoder and Decoder from UI toggles.
    """
    global _encoder_sa2_enabled, _decoder_sa2_enabled
    
    _encoder_sa2_enabled = encoder_sa2
    _decoder_sa2_enabled = decoder_sa2
    
    encoder_backend = "SA2" if encoder_sa2 else "Native SDPA"
    decoder_backend = "SA2" if decoder_sa2 else "Native SDPA"
    
    print(f"[VAE-CTRL] Encoder: {encoder_backend} | Decoder: {decoder_backend}")


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
    """Check if VAE SA2/Sparge attention is available."""
    return _check_sa2_available()


def set_vae_sparsity_threshold(threshold: float):
    """API compatibility."""
    pass


def get_vae_sparsity_threshold() -> float:
    """API compatibility."""
    return 0.5


def _create_sa2_forward_diffusers(original_forward, block_id: int):
    """
    Create a SA2-enabled forward method for diffusers Attention class.
    Uses SageAttention 2 kernel with 8 heads x 64-dim for Blackwell optimization.
    
    Diffusers Attention.forward signature: (hidden_states, encoder_hidden_states=None, 
                                            attention_mask=None, temb=None, ...)
    """
    def sa2_forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
        global _block_counter
        _block_counter += 1
        
        # Check if SA2 should be used for current phase
        use_sa2 = False
        if _current_phase == "encoder" and _encoder_sa2_enabled:
            use_sa2 = True
        elif _current_phase == "decoder" and _decoder_sa2_enabled:
            use_sa2 = True
        
        if not use_sa2 or not _check_sa2_available():
            # Use original forward (native attention)
            return original_forward(hidden_states, encoder_hidden_states, attention_mask, **kwargs)
        
        # SA2 Forward Path for diffusers Attention
        from sageattention import sageattn
        
        # diffusers Attention input: (B, C, H, W) for VAE or (B, seq_len, dim) for text
        # Video VAE uses 4D input: (B, C, H, W)
        if hidden_states.dim() == 4:
            batch_size, channels, height, width = hidden_states.shape
            residual = hidden_states
            
            # Apply group norm if available
            if hasattr(self, 'group_norm') and self.group_norm is not None:
                hidden_states = self.group_norm(hidden_states)
            elif hasattr(self, 'norm') and self.norm is not None:
                hidden_states = self.norm(hidden_states)
            
            # Reshape to (B, H*W, C) for attention
            hidden_states = hidden_states.view(batch_size, channels, height * width).transpose(1, 2)
            
            # Compute Q, K, V using diffusers Attention projections
            query = self.to_q(hidden_states)
            key = self.to_k(hidden_states if encoder_hidden_states is None else encoder_hidden_states)
            value = self.to_v(hidden_states if encoder_hidden_states is None else encoder_hidden_states)
            
            # Reshape for SA2: force 8 heads x 64-dim for Blackwell Tensor Cores
            inner_dim = query.size(-1)  # Should be 512 for VAE
            num_heads = 8
            head_dim = inner_dim // num_heads  # 512 / 8 = 64
            
            # (B, seq_len, inner_dim) -> (B, num_heads, seq_len, head_dim)
            query = query.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
            
            # Log block execution
            print(f"[VAE-SA2] Block {block_id:02d}: Phase={_current_phase}, Shape=[{batch_size}, {channels}, {height}, {width}], Heads={num_heads}, HeadDim={head_dim}")
            
            # SA2 Attention (sageattn expects (B, H, N, D))
            out = sageattn(query, key, value, tensor_layout="HND", is_causal=False)
            
            # Merge heads: (B, num_heads, seq_len, head_dim) -> (B, seq_len, inner_dim)
            out = out.transpose(1, 2).reshape(batch_size, height * width, inner_dim)
            
            # Output projection
            out = self.to_out[0](out)  # Linear
            if len(self.to_out) > 1:
                out = self.to_out[1](out)  # Dropout if present
            
            # Reshape back to (B, C, H, W)
            out = out.transpose(1, 2).view(batch_size, channels, height, width)
            
            # Residual connection if configured
            if hasattr(self, 'residual_connection') and self.residual_connection:
                out = out + residual
            
            return out
        else:
            # 3D input (B, seq_len, dim) - use original forward
            return original_forward(hidden_states, encoder_hidden_states, attention_mask, **kwargs)
    
    return sa2_forward


def inject_sparge_into_vae(vae, topk: Optional[float] = None, debug=None) -> int:
    """
    REAL SA2 injection into VAE diffusers Attention modules.
    Only patches when vae_encoder_sa2=True or vae_decoder_sa2=True.
    """
    global _original_forwards
    
    if not (_encoder_sa2_enabled or _decoder_sa2_enabled):
        # No patching needed - use native SDPA
        return 0
    
    if not _check_sa2_available():
        print("[VAE-SA2] WARNING: SageAttention 2 not available, using native SDPA")
        return 0
    
    patched_count = 0
    
    # Find all diffusers Attention modules in VAE
    try:
        from diffusers.models.attention_processor import Attention
    except ImportError:
        print("[VAE-SA2] WARNING: diffusers not available")
        return 0
    
    for name, module in vae.named_modules():
        if isinstance(module, Attention):
            # Cache original forward
            if id(module) not in _original_forwards:
                _original_forwards[id(module)] = module.forward
            
            # Create and bind SA2 forward
            sa2_forward = _create_sa2_forward_diffusers(_original_forwards[id(module)], patched_count + 1)
            import types
            module.forward = types.MethodType(sa2_forward, module)
            
            patched_count += 1
            print(f"[VAE-SA2] Patched Block {patched_count:02d}: {name}")
    
    if patched_count > 0:
        print(f"[VAE-SA2] Total: {patched_count} Attention blocks patched for SA2")
    
    return patched_count


def reset_vae_attention_logging():
    """Reset VAE attention logging state."""
    global _current_phase, _block_counter
    _current_phase = "unknown"
    _block_counter = 0


def restore_vae_original_forwards(vae):
    """Restore original forward methods to VAE diffusers Attention modules."""
    global _original_forwards
    
    try:
        from diffusers.models.attention_processor import Attention
    except ImportError:
        _original_forwards.clear()
        return 0
    
    restored_count = 0
    for name, module in vae.named_modules():
        if isinstance(module, Attention):
            if id(module) in _original_forwards:
                import types
                module.forward = types.MethodType(_original_forwards[id(module)], module)
                restored_count += 1
    
    _original_forwards.clear()
    return restored_count


# ============================================================================
# DiT BLOCK-LEVEL SPARSITY UPDATE
# ============================================================================

def update_dit_sparsity_blocks(runner, sparsity_threshold: float) -> int:
    """
    REAL block-level sparsity update for DiT model.
    Iterates through ALL FlashAttentionVarlen modules and sets sparsity_threshold.
    
    This MUST be called right before Phase 2 to ensure the value is applied.
    
    Args:
        runner: Runner with DiT model
        sparsity_threshold: Value from UI (0.3=Fast, 0.5=Balanced, 0.7=Quality)
        
    Returns:
        Number of blocks updated
    """
    if not hasattr(runner, 'dit') or runner.dit is None:
        print("[DiT-SPARSITY] WARNING: No DiT model found")
        return 0
    
    # Get the actual model (unwrap if needed)
    model = runner.dit
    if hasattr(model, 'dit_model'):
        model = model.dit_model
    
    updated_count = 0
    
    # Iterate through ALL modules and update FlashAttentionVarlen
    for name, module in model.named_modules():
        if type(module).__name__ == 'FlashAttentionVarlen':
            old_value = getattr(module, 'sparsity_threshold', None)
            module.sparsity_threshold = sparsity_threshold
            updated_count += 1
            print(f"[DiT-SPARSITY] Block {updated_count:02d}: sparsity_threshold = {sparsity_threshold} (was {old_value})")
    
    if updated_count > 0:
        mode_name = "Fast" if abs(sparsity_threshold - 0.3) < 0.01 else \
                    "Balanced" if abs(sparsity_threshold - 0.5) < 0.01 else \
                    "High Quality" if abs(sparsity_threshold - 0.7) < 0.01 else "Custom"
        print(f"[DiT-SPARSITY] Total: {updated_count} blocks updated to {sparsity_threshold} ({mode_name})")
    else:
        print("[DiT-SPARSITY] WARNING: No FlashAttentionVarlen blocks found!")
    
    return updated_count
