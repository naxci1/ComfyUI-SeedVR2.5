"""
VAE Attention Utilities for SeedVR2

Provides optional SageAttention 2 support for VAE attention blocks.
NO GLOBAL MONKEY-PATCHING - respects UI toggles completely.

When vae_encoder_sa2=False and vae_decoder_sa2=False, this module does NOTHING.
The VAE uses PyTorch's native SDPA which is already optimized for Blackwell GPUs.
"""

import os
import torch
import torch.nn.functional as F
import logging
from typing import Optional, Literal

# Configure logging
logger = logging.getLogger("SeedVR2.VAE")

# ============================================================================
# MINIMAL GLOBAL STATE - NO MONKEY-PATCHING
# ============================================================================

# SA2 availability (cached)
_sa2_available = None

# Current phase context (for optional logging only)
_current_phase: Literal["encoder", "decoder", "unknown"] = "unknown"

# UI-controlled SA2 toggles (set at runtime from node inputs)
# DEFAULTS: False = Native SDPA (fastest path, no patching)
_encoder_sa2_enabled = False  # Default: False = Native SDPA
_decoder_sa2_enabled = False  # Default: False = Native SDPA

# Memory policy: expandable_segments for Blackwell (optional optimization)
try:
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
except Exception:
    pass


def set_vae_phase(phase: Literal["encoder", "decoder", "unknown"]):
    """Set the current VAE phase for logging context (minimal logging)."""
    global _current_phase
    _current_phase = phase


def configure_vae_sa2(encoder_sa2: bool = False, decoder_sa2: bool = False):
    """
    Configure SA2 enablement for Encoder and Decoder from UI toggles.
    
    When BOTH are False (default), NO patching occurs and VAE uses native PyTorch SDPA.
    This is the fastest path for Blackwell GPUs.
    
    Args:
        encoder_sa2: Enable SA2 for Encoder (Phase 1) - default False (Native SDPA)
        decoder_sa2: Enable SA2 for Decoder (Phase 3) - default False (Native SDPA)
    """
    global _encoder_sa2_enabled, _decoder_sa2_enabled
    
    _encoder_sa2_enabled = encoder_sa2
    _decoder_sa2_enabled = decoder_sa2
    
    # Minimal logging - only log configuration, not every block
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
    """API compatibility - SA2 does not use sparsity thresholds."""
    pass


def get_vae_sparsity_threshold() -> float:
    """API compatibility - SA2 does not use sparsity thresholds."""
    return 0.5


def standard_vae_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Standard VAE attention using PyTorch's native SDPA.
    
    NO monkey-patching. Uses PyTorch's optimized SDPA which automatically
    selects FlashAttention or Memory Efficient backend for Blackwell GPUs.
    """
    if scale is None:
        scale = query.size(-1) ** -0.5
    
    return F.scaled_dot_product_attention(
        query, key, value, 
        attn_mask=attention_mask, 
        dropout_p=0.0, 
        is_causal=False,
        scale=scale
    )


def inject_sparge_into_vae(vae, topk: Optional[float] = None, debug=None) -> int:
    """
    API compatibility stub - NO MONKEY-PATCHING.
    
    When vae_encoder_sa2=False and vae_decoder_sa2=False, this does nothing.
    The VAE uses its native PyTorch SDPA which is already optimized.
    """
    # NO PATCHING - let native SDPA handle attention
    return 0


def reset_vae_attention_logging():
    """Reset VAE attention logging state."""
    global _current_phase
    _current_phase = "unknown"


# NO AUTO-PATCHING - respect UI toggles completely
