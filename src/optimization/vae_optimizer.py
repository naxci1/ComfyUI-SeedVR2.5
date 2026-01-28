# Copyright (c) 2025 ComfyUI-SeedVR2.5 Contributors
# SPDX-License-Identifier: Apache-2.0
#
# ULTRA-MINIMAL Blackwell sm_120 Native FP8 Optimizer
# NO DOUBLE CONVERSION | NO TILE OVERRIDE | PURE PERFORMANCE

"""
Ultra-minimal VAE optimizer for Blackwell sm_120 with native FP8.

Key principles:
- Direct FP8 conversion (no bf16 intermediate)
- NO tile size overrides (use model defaults)
- Only essential performance flags
- Maximum speed, minimal code
"""

import torch
import torch.nn as nn
from typing import Optional


def optimize_3d_vae_for_blackwell(
    model: nn.Module,
    device: str = "cuda",
    **kwargs
) -> nn.Module:
    """
    Ultra-lean Blackwell sm_120 FP8 optimizer.
    NO OVERHEAD - Direct FP8 conversion with cuDNN benchmark disabled.
    """
    torch.cuda.empty_cache()
    
    # Direct-to-FP8 with no intermediate buffers
    model.to(device=device, dtype=torch.float8_e4m3fn)
    
    # Use TF32 for speed but save VRAM
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    
    print("[BLACKWELL] ZERO-OVERHEAD FP8 ENABLED")
    return model


# Backward compatibility - keep the same function signature
def optimize_for_windows_blackwell(model: nn.Module, **kwargs) -> nn.Module:
    """Alias for optimize_3d_vae_for_blackwell for backward compatibility."""
    return optimize_3d_vae_for_blackwell(model, **kwargs)


# Export public API
__all__ = [
    'optimize_3d_vae_for_blackwell',
    'optimize_for_windows_blackwell',
]
