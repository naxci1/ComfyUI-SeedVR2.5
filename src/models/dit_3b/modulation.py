# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.

"""
Modulation layers for DiT (Diffusion Transformer).

BLACKWELL-OPTIMIZED: Handles meta tensor materialization for NVFP4 loading on RTX 50-series.
Uses aggressive device synchronization to ensure all parameters are on the correct device
before any arithmetic operations.
"""

from typing import Callable, List, Optional
import torch
from einops import rearrange
from torch import nn

from ...common.cache import Cache
from ...common.distributed.ops import slice_inputs

# (dim: int, emb_dim: int)
ada_layer_type = Callable[[int, int], nn.Module]


def get_ada_layer(ada_layer: str) -> ada_layer_type:
    if ada_layer == "single":
        return AdaSingle
    raise NotImplementedError(f"{ada_layer} is not supported")


def expand_dims(x: torch.Tensor, dim: int, ndim: int):
    """
    Expand tensor "x" to "ndim" by adding empty dims at "dim".
    Example: x is (b d), target ndim is 5, add dim at 1, return (b 1 1 1 d).
    """
    shape = x.shape
    shape = shape[:dim] + (1,) * (ndim - len(shape)) + shape[dim:]
    return x.reshape(shape)


def _ensure_tensor_on_device(tensor: Optional[torch.Tensor], target_device: torch.device, 
                              target_dtype: torch.dtype) -> Optional[torch.Tensor]:
    """
    BULLDOZER APPROACH: Force tensor to target device, handling meta tensors.
    
    For Blackwell RTX 50 GPUs with NVFP4 meta-tensor loading, we must:
    1. Check if tensor is None (using 'is not None', never 'if tensor')
    2. Check if tensor is on meta device (using tensor.is_meta)
    3. Materialize meta tensors with torch.empty_like to target device
    4. Move any non-meta tensors to target device
    
    Args:
        tensor: Tensor to move (may be None or meta)
        target_device: Target device (e.g., cuda:0)
        target_dtype: Target dtype for computation
        
    Returns:
        Tensor on target device, or None if input was None
    """
    if tensor is None:
        return None
    
    # Handle meta tensors - must use to_empty() pattern
    if tensor.is_meta:
        # Create empty tensor on target device with same shape
        materialized = torch.empty(
            tensor.shape, 
            dtype=target_dtype, 
            device=target_device
        )
        return materialized
    
    # Handle FP8 types for Blackwell NVFP4
    if hasattr(torch, 'float8_e4m3fn'):
        fp8_types = (torch.float8_e4m3fn, torch.float8_e5m2)
        if tensor.dtype in fp8_types:
            tensor = tensor.to(target_dtype)
    
    # Move to target device if not already there
    if tensor.device != target_device:
        tensor = tensor.to(device=target_device, dtype=target_dtype)
    elif tensor.dtype != target_dtype:
        tensor = tensor.to(dtype=target_dtype)
    
    return tensor


class AdaSingle(nn.Module):
    def __init__(
        self,
        dim: int,
        emb_dim: int,
        layers: List[str],
        modes: List[str] = ["in", "out"],
    ):
        assert emb_dim == 6 * dim, "AdaSingle requires emb_dim == 6 * dim"
        super().__init__()
        self.dim = dim
        self.emb_dim = emb_dim
        self.layers = layers
        for l in layers:
            if "in" in modes:
                self.register_parameter(f"{l}_shift", nn.Parameter(torch.randn(dim) / dim**0.5))
                self.register_parameter(
                    f"{l}_scale", nn.Parameter(torch.randn(dim) / dim**0.5 + 1)
                )
            if "out" in modes:
                self.register_parameter(f"{l}_gate", nn.Parameter(torch.randn(dim) / dim**0.5))

    def forward(
        self,
        hid: torch.FloatTensor,  # b ... c
        emb: torch.FloatTensor,  # b d
        layer: str,
        mode: str,
        cache: Cache = Cache(disable=True),
        branch_tag: str = "",
        hid_len: Optional[torch.LongTensor] = None,  # b
    ) -> torch.FloatTensor:
        """
        Forward pass with BULLDOZER meta tensor handling.
        
        BLACKWELL-OPTIMIZED: Forces all modulation parameters to the device of the
        input 'hid' tensor BEFORE any arithmetic operations at line 110+.
        """
        # Get target device and dtype from input tensor
        target_device = hid.device
        target_dtype = hid.dtype
        
        idx = self.layers.index(layer)
        emb = rearrange(emb, "b (d l g) -> b d l g", l=len(self.layers), g=3)[..., idx, :]
        emb = expand_dims(emb, 1, hid.ndim + 1)

        if hid_len is not None:
            emb = cache(
                f"emb_repeat_{idx}_{branch_tag}",
                lambda: slice_inputs(
                    torch.repeat_interleave(emb, hid_len, dim=0),
                    dim=0,
                ),
            )

        shiftA, scaleA, gateA = emb.unbind(-1)
        
        # Get raw parameters (may be None or on meta device)
        shiftB_raw = getattr(self, f"{layer}_shift", None)
        scaleB_raw = getattr(self, f"{layer}_scale", None)
        gateB_raw = getattr(self, f"{layer}_gate", None)
        
        # BULLDOZER APPROACH: Force all parameters to target device
        # This handles meta tensors from Blackwell NVFP4 loading
        shiftB = _ensure_tensor_on_device(shiftB_raw, target_device, target_dtype)
        scaleB = _ensure_tensor_on_device(scaleB_raw, target_device, target_dtype)
        gateB = _ensure_tensor_on_device(gateB_raw, target_device, target_dtype)

        # Now all tensors are guaranteed to be on the same device
        if mode == "in":
            # scaleB and shiftB are now guaranteed to be on target_device
            if scaleB is not None and shiftB is not None:
                return hid.mul_(scaleA + scaleB).add_(shiftA + shiftB)
            elif scaleB is not None:
                return hid.mul_(scaleA + scaleB).add_(shiftA)
            elif shiftB is not None:
                return hid.mul_(scaleA).add_(shiftA + shiftB)
            else:
                return hid.mul_(scaleA).add_(shiftA)
                
        if mode == "out":
            if gateB is not None:
                return hid.mul_(gateA + gateB)
            else:
                # If no gate parameter, just use the embedding gate
                return hid.mul_(gateA)
            
        raise NotImplementedError

    def extra_repr(self) -> str:
        return f"dim={self.dim}, emb_dim={self.emb_dim}, layers={self.layers}"