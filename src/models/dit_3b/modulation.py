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

from typing import Callable, List, Optional
import torch
from einops import rearrange
from torch import nn

from ...common.cache import Cache
from ...common.distributed.ops import slice_inputs

# (dim: int, emb_dim: int)
ada_layer_type = Callable[[int, int], nn.Module]


def _materialize_meta_tensor(t: torch.Tensor, target_device: torch.device, target_dtype: torch.dtype) -> torch.Tensor:
    """
    Materialize a meta tensor to actual VRAM on the target device.
    
    For Blackwell NVFP4 models, meta tensors are placeholders without actual data.
    This function allocates real memory on the GPU and returns a usable tensor.
    
    Args:
        t: The tensor to materialize (may be on 'meta' device)
        target_device: The device to allocate on (e.g., cuda:0)
        target_dtype: The dtype to use for the allocated tensor
        
    Returns:
        A tensor on the target device with actual memory allocated
    """
    if t is None:
        return None
    
    # Check if tensor is on meta device
    if t.device.type == 'meta' or (hasattr(t, 'is_meta') and t.is_meta):
        # Allocate actual memory on target device using zeros for stability
        return torch.zeros(t.shape, device=target_device, dtype=target_dtype)
    
    # Check if tensor is on wrong device
    if t.device != target_device:
        return t.to(device=target_device, dtype=target_dtype)
    
    # Check if tensor needs dtype conversion
    if t.dtype != target_dtype:
        return t.to(dtype=target_dtype)
    
    return t


def _ensure_tensor_ready(t: torch.Tensor, hid: torch.Tensor) -> torch.Tensor:
    """
    Ensure a tensor is ready for arithmetic operations with hid tensor.
    
    Handles:
    1. Meta tensors (materialized to hid's device)
    2. Wrong device tensors (moved to hid's device)
    3. FP8 tensors (converted to hid's dtype for arithmetic)
    
    Args:
        t: The tensor to check/prepare
        hid: The reference tensor for device/dtype
        
    Returns:
        A tensor ready for arithmetic with hid
    """
    if t is None:
        return None
    
    target_device = hid.device
    target_dtype = hid.dtype
    
    # Handle meta tensors - must materialize first
    if t.device.type == 'meta' or (hasattr(t, 'is_meta') and t.is_meta):
        return torch.zeros(t.shape, device=target_device, dtype=target_dtype)
    
    # Handle device mismatch
    if t.device != target_device:
        t = t.to(device=target_device)
    
    # Handle FP8 dtype incompatibility for arithmetic
    if hasattr(torch, 'float8_e4m3fn'):
        fp8_types = (torch.float8_e4m3fn, torch.float8_e5m2)
        if t.dtype in fp8_types:
            t = t.to(dtype=target_dtype)
    
    # Handle general dtype mismatch
    if t.dtype != target_dtype:
        t = t.to(dtype=target_dtype)
    
    return t


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
        shiftB, scaleB, gateB = (
            getattr(self, f"{layer}_shift", None),
            getattr(self, f"{layer}_scale", None),
            getattr(self, f"{layer}_gate", None),
        )

        # ============================================================
        # BLACKWELL NVFP4 META TENSOR MATERIALIZATION
        # ============================================================
        # For Blackwell GPUs with NVFP4 quantization, modulation parameters
        # may remain on 'meta' device after model loading. We MUST materialize
        # them to hid's device before any arithmetic operations.
        
        # Ensure all A tensors (from emb) are on correct device/dtype
        shiftA = _ensure_tensor_ready(shiftA, hid)
        scaleA = _ensure_tensor_ready(scaleA, hid)
        gateA = _ensure_tensor_ready(gateA, hid)
        
        # Ensure all B tensors (learned parameters) are on correct device/dtype
        # This handles: meta tensors, wrong device, FP8 conversion
        shiftB = _ensure_tensor_ready(shiftB, hid)
        scaleB = _ensure_tensor_ready(scaleB, hid)
        gateB = _ensure_tensor_ready(gateB, hid)
        
        # Update stored parameters if they were on meta/wrong device
        # This avoids repeated materialization on subsequent forward passes
        if shiftB is not None:
            orig_shiftB = getattr(self, f"{layer}_shift", None)
            if orig_shiftB is not None and (orig_shiftB.device.type == 'meta' or orig_shiftB.device != hid.device):
                with torch.no_grad():
                    try:
                        orig_shiftB.data = shiftB
                    except (RuntimeError, TypeError):
                        pass  # Parameter update failed, will materialize again next time
                        
        if scaleB is not None:
            orig_scaleB = getattr(self, f"{layer}_scale", None)
            if orig_scaleB is not None and (orig_scaleB.device.type == 'meta' or orig_scaleB.device != hid.device):
                with torch.no_grad():
                    try:
                        orig_scaleB.data = scaleB
                    except (RuntimeError, TypeError):
                        pass
                        
        if gateB is not None:
            orig_gateB = getattr(self, f"{layer}_gate", None)
            if orig_gateB is not None and (orig_gateB.device.type == 'meta' or orig_gateB.device != hid.device):
                with torch.no_grad():
                    try:
                        orig_gateB.data = gateB
                    except (RuntimeError, TypeError):
                        pass

        # ============================================================
        # IN-PLACE MODULATION OPERATIONS
        # ============================================================
        # All tensors are now guaranteed to be on hid.device with compatible dtype
        
        if mode == "in":
            # Compute scale and shift additions safely
            if scaleB is not None:
                scale_sum = scaleA + scaleB
            else:
                scale_sum = scaleA
                
            if shiftB is not None:
                shift_sum = shiftA + shiftB
            else:
                shift_sum = shiftA
                
            return hid.mul_(scale_sum).add_(shift_sum)
            
        if mode == "out":
            if gateB is not None:
                gate_sum = gateA + gateB
            else:
                gate_sum = gateA
            return hid.mul_(gate_sum)
            
        raise NotImplementedError

    def extra_repr(self) -> str:
        return f"dim={self.dim}, emb_dim={self.emb_dim}, layers={self.layers}"