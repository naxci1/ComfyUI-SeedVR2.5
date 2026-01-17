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

from typing import Callable, Optional
from diffusers.models.normalization import RMSNorm
from torch import nn
import torch
import torch.nn.functional as F
import numbers
from torch.nn.parameter import Parameter
from torch.nn import init

# (dim: int, eps: float, elementwise_affine: bool)
norm_layer_type = Callable[[int, float, bool], nn.Module]


# ==============================================================================
# BLACKWELL NVFP4 META TENSOR HANDLING
# ==============================================================================
# These helper functions ensure normalization layers work correctly when model
# weights remain on the 'meta' device after NVFP4 quantization on Blackwell GPUs.
# ==============================================================================

def _is_meta_tensor(t: torch.Tensor) -> bool:
    """Check if tensor is on the meta device (no actual VRAM allocated)."""
    if t is None:
        return False
    try:
        return t.device.type == 'meta' or getattr(t, 'is_meta', False)
    except Exception:
        return False


def _ensure_norm_tensor_ready(t: torch.Tensor, reference: torch.Tensor, is_weight: bool = True) -> torch.Tensor:
    """
    Ensure a normalization tensor (weight or bias) is on the correct device and dtype.
    
    For Blackwell NVFP4 models, tensors may still be on 'meta' device after loading.
    This function materializes them to actual VRAM on the reference tensor's device.
    
    Args:
        t: The tensor to check/materialize (weight or bias)
        reference: Reference tensor (normalized input) to get device/dtype from
        is_weight: If True, initializes to 1.0 (weight), else 0.0 (bias)
    
    Returns:
        Tensor ready for arithmetic on reference's device
    """
    if t is None:
        return None
    
    target_device = reference.device
    target_dtype = reference.dtype
    
    # Check if tensor is on meta device (common with NVFP4 Blackwell initialization)
    if _is_meta_tensor(t):
        # Materialize meta tensor with appropriate initialization
        if is_weight:
            # Weights should be initialized to 1.0 for normalization
            return torch.ones(t.shape, device=target_device, dtype=target_dtype)
        else:
            # Biases should be initialized to 0.0
            return torch.zeros(t.shape, device=target_device, dtype=target_dtype)
    
    # Check if tensor is on wrong device
    if t.device != target_device:
        return t.to(device=target_device, dtype=target_dtype)
    
    # Check if dtype is FP8 (needs conversion for arithmetic)
    if hasattr(torch, 'float8_e4m3fn'):
        fp8_types = (torch.float8_e4m3fn, torch.float8_e5m2)
        if t.dtype in fp8_types:
            return t.to(target_dtype)
    
    # Convert dtype if mismatched
    if t.dtype != target_dtype:
        return t.to(target_dtype)
    
    return t


class CustomLayerNorm(nn.Module):
    """
    Custom LayerNorm implementation to replace Apex FusedLayerNorm
    """
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(CustomLayerNorm, self).__init__()
        
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if self.elementwise_affine:
            self.weight = Parameter(torch.Tensor(*normalized_shape))
            self.bias = Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input):
        # ==================================================================
        # BLACKWELL NVFP4 META TENSOR HANDLING
        # ==================================================================
        # On Blackwell GPUs with NVFP4 quantization, weight/bias may still
        # be on 'meta' device. Materialize them to input's device before use.
        # ==================================================================
        
        weight = self.weight
        bias = self.bias
        
        if self.elementwise_affine and weight is not None:
            # Check and materialize meta tensors
            if _is_meta_tensor(weight):
                weight = _ensure_norm_tensor_ready(weight, input, is_weight=True)
                # Update stored weight to avoid repeated materialization
                try:
                    with torch.no_grad():
                        self.weight.data = weight
                except (RuntimeError, TypeError):
                    # Parameter is frozen or dtype mismatch - use local copy
                    pass
            elif weight.device != input.device or weight.dtype != input.dtype:
                weight = weight.to(device=input.device, dtype=input.dtype)
                
            if bias is not None:
                if _is_meta_tensor(bias):
                    bias = _ensure_norm_tensor_ready(bias, input, is_weight=False)
                    try:
                        with torch.no_grad():
                            self.bias.data = bias
                    except (RuntimeError, TypeError):
                        pass
                elif bias.device != input.device or bias.dtype != input.dtype:
                    bias = bias.to(device=input.device, dtype=input.dtype)
        
        return F.layer_norm(
            input, self.normalized_shape, weight, bias, self.eps)


class CustomRMSNorm(nn.Module):
    """
    Custom RMSNorm implementation to replace Apex FusedRMSNorm
    """
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(CustomRMSNorm, self).__init__()
        
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if self.elementwise_affine:
            self.weight = Parameter(torch.ones(*normalized_shape))
        else:
            self.register_parameter('weight', None)

    def forward(self, input):
        # ==================================================================
        # BLACKWELL NVFP4 META TENSOR HANDLING
        # ==================================================================
        # On Blackwell GPUs with NVFP4 quantization, weight may still be on
        # 'meta' device. Materialize it to input's device before arithmetic.
        # ==================================================================
        
        # RMS normalization: x / sqrt(mean(x^2) + eps) * weight
        dims = tuple(range(-len(self.normalized_shape), 0))
        
        # Calculate RMS: sqrt(mean(x^2))
        variance = input.pow(2).mean(dim=dims, keepdim=True)
        rms = torch.sqrt(variance + self.eps)
        
        # Normalize
        normalized = input / rms
        
        if self.elementwise_affine:
            weight = self.weight
            
            # Check for meta tensor (common with NVFP4 Blackwell initialization)
            if _is_meta_tensor(weight):
                weight = _ensure_norm_tensor_ready(weight, normalized, is_weight=True)
                # Update stored weight to avoid repeated materialization
                try:
                    with torch.no_grad():
                        self.weight.data = weight
                except (RuntimeError, TypeError):
                    # Parameter is frozen or dtype mismatch - use local copy
                    pass
            elif weight.device != normalized.device:
                # Device mismatch - move to correct device
                weight = weight.to(device=normalized.device, dtype=normalized.dtype)
                try:
                    with torch.no_grad():
                        self.weight.data = weight
                except (RuntimeError, TypeError):
                    pass
            else:
                # Check for FP8 dtype (needs conversion for arithmetic)
                if hasattr(torch, 'float8_e4m3fn'):
                    fp8_types = (torch.float8_e4m3fn, torch.float8_e5m2)
                    if weight.dtype in fp8_types:
                        weight = weight.to(normalized.dtype)
                elif weight.dtype != normalized.dtype:
                    weight = weight.to(normalized.dtype)
            
            return normalized * weight
        
        return normalized


def get_norm_layer(norm_type: Optional[str]) -> norm_layer_type:

    def _norm_layer(dim: int, eps: float, elementwise_affine: bool):
        if norm_type is None:
            return nn.Identity()

        if norm_type == "layer":
            return nn.LayerNorm(
                normalized_shape=dim,
                eps=eps,
                elementwise_affine=elementwise_affine,
            )

        if norm_type == "rms":
            return RMSNorm(
                dim=dim,
                eps=eps,
                elementwise_affine=elementwise_affine,
            )

        if norm_type == "fusedln":
            # Use custom LayerNorm instead of Apex FusedLayerNorm
            return CustomLayerNorm(
                normalized_shape=dim,
                elementwise_affine=elementwise_affine,
                eps=eps,
            )

        if norm_type == "fusedrms":
            # Use custom RMSNorm instead of Apex FusedRMSNorm
            return CustomRMSNorm(
                normalized_shape=dim,
                elementwise_affine=elementwise_affine,
                eps=eps,
            )

        raise NotImplementedError(f"{norm_type} is not supported")

    return _norm_layer