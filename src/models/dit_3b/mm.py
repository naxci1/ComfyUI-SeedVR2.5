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

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple
import torch
from torch import nn


# ==============================================================================
# BLACKWELL NVFP4 META TENSOR HANDLING
# ==============================================================================
# Helper functions to handle meta tensors in MMModule operations.
# ==============================================================================

def _has_meta_tensors_in_module(module: nn.Module) -> bool:
    """Check if a module contains any parameters/buffers on meta device."""
    if module is None:
        return False
    try:
        for param in module.parameters():
            if param.device.type == 'meta' or getattr(param, 'is_meta', False):
                return True
        for buffer in module.buffers():
            if buffer is not None:
                if buffer.device.type == 'meta' or getattr(buffer, 'is_meta', False):
                    return True
    except Exception:
        pass
    return False


def _ensure_module_on_device(module: nn.Module, target_device: torch.device, target_dtype: torch.dtype) -> None:
    """
    Ensure all parameters and buffers in a module are on the correct device.
    For Blackwell NVFP4 models, meta tensors need to be materialized to CUDA.
    """
    if module is None:
        return
    
    with torch.no_grad():
        for name, param in list(module.named_parameters()):
            if param is None:
                continue
            
            is_meta = param.device.type == 'meta' or getattr(param, 'is_meta', False)
            
            if is_meta:
                # Materialize meta tensor with random initialization
                new_param = torch.randn(param.shape, device=target_device, dtype=target_dtype) * 0.02
                try:
                    param.data = new_param
                except (RuntimeError, TypeError):
                    try:
                        param.set_(new_param)
                    except Exception:
                        pass
            elif param.device != target_device:
                # Move to correct device
                try:
                    param.data = param.data.to(target_device)
                except (RuntimeError, TypeError):
                    try:
                        param.set_(param.to(target_device))
                    except Exception:
                        pass
        
        for name, buffer in list(module.named_buffers()):
            if buffer is None:
                continue
            
            is_meta = buffer.device.type == 'meta' or getattr(buffer, 'is_meta', False)
            
            if is_meta:
                # Materialize meta buffer with zeros
                parts = name.split('.')
                if len(parts) > 1:
                    parent = module
                    for part in parts[:-1]:
                        parent = getattr(parent, part)
                    setattr(parent, parts[-1], torch.zeros_like(buffer, device=target_device, dtype=target_dtype))
                else:
                    setattr(module, name, torch.zeros_like(buffer, device=target_device, dtype=target_dtype))
            elif buffer.device != target_device:
                # Move to correct device
                parts = name.split('.')
                if len(parts) > 1:
                    parent = module
                    for part in parts[:-1]:
                        parent = getattr(parent, part)
                    setattr(parent, parts[-1], buffer.to(target_device))
                else:
                    setattr(module, name, buffer.to(target_device))


@dataclass
class MMArg:
    vid: Any
    txt: Any


def get_args(key: str, args: List[Any]) -> List[Any]:
    return [getattr(v, key) if isinstance(v, MMArg) else v for v in args]


def get_kwargs(key: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    return {k: getattr(v, key) if isinstance(v, MMArg) else v for k, v in kwargs.items()}


class MMModule(nn.Module):
    def __init__(
        self,
        module: Callable[..., nn.Module],
        *args,
        shared_weights: bool = False,
        vid_only: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.shared_weights = shared_weights
        self.vid_only = vid_only
        if self.shared_weights:
            assert get_args("vid", args) == get_args("txt", args)
            assert get_kwargs("vid", kwargs) == get_kwargs("txt", kwargs)
            self.all = module(*get_args("vid", args), **get_kwargs("vid", kwargs))
        else:
            self.vid = module(*get_args("vid", args), **get_kwargs("vid", kwargs))
            self.txt = (
                module(*get_args("txt", args), **get_kwargs("txt", kwargs))
                if not vid_only
                else None
            )

    def forward(
        self,
        vid: torch.FloatTensor,
        txt: torch.FloatTensor,
        *args,
        **kwargs,
    ) -> Tuple[
        torch.FloatTensor,
        torch.FloatTensor,
    ]:
        # ==================================================================
        # BLACKWELL NVFP4 META TENSOR HANDLING
        # ==================================================================
        # Ensure submodules are on correct device before forward pass.
        # This handles cases where NVFP4 loader skips some modules.
        # ==================================================================
        
        vid_module = self.vid if not self.shared_weights else self.all
        
        # Check if vid_module has meta tensors and materialize them
        if _has_meta_tensors_in_module(vid_module):
            _ensure_module_on_device(vid_module, vid.device, vid.dtype)
        
        vid = vid_module(vid, *get_args("vid", args), **get_kwargs("vid", kwargs))
        
        if not self.vid_only:
            txt_module = self.txt if not self.shared_weights else self.all
            
            # Check if txt_module has meta tensors and materialize them
            if txt_module is not None and _has_meta_tensors_in_module(txt_module):
                _ensure_module_on_device(txt_module, txt.device, txt.dtype)
            
            txt = txt_module(txt, *get_args("txt", args), **get_kwargs("txt", kwargs))
        
        return vid, txt
