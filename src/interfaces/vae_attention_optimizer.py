"""
SeedVR2 VAE Attention Optimizer Node

Patches VAE attention blocks to use SageAttention v2 (when available) or
optimized SDPA for maximum inference speed on Windows systems.

This node is designed for users who cannot use torch.compile (Windows) and
want to maximize VAE performance through attention-level optimizations.
"""

import types
import torch
import torch.nn.functional as F
from comfy_api.latest import io
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Check for SageAttention availability
SAGE_ATTN_AVAILABLE = False
sageattn_func = None

try:
    from sageattention import sageattn
    sageattn_func = sageattn
    SAGE_ATTN_AVAILABLE = True
except (ImportError, AttributeError, OSError):
    pass


def _optimized_attention_forward(self, hidden_states, encoder_hidden_states=None, 
                                  attention_mask=None, **cross_attention_kwargs):
    """
    Optimized attention forward pass using SageAttention or SDPA.
    
    This replaces the default diffusers Attention forward with a faster
    implementation optimized for NVIDIA GPUs.
    """
    residual = hidden_states
    
    input_ndim = hidden_states.ndim
    
    if input_ndim == 4:
        batch_size, channel, height, width = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
    
    batch_size, sequence_length, _ = (
        hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
    )
    
    if self.group_norm is not None:
        hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
    
    query = self.to_q(hidden_states)
    
    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states
    elif self.norm_cross:
        encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)
    
    key = self.to_k(encoder_hidden_states)
    value = self.to_v(encoder_hidden_states)
    
    inner_dim = key.shape[-1]
    head_dim = inner_dim // self.heads
    
    query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
    key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
    
    # Use SageAttention if available, otherwise use optimized SDPA
    use_sage = getattr(self, '_use_sage_attention', False) and SAGE_ATTN_AVAILABLE
    
    if use_sage:
        # SageAttention expects (B, H, N, D) format
        # Convert to half precision for SageAttention
        original_dtype = query.dtype
        if query.dtype == torch.float32:
            query = query.to(torch.float16)
            key = key.to(torch.float16)
            value = value.to(torch.float16)
        
        try:
            hidden_states = sageattn_func(query, key, value, is_causal=False)
            if hidden_states.dtype != original_dtype:
                hidden_states = hidden_states.to(original_dtype)
        except (RuntimeError, ValueError, TypeError) as e:
            # Fallback to SDPA if SageAttention fails (e.g., unsupported tensor shapes)
            if query.dtype != original_dtype:
                query = query.to(original_dtype)
                key = key.to(original_dtype)
                value = value.to(original_dtype)
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, 
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False
            )
    else:
        # Optimized SDPA with memory efficient settings
        hidden_states = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False
        )
    
    hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
    hidden_states = hidden_states.to(query.dtype)
    
    # Linear projection
    hidden_states = self.to_out[0](hidden_states)
    # Dropout
    hidden_states = self.to_out[1](hidden_states)
    
    if input_ndim == 4:
        hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
    
    if self.residual_connection:
        hidden_states = hidden_states + residual
    
    hidden_states = hidden_states / self.rescale_output_factor
    
    return hidden_states


def _convert_mid_block_to_fp16(model_part, part_name: str):
    """Convert mid_block attentions to FP16 if available."""
    if hasattr(model_part, 'mid_block') and hasattr(model_part.mid_block, 'attentions'):
        for attn in model_part.mid_block.attentions:
            if attn is not None:
                attn.to(torch.float16)


def patch_vae_attention(vae_model, use_sage: bool = True, use_fp16: bool = True) -> int:
    """
    Patch VAE attention blocks to use optimized attention.
    
    Args:
        vae_model: The VAE model to patch
        use_sage: Whether to use SageAttention (if available)
        use_fp16: Whether to cast to FP16 for Tensor Core usage
        
    Returns:
        Number of attention blocks patched
    """
    from diffusers.models.attention_processor import Attention
    
    patched_count = 0
    
    # Enable cuDNN benchmark for better performance
    torch.backends.cudnn.benchmark = True
    
    # Iterate through all modules
    for name, module in vae_model.named_modules():
        if isinstance(module, Attention):
            # Store original forward and mark for SageAttention
            if not hasattr(module, '_original_forward'):
                module._original_forward = module.forward
                module._use_sage_attention = use_sage and SAGE_ATTN_AVAILABLE
                
                # Bind the optimized forward
                module.forward = types.MethodType(_optimized_attention_forward, module)
                patched_count += 1
                
                logger.debug(f"Patched attention: {name}")
    
    # Convert to FP16 if requested (for Tensor Core usage)
    if use_fp16:
        if hasattr(vae_model, 'encoder'):
            _convert_mid_block_to_fp16(vae_model.encoder, 'encoder')
        if hasattr(vae_model, 'decoder'):
            _convert_mid_block_to_fp16(vae_model.decoder, 'decoder')
    
    return patched_count


def unpatch_vae_attention(vae_model) -> int:
    """
    Restore original VAE attention forward methods.
    
    Args:
        vae_model: The VAE model to restore
        
    Returns:
        Number of attention blocks restored
    """
    from diffusers.models.attention_processor import Attention
    
    restored_count = 0
    
    for name, module in vae_model.named_modules():
        if isinstance(module, Attention):
            if hasattr(module, '_original_forward'):
                module.forward = module._original_forward
                delattr(module, '_original_forward')
                if hasattr(module, '_use_sage_attention'):
                    delattr(module, '_use_sage_attention')
                restored_count += 1
                logger.debug(f"Restored attention: {name}")
    
    return restored_count


class SeedVR2VAEAttentionOptimizer(io.ComfyNode):
    """
    Optimize VAE attention blocks for maximum inference speed.
    
    This node patches the VAE's attention mechanisms to use:
    - SageAttention v2 (if available and enabled)
    - Optimized PyTorch SDPA with memory-efficient settings
    
    Designed for Windows users who cannot use torch.compile.
    Works best with NVIDIA RTX GPUs (30xx, 40xx, 50xx series).
    """
    
    @classmethod
    def define_schema(cls) -> io.Schema:
        sage_status = "✓ Available" if SAGE_ATTN_AVAILABLE else "✗ Not installed"
        
        return io.Schema(
            node_id="SeedVR2VAEAttentionOptimizer",
            display_name="SeedVR2 VAE Attention Optimizer",
            category="SEEDVR2",
            description=(
                f"Optimize VAE attention for maximum speed on NVIDIA GPUs.\n\n"
                f"SageAttention Status: {sage_status}\n\n"
                "This node patches VAE attention blocks to use:\n"
                "• SageAttention v2 (if available) - fastest option\n"
                "• Optimized SDPA - fallback with memory-efficient settings\n\n"
                "Best for Windows users who cannot use torch.compile.\n"
                "Works with RTX 30xx, 40xx, and 50xx series GPUs."
            ),
            inputs=[
                io.Custom("SEEDVR2_VAE").Input("vae",
                    tooltip="VAE configuration from SeedVR2 Load VAE Model node"
                ),
                io.Boolean.Input("use_sage_attention",
                    default=True,
                    tooltip=(
                        f"Use SageAttention v2 for maximum speed (Status: {sage_status}).\n"
                        "Automatically falls back to optimized SDPA if unavailable.\n\n"
                        "To install SageAttention: pip install sageattention"
                    )
                ),
                io.Boolean.Input("use_fp16",
                    default=True,
                    tooltip=(
                        "Cast attention blocks to FP16 for Tensor Core acceleration.\n"
                        "Provides significant speedup on RTX GPUs.\n"
                        "May slightly affect output quality (usually imperceptible)."
                    )
                ),
                io.Boolean.Input("enable_cudnn_benchmark",
                    default=True,
                    tooltip=(
                        "Enable cuDNN benchmark mode for faster convolutions.\n"
                        "Slightly increases memory usage but improves speed.\n"
                        "Recommended for consistent input sizes."
                    )
                ),
            ],
            outputs=[
                io.Custom("SEEDVR2_VAE").Output(
                    tooltip="Optimized VAE configuration with patched attention blocks"
                )
            ]
        )
    
    @classmethod
    def execute(cls, vae: Dict[str, Any], use_sage_attention: bool = True,
                use_fp16: bool = True, enable_cudnn_benchmark: bool = True) -> io.NodeOutput:
        """
        Apply attention optimizations to VAE configuration.
        
        The actual patching happens when the VAE model is loaded by the upscaler node.
        This node adds optimization flags to the configuration.
        """
        # Enable cuDNN benchmark if requested
        if enable_cudnn_benchmark:
            torch.backends.cudnn.benchmark = True
        
        # Create optimized config by copying and adding optimization flags
        optimized_config = dict(vae)
        optimized_config["attention_optimization"] = {
            "enabled": True,
            "use_sage_attention": use_sage_attention and SAGE_ATTN_AVAILABLE,
            "use_fp16": use_fp16,
            "sage_available": SAGE_ATTN_AVAILABLE,
            "patch_function": patch_vae_attention,
            "unpatch_function": unpatch_vae_attention,
        }
        
        # Log optimization status
        if use_sage_attention and SAGE_ATTN_AVAILABLE:
            logger.info("VAE Attention Optimizer: SageAttention v2 enabled")
        elif use_sage_attention:
            logger.info("VAE Attention Optimizer: SageAttention not available, using optimized SDPA")
        else:
            logger.info("VAE Attention Optimizer: Using optimized SDPA")
        
        return io.NodeOutput(optimized_config)


# Export for node registration
NODES = [SeedVR2VAEAttentionOptimizer]
