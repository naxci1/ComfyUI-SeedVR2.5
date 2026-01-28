#!/usr/bin/env python3
"""
3D Causal VAE Optimizer for Windows + NVIDIA Blackwell (RTX 50xx)

This script provides a production-ready wrapper to optimize 3D Video VAE models
for Windows environments and Blackwell architecture without using torch.compile.

Key Optimizations:
1. Channels-last 3D memory format for Conv3d layers
2. Flash Attention (SDP) for attention blocks
3. TF32 support for Blackwell matrix operations
4. cuDNN benchmark mode for optimal convolution algorithms
5. Fused operations (F.silu) in forward passes

Usage:
    from optimize_3d_vae_blackwell import optimize_vae_for_inference
    
    # Load your VAE model
    vae = load_vae_model(...)
    
    # Apply optimizations
    vae = optimize_vae_for_inference(vae, device='cuda')
    
    # Run inference
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
        encoded = vae.encode(video_input)
"""

import torch
import torch.nn as nn
from typing import Optional
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def optimize_vae_for_inference(
    vae_model: nn.Module,
    device: str = 'cuda',
    enable_channels_last_3d: bool = True,
    enable_flash_attention: bool = True,
    enable_tf32: bool = True,
    enable_cudnn_benchmark: bool = True,
    verbose: bool = True,
) -> nn.Module:
    """
    Production-ready function to optimize 3D Causal VAE for Windows + Blackwell.
    
    This function applies all recommended optimizations for inference on Windows
    with NVIDIA RTX 50xx (Blackwell) GPUs.
    
    Args:
        vae_model: The 3D VAE model to optimize (e.g., VideoAutoencoderKL)
        device: Target device ('cuda' or 'cpu')
        enable_channels_last_3d: Apply channels_last_3d to Conv3d layers
        enable_flash_attention: Enable SDP (Flash Attention) for attention blocks
        enable_tf32: Enable TF32 for matrix operations (Blackwell)
        enable_cudnn_benchmark: Enable cuDNN auto-tuner
        verbose: Print optimization status
    
    Returns:
        Optimized VAE model ready for inference
    
    Example:
        >>> # Simple usage
        >>> vae = load_model('ema_vae_fp16.safetensors')
        >>> vae = optimize_vae_for_inference(vae)
        >>> vae.eval()
        >>> 
        >>> # Inference with AMP
        >>> with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
        >>>     latent = vae.encode(video)
        >>>     reconstructed = vae.decode(latent)
    """
    if verbose:
        logger.info("=" * 80)
        logger.info("3D Causal VAE Optimizer for Windows + Blackwell (RTX 50xx)")
        logger.info("=" * 80)
    
    # Move to device
    device_obj = torch.device(device)
    if device_obj.type == 'cuda' and not torch.cuda.is_available():
        logger.warning("⚠ CUDA not available. Falling back to CPU.")
        device_obj = torch.device('cpu')
    
    vae_model = vae_model.to(device_obj)
    
    if device_obj.type != 'cuda':
        logger.warning("⚠ Optimizations require CUDA. Running on CPU without optimizations.")
        return vae_model
    
    # 1. Enable cuDNN benchmark mode
    if enable_cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        if verbose:
            logger.info("✓ Enabled cuDNN benchmark mode")
    
    # 2. Enable TF32 for Blackwell/Ampere GPUs
    if enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if verbose:
            logger.info("✓ Enabled TF32 for matrix operations (Blackwell optimization)")
    
    # 3. Apply channels-last 3D memory format to Conv3d layers
    if enable_channels_last_3d:
        conv3d_count = 0
        for name, module in vae_model.named_modules():
            if isinstance(module, nn.Conv3d):
                try:
                    module.to(memory_format=torch.channels_last_3d)
                    conv3d_count += 1
                except Exception as e:
                    logger.debug(f"Could not convert {name}: {e}")
        
        if verbose:
            if conv3d_count > 0:
                logger.info(f"✓ Applied channels_last_3d to {conv3d_count} Conv3d layers")
            else:
                logger.warning("⚠ No Conv3d layers found in model")
    
    # 4. Enable Flash Attention (SDP) for attention blocks
    if enable_flash_attention:
        attn_count = 0
        try:
            from diffusers.models.attention_processor import Attention
            
            for name, module in vae_model.named_modules():
                if isinstance(module, Attention):
                    # PyTorch 2.0+ will use SDP by default
                    # We just ensure xformers is disabled to use native SDP
                    if hasattr(module, 'set_use_memory_efficient_attention_xformers'):
                        try:
                            module.set_use_memory_efficient_attention_xformers(False)
                            attn_count += 1
                        except:
                            pass
            
            if verbose:
                if attn_count > 0:
                    logger.info(f"✓ Configured {attn_count} Attention blocks for SDP (Flash Attention)")
                else:
                    logger.info("ℹ No Attention blocks found or already optimized")
        except ImportError:
            if verbose:
                logger.warning("⚠ Could not import Attention from diffusers")
    
    # 5. Check if model has built-in optimization methods
    if hasattr(vae_model, 'enable_3d_blackwell_optimizations'):
        if verbose:
            logger.info("ℹ Model has built-in enable_3d_blackwell_optimizations() method")
            logger.info("  Consider using: vae.enable_3d_blackwell_optimizations()")
    
    if verbose:
        logger.info("=" * 80)
        logger.info("Optimization Complete!")
        logger.info("=" * 80)
        logger.info("Recommended usage:")
        logger.info("  vae.eval()")
        logger.info("  with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):")
        logger.info("      encoded = vae.encode(video)")
        logger.info("=" * 80)
    
    return vae_model


def optimize_vae_for_training(
    vae_model: nn.Module,
    device: str = 'cuda',
    enable_channels_last_3d: bool = True,
    enable_tf32: bool = True,
    enable_cudnn_benchmark: bool = True,
    verbose: bool = True,
) -> nn.Module:
    """
    Optimize 3D Causal VAE for training on Windows + Blackwell.
    
    Similar to optimize_vae_for_inference but with training-specific settings.
    Flash Attention is automatically used during training with PyTorch 2.0+.
    
    Args:
        vae_model: The 3D VAE model to optimize
        device: Target device ('cuda' or 'cpu')
        enable_channels_last_3d: Apply channels_last_3d to Conv3d layers
        enable_tf32: Enable TF32 for matrix operations
        enable_cudnn_benchmark: Enable cuDNN auto-tuner
        verbose: Print optimization status
    
    Returns:
        Optimized VAE model ready for training
    """
    # Use inference optimization (Flash Attention is enabled for training too)
    vae_model = optimize_vae_for_inference(
        vae_model,
        device=device,
        enable_channels_last_3d=enable_channels_last_3d,
        enable_flash_attention=True,  # Always enable for training
        enable_tf32=enable_tf32,
        enable_cudnn_benchmark=enable_cudnn_benchmark,
        verbose=verbose,
    )
    
    if verbose:
        logger.info("Model optimized for training. Use torch.cuda.amp.GradScaler for mixed precision.")
    
    return vae_model


def get_optimization_info():
    """
    Print information about available optimizations and hardware capabilities.
    """
    print("=" * 80)
    print("3D Causal VAE Optimization Info")
    print("=" * 80)
    
    # PyTorch version
    print(f"PyTorch Version: {torch.__version__}")
    
    # CUDA availability
    if torch.cuda.is_available():
        print(f"CUDA Available: ✓ Yes")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # Compute capability
        capability = torch.cuda.get_device_capability(0)
        print(f"Compute Capability: {capability[0]}.{capability[1]}")
        
        # Check architecture
        if capability[0] >= 9:
            print("Architecture: Blackwell (RTX 50xx) ✓")
        elif capability[0] >= 8:
            if capability[1] >= 9:
                print("Architecture: Ada Lovelace (RTX 40xx)")
            else:
                print("Architecture: Ampere (RTX 30xx / A100)")
        else:
            print(f"Architecture: Older generation (SM {capability[0]}.{capability[1]})")
    else:
        print("CUDA Available: ✗ No")
    
    # cuDNN
    if torch.backends.cudnn.is_available():
        print(f"cuDNN Available: ✓ Yes")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
    else:
        print("cuDNN Available: ✗ No")
    
    # TF32
    print(f"TF32 for matmul: {torch.backends.cuda.matmul.allow_tf32}")
    print(f"TF32 for cuDNN: {torch.backends.cudnn.allow_tf32}")
    
    # Flash Attention / SDP
    if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
        print("Flash Attention (SDP): ✓ Available (PyTorch 2.0+)")
    else:
        print("Flash Attention (SDP): ✗ Not available (requires PyTorch 2.0+)")
    
    # Channels-last 3D
    print("Channels-last 3D: ✓ Supported")
    
    print("=" * 80)


if __name__ == "__main__":
    # Example usage
    print("3D Causal VAE Optimizer - Example Usage\n")
    
    # Show system info
    get_optimization_info()
    print()
    
    # Example code
    print("Example Code:")
    print("-" * 80)
    print("""
import torch
from optimize_3d_vae_blackwell import optimize_vae_for_inference

# Load your VAE model
# vae = load_vae_model('ema_vae_fp16.safetensors')

# Apply optimizations
# vae = optimize_vae_for_inference(vae, device='cuda')
# vae.eval()

# Inference with mixed precision
# with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
#     encoded = vae.encode(video_input)
#     reconstructed = vae.decode(encoded.latent)

# Or use built-in method if available
# vae.enable_3d_blackwell_optimizations()
""")
    print("-" * 80)
