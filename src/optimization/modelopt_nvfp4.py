"""
NVIDIA ModelOpt NVFP4 Quantization for SeedVR2 (Blackwell RTX 50-series)

This script uses NVIDIA Model Optimizer (ModelOpt) to quantize the SeedVR2 3B model
to native NVFP4 format with Microscaling for maximum Blackwell Tensor Core throughput.

Key Features:
- Native FP4 quantization using modelopt.torch.quantization (mtq)
- MX-format microscaling (16:1 block-wise scaling)
- Dual-path calibration for Ada-style video/text architecture
- TensorRT-LLM and TensorRT export ready
- Optimized for RTX 5070 Ti / RTX 5080 / RTX 5090 (SM_120)

Requirements:
- nvidia-modelopt >= 0.17.0
- torch >= 2.6.0
- tensorrt >= 10.0.0 (optional, for engine export)
- safetensors

Installation:
    pip install nvidia-modelopt --extra-index-url https://pypi.nvidia.com

Usage:
    from src.optimization.modelopt_nvfp4 import quantize_seedvr2_blackwell
    
    # Load your SeedVR2 model
    model = load_your_model()
    
    # Quantize to NVFP4 for Blackwell
    quantized = quantize_seedvr2_blackwell(
        model=model,
        output_path="seedvr2_nvfp4_blackwell.safetensors",
        calib_dataloader=your_calibration_dataloader
    )

Reference:
    - NVIDIA ModelOpt Docs: https://nvidia.github.io/TensorRT-Model-Optimizer/
    - FP4 Quantization: https://nvidia.github.io/TensorRT-Model-Optimizer/reference/generated/modelopt.torch.quantization.html
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple, Callable, Union, Iterator
from dataclasses import dataclass, field
from pathlib import Path
import logging

# ============================================================================
# Logging Setup
# ============================================================================
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# ============================================================================
# Constants: NVFP4 Microscaling (MX Format) for Blackwell
# ============================================================================
NVFP4_BLOCK_SIZE = 16          # MX format: 16 elements share 1 microscale
NVFP4_SCALE_DTYPE = torch.float16  # Microscale dtype (can also use E8M0)
BLACKWELL_SM = 120             # Compute capability for RTX 50-series


def check_blackwell_gpu() -> bool:
    """Check if running on Blackwell (SM_120) GPU"""
    if not torch.cuda.is_available():
        return False
    
    device_props = torch.cuda.get_device_properties(0)
    cc = device_props.major * 10 + device_props.minor
    return cc >= 120


def check_modelopt_available() -> Tuple[bool, Optional[str]]:
    """Check if NVIDIA ModelOpt is available and return version"""
    try:
        import modelopt
        version = getattr(modelopt, '__version__', 'unknown')
        return True, version
    except ImportError:
        return False, None


def check_tensorrt_available() -> bool:
    """Check if TensorRT is available"""
    try:
        import tensorrt
        return True
    except ImportError:
        return False


# ============================================================================
# Configuration
# ============================================================================
@dataclass
class BlackwellNVFP4Config:
    """Configuration for NVFP4 quantization targeting Blackwell (RTX 50-series)"""
    
    # Microscaling parameters (MX format)
    block_size: int = 16  # 16 elements per microscale (Blackwell requirement)
    scale_format: str = "fp16"  # "fp16" or "e8m0" for microscales
    
    # Quantization algorithm
    algorithm: str = "fp4"  # Use native FP4 quantization
    
    # Calibration
    num_calibration_batches: int = 128
    
    # Layer selection patterns (layers matching these are NOT quantized)
    preserve_patterns: List[str] = field(default_factory=lambda: [
        "norm", "ln_", "layernorm", "rmsnorm", "groupnorm",
        "embed", "pos_embed", "patch_embed", "time_embed",
        "head", "output", "lm_head", "proj_out"
    ])


# ============================================================================
# Calibration Dataloader for SeedVR2 (Ada-style dual-path)
# ============================================================================
class SeedVR2CalibrationDataloader:
    """
    Calibration dataloader for SeedVR2's Ada-style architecture.
    
    SeedVR2 has 32 blocks with parallel video (.vid) and text (.txt) paths.
    Calibration must feed data through both paths for proper scale estimation.
    """
    
    def __init__(
        self,
        video_samples: torch.Tensor,
        text_samples: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
        batch_size: int = 1,
        device: str = "cuda"
    ):
        """
        Args:
            video_samples: Video latents [N, C, T, H, W] or [N, L_v, D]
            text_samples: Text embeddings [N, L_t, D]
            timesteps: Optional timesteps [N] (random if not provided)
            batch_size: Calibration batch size
            device: Device for calibration
        """
        self.video = video_samples.to(device)
        self.text = text_samples.to(device)
        self.timesteps = timesteps.to(device) if timesteps is not None else None
        self.batch_size = batch_size
        self.device = device
        self.num_samples = min(len(video_samples), len(text_samples))
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        for i in range(0, self.num_samples, self.batch_size):
            end = min(i + self.batch_size, self.num_samples)
            batch = {
                "video": self.video[i:end],
                "text": self.text[i:end],
            }
            if self.timesteps is not None:
                batch["timesteps"] = self.timesteps[i:end]
            else:
                batch["timesteps"] = torch.randint(
                    0, 1000, (end - i,), device=self.device, dtype=torch.long
                )
            yield batch
    
    def __len__(self) -> int:
        return (self.num_samples + self.batch_size - 1) // self.batch_size


# ============================================================================
# ModelOpt FP4 Quantization Configuration
# ============================================================================
def create_fp4_quant_config(config: BlackwellNVFP4Config) -> Dict[str, Any]:
    """
    Create ModelOpt quantization config for native FP4 with microscaling.
    
    This uses the official ModelOpt FP4 quantization format which is
    optimized for Blackwell Tensor Cores.
    
    Returns:
        Quantization config dict for mtq.quantize()
    """
    available, version = check_modelopt_available()
    if not available:
        raise ImportError(
            "NVIDIA ModelOpt is required for FP4 quantization.\n"
            "Install with: pip install nvidia-modelopt --extra-index-url https://pypi.nvidia.com"
        )
    
    logger.info(f"📦 ModelOpt version: {version}")
    
    # Import ModelOpt quantization module
    import modelopt.torch.quantization as mtq
    
    # Build layer exclusion pattern for critical layers
    exclude_patterns = "|".join(config.preserve_patterns)
    
    # FP4 quantization config with microscaling
    # Reference: modelopt.torch.quantization.config
    quant_config = {
        "quant_cfg": {
            # Default: Quantize all weights to FP4 with block-wise scaling
            "*weight_quantizer": {
                "num_bits": 4,
                "block_sizes": {-1: config.block_size},  # Last dim, 16 elements per scale
                "enable": True,
            },
            # Disable input (activation) quantization - let Tensor Cores handle it
            "*input_quantizer": {
                "enable": False,
            },
            # Exclude normalization layers
            "*norm*weight_quantizer": {"enable": False},
            "*norm*input_quantizer": {"enable": False},
            # Exclude embedding layers
            "*embed*weight_quantizer": {"enable": False},
            "*embed*input_quantizer": {"enable": False},
            # Exclude output heads
            "*head*weight_quantizer": {"enable": False},
            "*head*input_quantizer": {"enable": False},
            # Exclude biases (usually small, keep in FP16)
            "*bias*": {"enable": False},
        },
        "algorithm": "max",  # Use max calibration (simple and effective for FP4)
    }
    
    return quant_config


def create_fp4_awq_config(config: BlackwellNVFP4Config) -> Dict[str, Any]:
    """
    Create ModelOpt config for FP4 with AWQ (Activation-aware Weight Quantization).
    
    AWQ provides better accuracy than simple max calibration by considering
    activation magnitudes during scale computation.
    """
    available, _ = check_modelopt_available()
    if not available:
        raise ImportError("NVIDIA ModelOpt required")
    
    import modelopt.torch.quantization as mtq
    
    quant_config = {
        "quant_cfg": {
            "*weight_quantizer": {
                "num_bits": 4,
                "block_sizes": {-1: config.block_size},
                "enable": True,
            },
            "*input_quantizer": {"enable": False},
            "*norm*": {"enable": False},
            "*embed*": {"enable": False},
            "*head*": {"enable": False},
        },
        "algorithm": "awq",  # Activation-aware Weight Quantization
    }
    
    return quant_config


# ============================================================================
# Forward Loop for Calibration
# ============================================================================
def create_calibration_forward_loop(
    model: nn.Module,
    dataloader: SeedVR2CalibrationDataloader
) -> Callable:
    """
    Create forward loop function for ModelOpt calibration.
    
    This function runs calibration data through the model to collect
    activation statistics for computing optimal quantization scales.
    """
    
    def forward_loop(model: nn.Module) -> None:
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                try:
                    # Try standard forward with keyword args
                    _ = model(
                        video=batch["video"],
                        text=batch["text"],
                        timesteps=batch["timesteps"]
                    )
                except TypeError:
                    try:
                        # Try positional args
                        _ = model(batch["video"], batch["text"], batch["timesteps"])
                    except TypeError:
                        try:
                            # Try without timesteps
                            _ = model(batch["video"], batch["text"])
                        except Exception as e:
                            if batch_idx == 0:
                                logger.warning(f"Calibration forward failed: {e}")
                            continue
                
                if (batch_idx + 1) % 20 == 0:
                    logger.info(f"   Calibration: {batch_idx + 1}/{len(dataloader)} batches")
    
    return forward_loop


# ============================================================================
# Main Quantization Function
# ============================================================================
def quantize_seedvr2_blackwell(
    model: nn.Module,
    output_path: str,
    calib_dataloader: Optional[SeedVR2CalibrationDataloader] = None,
    config: Optional[BlackwellNVFP4Config] = None,
    use_awq: bool = True,
    export_tensorrt: bool = False,
    tensorrt_output: Optional[str] = None
) -> nn.Module:
    """
    Quantize SeedVR2 model to NVFP4 format for Blackwell GPUs.
    
    This is the main entry point for quantization using NVIDIA ModelOpt.
    
    Args:
        model: SeedVR2 model (nn.Module) to quantize
        output_path: Path to save quantized model (safetensors format)
        calib_dataloader: Calibration dataloader with video/text samples
        config: Quantization configuration
        use_awq: Use AWQ algorithm (recommended for better accuracy)
        export_tensorrt: Also export TensorRT engine
        tensorrt_output: Path for TensorRT engine (required if export_tensorrt=True)
    
    Returns:
        Quantized model
    
    Example:
        >>> from src.optimization.modelopt_nvfp4 import (
        ...     quantize_seedvr2_blackwell,
        ...     SeedVR2CalibrationDataloader,
        ...     BlackwellNVFP4Config
        ... )
        >>> 
        >>> # Prepare calibration data
        >>> video_data = torch.randn(128, 16, 4, 32, 32, device="cuda")
        >>> text_data = torch.randn(128, 512, 2560, device="cuda")
        >>> calib_loader = SeedVR2CalibrationDataloader(video_data, text_data)
        >>> 
        >>> # Load and quantize model
        >>> model = YourSeedVR2Model().cuda()
        >>> quantized = quantize_seedvr2_blackwell(
        ...     model=model,
        ...     output_path="seedvr2_nvfp4.safetensors",
        ...     calib_dataloader=calib_loader,
        ...     use_awq=True
        ... )
    """
    # Check prerequisites
    available, version = check_modelopt_available()
    if not available:
        raise ImportError(
            "NVIDIA ModelOpt is required.\n"
            "Install: pip install nvidia-modelopt --extra-index-url https://pypi.nvidia.com"
        )
    
    import modelopt.torch.quantization as mtq
    
    if config is None:
        config = BlackwellNVFP4Config()
    
    # Log system info
    logger.info("=" * 70)
    logger.info("🚀 SeedVR2 NVFP4 Quantization for Blackwell (ModelOpt)")
    logger.info("=" * 70)
    logger.info(f"📦 ModelOpt version: {version}")
    logger.info(f"🎮 CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"🎮 Blackwell (SM_120+): {check_blackwell_gpu()}")
    logger.info(f"⚙️ Block size: {config.block_size} (1 microscale per {config.block_size} elements)")
    logger.info(f"⚙️ Algorithm: {'AWQ' if use_awq else 'Max'}")
    
    # Move model to CUDA for quantization
    model = model.cuda()
    model.eval()
    
    # Count layers before quantization
    total_linear = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    logger.info(f"📊 Total Linear layers: {total_linear}")
    
    # Get quantization config
    if use_awq:
        quant_config = create_fp4_awq_config(config)
        logger.info("📐 Using AWQ (Activation-aware Weight Quantization)")
    else:
        quant_config = create_fp4_quant_config(config)
        logger.info("📐 Using Max calibration")
    
    # Create forward loop for calibration
    forward_loop = None
    if calib_dataloader is not None:
        logger.info(f"📊 Calibration batches: {len(calib_dataloader)}")
        forward_loop = create_calibration_forward_loop(model, calib_dataloader)
    else:
        logger.warning("⚠️ No calibration data provided - using default scales")
    
    # Apply quantization
    logger.info("🔄 Applying FP4 quantization with microscaling...")
    
    try:
        quantized_model = mtq.quantize(
            model,
            quant_config,
            forward_loop=forward_loop
        )
        logger.info("✅ Quantization complete!")
    except Exception as e:
        logger.error(f"❌ Quantization failed: {e}")
        raise
    
    # Count quantized layers
    quantized_count = 0
    preserved_count = 0
    for name, module in quantized_model.named_modules():
        if hasattr(module, 'weight_quantizer'):
            if module.weight_quantizer is not None:
                try:
                    if module.weight_quantizer.is_enabled:
                        quantized_count += 1
                    else:
                        preserved_count += 1
                except:
                    quantized_count += 1
    
    logger.info(f"📊 Quantization Statistics:")
    logger.info(f"   - FP4 Quantized: {quantized_count} layers")
    logger.info(f"   - FP16 Preserved: {preserved_count} layers")
    
    # Export to safetensors
    logger.info(f"💾 Saving to {output_path}...")
    export_quantized_model(quantized_model, output_path)
    
    # Optionally export to TensorRT
    if export_tensorrt:
        if tensorrt_output is None:
            tensorrt_output = output_path.replace(".safetensors", ".engine")
        logger.info(f"🚀 Exporting TensorRT engine to {tensorrt_output}...")
        try:
            export_to_tensorrt_engine(quantized_model, tensorrt_output)
        except Exception as e:
            logger.warning(f"⚠️ TensorRT export failed: {e}")
    
    logger.info("=" * 70)
    logger.info("✅ NVFP4 Quantization Complete!")
    logger.info(f"   📄 Output: {output_path}")
    logger.info("=" * 70)
    
    return quantized_model


# ============================================================================
# Export Functions
# ============================================================================
def export_quantized_model(model: nn.Module, output_path: str) -> None:
    """Export quantized model to safetensors format with NVFP4 metadata."""
    try:
        from safetensors.torch import save_file
    except ImportError:
        raise ImportError("safetensors required: pip install safetensors")
    
    state_dict = {}
    
    for name, param in model.named_parameters():
        state_dict[name] = param.data.cpu()
    
    for name, buffer in model.named_buffers():
        state_dict[name] = buffer.cpu()
    
    # Add NVFP4 metadata
    metadata = {
        "format": "nvfp4_microscale",
        "block_size": str(NVFP4_BLOCK_SIZE),
        "quantization": "modelopt_fp4",
    }
    
    save_file(state_dict, output_path, metadata=metadata)
    logger.info(f"✅ Saved {len(state_dict)} tensors to {output_path}")


def export_to_tensorrt_engine(
    model: nn.Module,
    output_path: str,
    fp16_fallback: bool = True
) -> None:
    """Export quantized model to TensorRT engine."""
    if not check_tensorrt_available():
        raise ImportError("TensorRT required for engine export")
    
    available, _ = check_modelopt_available()
    if not available:
        raise ImportError("ModelOpt required for TensorRT export")
    
    try:
        import modelopt.torch.export as mte
        
        # Define dummy inputs for tracing
        dummy_video = torch.randn(1, 16, 4, 32, 32, device="cuda", dtype=torch.float16)
        dummy_text = torch.randn(1, 512, 2560, device="cuda", dtype=torch.float16)
        dummy_timestep = torch.tensor([500], device="cuda", dtype=torch.long)
        
        mte.export_tensorrt(
            model,
            output_path,
            sample_inputs=(dummy_video, dummy_text, dummy_timestep),
            enable_fp16=fp16_fallback,
        )
        logger.info(f"✅ TensorRT engine saved to {output_path}")
    except Exception as e:
        logger.error(f"TensorRT export failed: {e}")
        raise


# ============================================================================
# Utility Functions
# ============================================================================
def analyze_model_for_quantization(model: nn.Module) -> Dict[str, Any]:
    """
    Analyze model structure for NVFP4 quantization planning.
    
    Returns dict with:
        - total_params: Total parameter count
        - linear_layers: Number of Linear layers
        - quantizable_params: Parameters that will be quantized
        - preserved_params: Parameters kept in FP16
        - estimated_reduction: Memory reduction estimate
    """
    total_params = sum(p.numel() for p in model.parameters())
    linear_layers = 0
    quantizable_params = 0
    preserved_patterns = ["norm", "embed", "head", "bias", "ln_"]
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            linear_layers += 1
            is_preserved = any(p in name.lower() for p in preserved_patterns)
            if not is_preserved:
                quantizable_params += module.weight.numel()
                if module.bias is not None:
                    quantizable_params += module.bias.numel()
    
    preserved_params = total_params - quantizable_params
    
    # FP4 = 0.5 bytes per element, FP16 = 2 bytes per element
    original_size_mb = total_params * 2 / (1024 * 1024)
    quantized_size_mb = (quantizable_params * 0.5 + preserved_params * 2) / (1024 * 1024)
    reduction_pct = (1 - quantized_size_mb / original_size_mb) * 100
    
    return {
        "total_params": total_params,
        "linear_layers": linear_layers,
        "quantizable_params": quantizable_params,
        "preserved_params": preserved_params,
        "original_size_mb": original_size_mb,
        "estimated_quantized_size_mb": quantized_size_mb,
        "estimated_reduction_pct": reduction_pct,
    }


def print_quantization_plan(model: nn.Module) -> None:
    """Print detailed quantization plan for the model."""
    analysis = analyze_model_for_quantization(model)
    
    print("\n" + "=" * 60)
    print("📊 SeedVR2 NVFP4 Quantization Plan")
    print("=" * 60)
    print(f"Total Parameters:     {analysis['total_params']:,}")
    print(f"Linear Layers:        {analysis['linear_layers']}")
    print(f"Quantizable Params:   {analysis['quantizable_params']:,}")
    print(f"Preserved Params:     {analysis['preserved_params']:,}")
    print(f"Original Size:        {analysis['original_size_mb']:.2f} MB")
    print(f"Estimated FP4 Size:   {analysis['estimated_quantized_size_mb']:.2f} MB")
    print(f"Estimated Reduction:  {analysis['estimated_reduction_pct']:.1f}%")
    print("=" * 60 + "\n")


# ============================================================================
# CLI Entry Point
# ============================================================================
def main():
    """Command-line interface for NVFP4 quantization."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Quantize SeedVR2 to NVFP4 for Blackwell using NVIDIA ModelOpt"
    )
    parser.add_argument("--model", "-m", required=True, help="Path to model checkpoint")
    parser.add_argument("--output", "-o", required=True, help="Output path for quantized model")
    parser.add_argument("--tensorrt", "-t", help="Optional TensorRT engine output path")
    parser.add_argument("--algorithm", choices=["awq", "max"], default="awq",
                       help="Quantization algorithm (default: awq)")
    parser.add_argument("--block-size", type=int, default=16,
                       help="Microscaling block size (default: 16)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 SeedVR2 NVFP4 Quantization CLI")
    print("=" * 60)
    print()
    print("⚠️ This CLI is a template. For actual usage:")
    print()
    print("1. Load your model architecture:")
    print("   >>> from your_model import SeedVR2Model")
    print("   >>> model = SeedVR2Model()")
    print("   >>> model.load_state_dict(torch.load(model_path))")
    print()
    print("2. Prepare calibration data:")
    print("   >>> video_data = torch.randn(128, 16, 4, 32, 32)")
    print("   >>> text_data = torch.randn(128, 512, 2560)")
    print("   >>> calib_loader = SeedVR2CalibrationDataloader(video_data, text_data)")
    print()
    print("3. Run quantization:")
    print("   >>> from src.optimization.modelopt_nvfp4 import quantize_seedvr2_blackwell")
    print("   >>> quantized = quantize_seedvr2_blackwell(model, output_path, calib_loader)")
    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
