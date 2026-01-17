#!/usr/bin/env python3
"""
NVFP4 Model Conversion Launcher for SeedVR2 on Blackwell (RTX 50-series)

This script sets up the Python path correctly and launches NVFP4 quantization
using NVIDIA ModelOpt for optimal Blackwell Tensor Core performance.

Usage (from the custom_nodes/seedvr2_videoupscaler directory):
    Windows (ComfyUI embedded Python):
        D:\\ComfyUI\\python_embeded\\python.exe convert_to_nvfp4.py --model seedvr2_3b.safetensors --output seedvr2_nvfp4.safetensors
    
    Windows (system Python):
        python convert_to_nvfp4.py --model seedvr2_3b.safetensors --output seedvr2_nvfp4.safetensors
    
    Linux/Mac:
        python3 convert_to_nvfp4.py --model seedvr2_3b.safetensors --output seedvr2_nvfp4.safetensors

Options:
    --model         : Path to input model (safetensors format)
    --output        : Path for output NVFP4 model
    --use-awq       : Enable AWQ (Activation-aware Weight Quantization) for best accuracy
    --calib-samples : Number of calibration samples (default: 128)
    --export-trt    : Also export TensorRT engine

Requirements:
    pip install nvidia-modelopt --extra-index-url https://pypi.nvidia.com
    pip install safetensors torch>=2.6.0

Author: Copilot
Date: 2026-01-17
"""

import os
import sys
from pathlib import Path

# ============================================================================
# CRITICAL: Set up sys.path before any other imports
# ============================================================================

# Get the script's directory (should be the root of seedvr2_videoupscaler)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR

# Add project root to sys.path so imports work correctly
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Also add ComfyUI root if we're in custom_nodes
COMFYUI_ROOT = PROJECT_ROOT.parent.parent  # custom_nodes/../..
if (COMFYUI_ROOT / "comfy").exists():
    if str(COMFYUI_ROOT) not in sys.path:
        sys.path.insert(0, str(COMFYUI_ROOT))

print(f"🔧 Project root: {PROJECT_ROOT}")
print(f"🔧 Python path configured: {sys.path[:3]}...")

# ============================================================================
# Now safe to import project modules
# ============================================================================

import argparse
import torch
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_dependencies():
    """Verify all required dependencies are available"""
    print("\n📦 Checking dependencies...")
    
    # Check torch
    print(f"   ✅ PyTorch {torch.__version__}")
    
    # Check CUDA
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        cc = torch.cuda.get_device_capability(0)
        print(f"   ✅ CUDA {torch.version.cuda} - {gpu_name} (SM_{cc[0]}{cc[1]})")
        
        # Check for Blackwell (SM_120+)
        if cc[0] * 10 + cc[1] >= 120:
            print(f"   🚀 BLACKWELL GPU DETECTED - NVFP4 Native Support Available!")
        else:
            print(f"   ⚠️ Not a Blackwell GPU - NVFP4 may have limited support")
    else:
        print("   ❌ CUDA not available")
        return False
    
    # Check safetensors
    try:
        import safetensors
        print(f"   ✅ safetensors {getattr(safetensors, '__version__', 'unknown')}")
    except ImportError:
        print("   ❌ safetensors not installed: pip install safetensors")
        return False
    
    # Check ModelOpt
    try:
        import modelopt
        version = getattr(modelopt, '__version__', 'unknown')
        print(f"   ✅ nvidia-modelopt {version}")
    except ImportError:
        print("   ❌ nvidia-modelopt not installed")
        print("   📥 Install with: pip install nvidia-modelopt --extra-index-url https://pypi.nvidia.com")
        return False
    
    return True


def load_seedvr2_model(model_path: str):
    """
    Load SeedVR2 model from safetensors checkpoint
    
    Args:
        model_path: Path to the model checkpoint
        
    Returns:
        Loaded model on CUDA
    """
    from safetensors.torch import load_file
    
    print(f"\n📥 Loading model from: {model_path}")
    
    # Determine model size from filename
    model_name = Path(model_path).stem.lower()
    is_7b = "7b" in model_name
    model_size = "7B" if is_7b else "3B"
    
    print(f"   🎯 Detected model size: {model_size}")
    
    # Import the appropriate model class
    if is_7b:
        from src.models.dit_7b.nadit import NaDiT
        # 7B model config
        config = {
            'dim': 3456,
            'n_blocks': 35,
            'n_heads': 24,
            'ffn_ratio': 4.0,
        }
    else:
        from src.models.dit_3b.nadit import NaDiT
        # 3B model config
        config = {
            'dim': 2560,
            'n_blocks': 32,
            'n_heads': 20,
            'ffn_ratio': 4.0,
        }
    
    print(f"   🔧 Creating NaDiT model with config: {config}")
    
    # Create model on meta device first
    with torch.device('meta'):
        model = NaDiT(**config)
    
    # Load state dict
    print(f"   📦 Loading weights...")
    state_dict = load_file(model_path)
    
    # Materialize model on CUDA
    print(f"   🚀 Materializing to CUDA...")
    model = model.to_empty(device='cuda')
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   ✅ Model loaded: {total_params:,} parameters")
    
    return model


def create_calibration_data(model, num_samples: int = 128):
    """
    Create calibration data for AWQ quantization
    
    SeedVR2 uses Ada-style architecture with separate video and text paths
    """
    print(f"\n📊 Creating calibration data ({num_samples} samples)...")
    
    # Import calibration dataloader
    from src.optimization.modelopt_nvfp4 import SeedVR2CalibrationDataloader
    
    # Get model dimension
    if hasattr(model, 'dim'):
        dim = model.dim
    else:
        # Try to infer from first linear layer
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                dim = module.in_features if module.in_features > 256 else module.out_features
                break
        else:
            dim = 2560  # Default for 3B
    
    print(f"   🔧 Model dimension: {dim}")
    
    # Create synthetic calibration data
    # Video path: (batch, frames, channels, height, width)
    video_data = torch.randn(num_samples, 16, 4, 32, 32, device='cuda', dtype=torch.bfloat16)
    
    # Text path: (batch, seq_len, embed_dim)  
    text_data = torch.randn(num_samples, 512, dim, device='cuda', dtype=torch.bfloat16)
    
    calib_loader = SeedVR2CalibrationDataloader(
        video_data=video_data,
        text_data=text_data,
        batch_size=8
    )
    
    print(f"   ✅ Created calibration dataloader")
    
    return calib_loader


def quantize_model(
    model,
    output_path: str,
    calib_loader,
    use_awq: bool = True,
    export_tensorrt: bool = False
):
    """
    Quantize model to NVFP4 using NVIDIA ModelOpt
    
    Args:
        model: SeedVR2 model to quantize
        output_path: Path for output NVFP4 model
        calib_loader: Calibration dataloader
        use_awq: Whether to use AWQ algorithm
        export_tensorrt: Whether to also export TensorRT engine
    """
    from src.optimization.modelopt_nvfp4 import quantize_seedvr2_blackwell
    
    print(f"\n🚀 Starting NVFP4 quantization...")
    print(f"   📍 Output path: {output_path}")
    print(f"   🔧 AWQ enabled: {use_awq}")
    print(f"   🔧 TensorRT export: {export_tensorrt}")
    
    # Run quantization
    quantized_model = quantize_seedvr2_blackwell(
        model=model,
        output_path=output_path,
        calib_dataloader=calib_loader,
        use_awq=use_awq,
        export_tensorrt=export_tensorrt,
        microscale_block_size=16  # Blackwell optimal
    )
    
    print(f"\n✅ Quantization complete!")
    print(f"   📁 NVFP4 model saved to: {output_path}")
    
    return quantized_model


def main():
    """Main entry point for NVFP4 conversion"""
    parser = argparse.ArgumentParser(
        description="Convert SeedVR2 model to NVFP4 for Blackwell GPUs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic conversion with AWQ
    python convert_to_nvfp4.py --model seedvr2_3b.safetensors --output seedvr2_nvfp4.safetensors --use-awq
    
    # With TensorRT export
    python convert_to_nvfp4.py --model seedvr2_3b.safetensors --output seedvr2_nvfp4.safetensors --use-awq --export-trt
    
    # ComfyUI embedded Python
    D:\\ComfyUI\\python_embeded\\python.exe convert_to_nvfp4.py --model seedvr2_3b.safetensors --output seedvr2_nvfp4.safetensors
        """
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        help='Path to input SeedVR2 model (safetensors format)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Path for output NVFP4 model'
    )
    
    parser.add_argument(
        '--use-awq',
        action='store_true',
        default=True,
        help='Use AWQ (Activation-aware Weight Quantization) for best accuracy (default: True)'
    )
    
    parser.add_argument(
        '--no-awq',
        action='store_true',
        help='Disable AWQ, use standard FP4 quantization'
    )
    
    parser.add_argument(
        '--calib-samples',
        type=int,
        default=128,
        help='Number of calibration samples for AWQ (default: 128)'
    )
    
    parser.add_argument(
        '--export-trt',
        action='store_true',
        help='Also export TensorRT engine'
    )
    
    args = parser.parse_args()
    
    # Banner
    print("=" * 70)
    print("🚀 SeedVR2 NVFP4 Conversion for NVIDIA Blackwell (RTX 50-series)")
    print("=" * 70)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Missing dependencies. Please install them and try again.")
        sys.exit(1)
    
    # Check model file exists
    model_path = Path(args.model)
    if not model_path.exists():
        # Try relative to ComfyUI models directory
        comfyui_models = PROJECT_ROOT.parent.parent / "models" / "SEEDVR2"
        alt_path = comfyui_models / args.model
        if alt_path.exists():
            model_path = alt_path
        else:
            print(f"\n❌ Model file not found: {args.model}")
            print(f"   Also tried: {alt_path}")
            sys.exit(1)
    
    print(f"\n📁 Input model: {model_path}")
    
    try:
        # Load model
        model = load_seedvr2_model(str(model_path))
        
        # Create calibration data
        use_awq = args.use_awq and not args.no_awq
        if use_awq:
            calib_loader = create_calibration_data(model, args.calib_samples)
        else:
            calib_loader = None
        
        # Quantize
        quantize_model(
            model=model,
            output_path=args.output,
            calib_loader=calib_loader,
            use_awq=use_awq,
            export_tensorrt=args.export_trt
        )
        
        print("\n" + "=" * 70)
        print("✅ NVFP4 CONVERSION COMPLETE!")
        print("=" * 70)
        print(f"\n🎯 Your Blackwell-optimized model is ready at: {args.output}")
        print("\n📋 To use with ComfyUI:")
        print(f"   1. Copy '{args.output}' to your ComfyUI/models/SEEDVR2/ directory")
        print(f"   2. Select it in the SeedVR2 (Down)Load DiT Model node")
        print(f"   3. Enable 'nvfp4' in the optimization settings")
        print("\n🚀 Enjoy up to 2x speedup on your RTX 50-series GPU!")
        
    except Exception as e:
        print(f"\n❌ Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
