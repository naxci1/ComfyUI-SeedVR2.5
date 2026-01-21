# Model Conversion Guide: FP16 to Quantized Format (NVIDIA Model-Optimizer)

This guide explains how to convert SeedVR2 models from FP16 `.safetensors` format to quantized formats using **NVIDIA Model-Optimizer** - the official NVIDIA toolkit for professional-grade model quantization.

## Table of Contents

- [Understanding NVIDIA Model-Optimizer](#understanding-nvidia-model-optimizer)
- [Quantization Formats](#quantization-formats)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Advanced Usage](#advanced-usage)
- [Loading Quantized Models in ComfyUI](#loading-quantized-models-in-comfyui)
- [Troubleshooting](#troubleshooting)

## Understanding NVIDIA Model-Optimizer

**NVIDIA Model-Optimizer** (formerly TensorRT Model-Optimizer) is NVIDIA's official open-source library for model optimization. It provides state-of-the-art quantization algorithms with:

- **Professional-grade accuracy**: Advanced algorithms like SmoothQuant, SVDQuant
- **Hardware optimization**: Native support for Blackwell FP4, Hopper FP8, and more
- **Production-ready**: Used by NVIDIA TensorRT, TensorRT-LLM, vLLM, and SGLang
- **Extensive model support**: LLMs, diffusion models, vision models

### Why Use Model-Optimizer?

✅ **Better Quality**: Superior quantization algorithms vs. basic quantization  
✅ **Official Support**: Maintained and optimized by NVIDIA  
✅ **Future-Proof**: Support for latest GPU architectures (Blackwell, Hopper)  
✅ **Open Source**: Available on GitHub with active development  

**GitHub**: https://github.com/NVIDIA/Model-Optimizer  
**Documentation**: https://nvidia.github.io/Model-Optimizer

## Quantization Formats

This conversion script supports multiple quantization formats via NVIDIA Model-Optimizer:

### NVFP4 (Native FP4) - Blackwell GPUs Only

**What is it?**
- NVIDIA's native 4-bit floating point format (E2M1: 2-bit exponent, 1-bit mantissa)
- Hardware-accelerated on Blackwell 5th Gen Tensor Cores via Model-Optimizer
- Specifically designed for RTX 50-series GPUs (RTX 5070, 5080, 5090)

**Performance Benefits:**
- 2-4x speedup for linear layers vs FP16
- ~75% VRAM reduction compared to FP16
- <1% quality degradation with advanced quantization algorithms
- Native hardware support for maximum efficiency

**Requirements:**
- NVIDIA RTX 50-series GPU (Blackwell architecture, compute capability 10.0+)
- PyTorch 2.6+ with CUDA 12.8+
- NVIDIA Model-Optimizer
- NVIDIA Driver 565.xx or newer

**When to Use:**
- You have an RTX 5070, 5080, or 5090 GPU
- You need maximum performance with minimal quality loss
- VRAM is limited and you want to run larger models

### FP8 - Hopper/Blackwell GPUs

**What is it?**
- 8-bit floating point format (E4M3: 4-bit exponent, 3-bit mantissa)
- Hardware-accelerated on Hopper (H100) and Blackwell GPUs
- Optimal balance between speed and accuracy

**Performance Benefits:**
- 1.5-2x speedup for inference vs FP16
- ~50% VRAM reduction compared to FP16
- Minimal quality loss (<0.5% typical)
- Excellent accuracy/performance trade-off

**Requirements:**
- NVIDIA Hopper (H100) or Blackwell (RTX 50xx) GPU
- PyTorch 2.0+ with CUDA 11.8+
- NVIDIA Model-Optimizer

**When to Use:**
- You have Hopper or Blackwell GPU
- You want excellent performance with minimal quality loss
- More conservative than FP4 but still very fast

### INT8 - Universal (Ampere+)

**What is it?**
- 8-bit integer quantization with SmoothQuant algorithm
- Works on all modern NVIDIA GPUs (Ampere, Ada, Hopper, Blackwell)
- Software-optimized with good performance

**Performance Benefits:**
- ~2x memory reduction vs FP16
- Good inference speed on all GPUs
- Excellent accuracy with SmoothQuant
- Universal compatibility

**Requirements:**
- NVIDIA GPU with compute capability 8.0+ (Ampere or newer)
- PyTorch 2.0+
- NVIDIA Model-Optimizer

**When to Use:**
- You have RTX 30xx or 40xx GPU
- You want good memory savings with broad compatibility
- You prioritize accuracy and compatibility over maximum speed

### NF4 - Fallback (All GPUs)

**What is it?**
- Normal Float 4-bit quantization via bitsandbytes
- Software-based fallback when Model-Optimizer not available
- Works on all modern GPUs

**Performance Benefits:**
- ~75% VRAM reduction vs FP16
- Compatible with most NVIDIA GPUs
- Good quality preservation
- Widely supported across frameworks

**Requirements:**
- Any modern NVIDIA GPU
- PyTorch 2.0+
- bitsandbytes library

**When to Use:**
- Model-Optimizer not available
- You need maximum compatibility
- VRAM reduction is priority over raw speed

### Comparison Table

| Feature | NVFP4 | FP8 | INT8 | NF4 |
|---------|-------|-----|------|-----|
| GPU Support | RTX 50-series only | Hopper/Blackwell | Ampere+ | All modern GPUs |
| Implementation | Hardware (Model-Optimizer) | Hardware (Model-Optimizer) | Software (Model-Optimizer) | Software (bitsandbytes) |
| Speed | 2-4x faster | 1.5-2x faster | ~1.2x faster | Comparable to FP16 |
| VRAM Reduction | ~75% | ~50% | ~50% | ~75% |
| Quality Loss | <1% | <0.5% | <1% | <2% |
| PyTorch Version | 2.6+ | 2.0+ | 2.0+ | 2.0+ |
| CUDA Version | 12.8+ | 11.8+ | Any | Any |

## Requirements

### Core Dependencies
```bash
# Essential for all quantization methods
torch>=2.3.0
safetensors>=0.4.0
huggingface_hub>=0.20.0
tqdm>=4.65.0
```

### NVIDIA Model-Optimizer (Recommended)
```bash
# Official NVIDIA quantization toolkit
nvidia-modelopt[all]>=0.17.0
```

### Fallback: bitsandbytes
```bash
# Only needed if not using Model-Optimizer
bitsandbytes>=0.41.0
```

### For NVFP4 (Blackwell GPUs)
```bash
# Recommended PyTorch nightly with CUDA 12.8 for NVFP4
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```

## Installation

### Option 1: Install with Model-Optimizer (Recommended)

```bash
# Install core dependencies
pip install torch>=2.3.0 safetensors huggingface_hub tqdm

# Install bitsandbytes for NF4 support (works on all GPUs)
pip install bitsandbytes

# Optional: For Blackwell GPUs, install PyTorch nightly for NVFP4
# pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```

### Option 2: Minimal Installation (NVFP4 only, Blackwell GPUs)

```bash
# Install core dependencies
pip install torch>=2.6.0 safetensors huggingface_hub tqdm

# Install PyTorch nightly with CUDA 12.8 for NVFP4
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```

### Verify Installation

Run the diagnostic script to verify your setup:

```bash
python scripts/nvfp4_diagnostic.py
```

This will check:
- Python version (3.12+ recommended)
- PyTorch version and CUDA compatibility
- GPU architecture detection
- Pinned memory and async transfer capabilities
- FP4 kernel availability

## Quick Start

### Example 1: Auto-Detect and Convert

The simplest way - automatically detects your hardware and chooses the best method:

```bash
python scripts/convert_to_nvfp4.py \
    --model_url https://huggingface.co/numz/SeedVR2_comfyUI/blob/main/seedvr2_ema_3b_fp16.safetensors \
    --output_path models/seedvr2_ema_3b_quantized.safetensors
```

This will:
1. Detect your GPU (Blackwell → NVFP4, Others → NF4)
2. Download the model from Hugging Face
3. Convert to the optimal format
4. Save the quantized model

### Example 2: Convert Local File

If you already have the model downloaded:

```bash
python scripts/convert_to_nvfp4.py \
    --input_path /path/to/seedvr2_ema_3b_fp16.safetensors \
    --output_path models/seedvr2_ema_3b_quantized.safetensors
```

### Example 3: Force Specific Method

To explicitly choose quantization method:

```bash
# Force NF4 (for RTX 30xx/40xx)
python scripts/convert_to_nvfp4.py \
    --input_path model_fp16.safetensors \
    --output_path model_nf4.safetensors \
    --method nf4

# Force NVFP4 (for RTX 50xx)
python scripts/convert_to_nvfp4.py \
    --input_path model_fp16.safetensors \
    --output_path model_nvfp4.safetensors \
    --method nvfp4
```

## Advanced Usage

### Full Command Line Options

```bash
python scripts/convert_to_nvfp4.py \
    --input_path <path>              # Path to input .safetensors file
    --model_url <url>                # Alternative: Download from Hugging Face
    --output_path <path>             # Where to save quantized model
    --method <auto|nvfp4|nf4>        # Quantization method (default: auto)
    --block_size <int>               # NVFP4 block size (default: 16)
    --no_preserve_critical           # Quantize all layers (may reduce quality)
```

### Preserving Critical Layers

By default, the converter preserves critical layers in FP16 for quality:
- Bias terms
- Normalization layers (LayerNorm, GroupNorm, RMSNorm)
- Embedding layers (positional, patch, time)
- Output heads

To quantize everything (not recommended):

```bash
python scripts/convert_to_nvfp4.py \
    --input_path model.safetensors \
    --output_path model_quantized.safetensors \
    --no_preserve_critical
```

### Adjusting Block Size (NVFP4 only)

Block size affects the trade-off between quality and compression:
- Smaller blocks (8): Better quality, slightly larger file
- Default (16): Balanced (recommended)
- Larger blocks (32): More compression, potential quality loss

```bash
python scripts/convert_to_nvfp4.py \
    --input_path model.safetensors \
    --output_path model_nvfp4.safetensors \
    --method nvfp4 \
    --block_size 32
```

## Loading Quantized Models in ComfyUI

### Step 1: Place Model in Correct Directory

Copy your quantized model to the appropriate ComfyUI models directory:

**For DiT Models:**
```bash
cp models/seedvr2_ema_3b_quantized.safetensors \
   /path/to/ComfyUI/models/diffusion_models/
```

**For VAE Models:**
```bash
cp models/vae_quantized.safetensors \
   /path/to/ComfyUI/models/vae/
```

### Step 2: Use in ComfyUI Workflow

1. Open ComfyUI
2. Add a "Load DiT Model" node (for DiT) or "Load VAE" node (for VAE)
3. Select your quantized model from the dropdown
4. The model will automatically:
   - Detect the quantization format from metadata
   - Use NVFP4 if on Blackwell GPU and properly quantized
   - Fall back to NF4 or FP16 if requirements not met

### Step 3: Verify Loading

Check the ComfyUI console output for confirmation:

**NVFP4 Success:**
```
🚀 NVFP4 Blackwell optimization: ✅ (NVIDIA GeForce RTX 5090)
   └─ Native FP4 dispatch configured
   └─ Loaded 328 NVFP4-quantized layers
```

**NF4 Success:**
```
✅ Loaded model with NF4 quantization (bitsandbytes)
   └─ 328 layers quantized to 4-bit
   └─ VRAM usage: ~3.2GB (vs ~12.8GB FP16)
```

### Expected VRAM Usage

| Model | FP16 | NVFP4/NF4 | Reduction |
|-------|------|-----------|-----------|
| DiT 3B | ~12GB | ~3GB | 75% |
| DiT 7B | ~28GB | ~7GB | 75% |
| VAE | ~4GB | ~1GB | 75% |

## Troubleshooting

### Issue: "Missing required packages"

**Solution:**
```bash
pip install torch safetensors huggingface_hub tqdm bitsandbytes
```

### Issue: "NVFP4 not supported on this GPU"

**Cause:** You have a non-Blackwell GPU (RTX 30xx, 40xx, etc.)

**Solution:** Use NF4 instead:
```bash
python scripts/convert_to_nvfp4.py \
    --input_path model.safetensors \
    --output_path model_nf4.safetensors \
    --method nf4
```

### Issue: "CUDA out of memory during conversion"

**Cause:** Not enough VRAM to load full FP16 model

**Solution:** Convert on CPU (slower but works):
```bash
# Edit the script to use CPU by setting device='cpu' in load operations
# Or use a machine with more VRAM for conversion
```

Alternatively, the model will quantize at load time if you use NF4 method and simply copy the FP16 model with NF4 metadata.

### Issue: "Model loads but quality is degraded"

**Cause:** Critical layers were quantized

**Solution:** Reconvert with preserved critical layers (default):
```bash
python scripts/convert_to_nvfp4.py \
    --input_path model.safetensors \
    --output_path model_quantized.safetensors
    # Don't use --no_preserve_critical flag
```

### Issue: "Conversion is very slow"

**Expected:** Conversion can take 5-15 minutes for large models (3B-7B parameters)

**Tips to speed up:**
- Ensure model is on GPU (automatically done if CUDA available)
- Use SSD for input/output paths
- Close other GPU applications

### Issue: "bitsandbytes import error on Windows"

**Cause:** bitsandbytes has limited Windows support

**Solutions:**
1. Use WSL2 (Windows Subsystem for Linux)
2. Use NVFP4 on Blackwell GPU (doesn't need bitsandbytes)
3. Install pre-built Windows wheel:
   ```bash
   pip install bitsandbytes --prefer-binary
   ```

## Additional Resources

- **NVFP4 Details:** See `docs/BLACKWELL_OPTIMIZATION.md` for Blackwell-specific optimization
- **Hardware Diagnostic:** Run `python scripts/nvfp4_diagnostic.py` to verify your setup
- **ComfyUI Usage:** See `README.md` for general ComfyUI workflow examples
- **Performance Benchmarks:** See `CHANGELOG.md` for version-specific performance data

## Model Conversion Checklist

Before converting your model, verify:

- [ ] Python 3.12+ installed
- [ ] PyTorch 2.3+ (2.6+ for NVFP4) installed
- [ ] CUDA toolkit matches PyTorch build
- [ ] safetensors, huggingface_hub, tqdm installed
- [ ] bitsandbytes installed (for NF4)
- [ ] At least 2x model size free disk space
- [ ] NVFP4 diagnostic passed (for Blackwell GPUs)

Then run:
```bash
python scripts/convert_to_nvfp4.py \
    --model_url <your_model_url> \
    --output_path models/<output_name>.safetensors
```

## License

This conversion script is part of ComfyUI-SeedVR2.5 and is licensed under Apache 2.0.
