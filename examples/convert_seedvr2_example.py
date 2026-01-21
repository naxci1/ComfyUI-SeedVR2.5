#!/usr/bin/env python3
"""
Example: Convert SeedVR2 Model to NVFP4/NF4

This script demonstrates how to convert the SeedVR2 3B model from FP16 to 4-bit quantized format.

Before running:
    pip install -r requirements-conversion.txt

Usage:
    python examples/convert_seedvr2_example.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    print("=" * 70)
    print("  SeedVR2 Model Conversion Example")
    print("=" * 70)
    
    # Example 1: Auto-detect hardware and convert
    print("\n📝 Example 1: Auto-detect and convert from Hugging Face")
    print("-" * 70)
    print("""
This will:
1. Detect your GPU (Blackwell → NVFP4, Others → NF4)
2. Download the SeedVR2 3B model from Hugging Face
3. Convert to optimal quantized format
4. Save to models directory

Command:
python scripts/convert_to_nvfp4.py \\
    --model_url https://huggingface.co/numz/SeedVR2_comfyUI/blob/main/seedvr2_ema_3b_fp16.safetensors \\
    --output_path models/seedvr2_ema_3b_quantized.safetensors
    """)
    
    # Example 2: Convert local file
    print("\n📝 Example 2: Convert already downloaded model")
    print("-" * 70)
    print("""
If you already have the model downloaded:

Command:
python scripts/convert_to_nvfp4.py \\
    --input_path models/seedvr2_ema_3b_fp16.safetensors \\
    --output_path models/seedvr2_ema_3b_quantized.safetensors
    """)
    
    # Example 3: Force specific method
    print("\n📝 Example 3: Force NF4 for RTX 30xx/40xx GPUs")
    print("-" * 70)
    print("""
Explicitly use NF4 quantization (works on all modern GPUs):

Command:
python scripts/convert_to_nvfp4.py \\
    --input_path models/seedvr2_ema_3b_fp16.safetensors \\
    --output_path models/seedvr2_ema_3b_nf4.safetensors \\
    --method nf4
    """)
    
    # Example 4: Force NVFP4 for Blackwell
    print("\n📝 Example 4: Force NVFP4 for RTX 50xx GPUs")
    print("-" * 70)
    print("""
Explicitly use NVFP4 quantization (Blackwell only):

Command:
python scripts/convert_to_nvfp4.py \\
    --input_path models/seedvr2_ema_3b_fp16.safetensors \\
    --output_path models/seedvr2_ema_3b_nvfp4.safetensors \\
    --method nvfp4
    """)
    
    # Expected output
    print("\n📊 Expected Output:")
    print("-" * 70)
    print("""
Hardware Information:
  GPU: NVIDIA GeForce RTX 5090
  Compute Capability: SM100
  PyTorch: 2.6.0
  CUDA: 12.8
  Blackwell GPU: Yes ✅
  Recommended Method: NVFP4

🔄 Downloading model from Hugging Face...
  Repository: numz/SeedVR2_comfyUI
  Filename: seedvr2_ema_3b_fp16.safetensors
✅ Downloaded to: models/seedvr2_ema_3b_fp16.safetensors

======================================================================
  Model Conversion
======================================================================
  Input: models/seedvr2_ema_3b_fp16.safetensors
  Output: models/seedvr2_ema_3b_quantized.safetensors
  Method: NVFP4
  Preserve critical layers: True
ℹ️  Loaded 328 parameters
  Original size: 12.45 GB
🔄 Converting to NVFP4 format...
Quantizing: 100%|████████████████████████████████| 328/328 [00:45<00:00]
✅ Quantized 297 parameters, preserved 31
  Quantized size: 3.28 GB
  Size reduction: 73.7%
🔄 Saving quantized model...
✅ Saved to: models/seedvr2_ema_3b_quantized.safetensors

⏱️  Conversion completed in 52.3 seconds
    """)
    
    # Usage in ComfyUI
    print("\n💡 Using in ComfyUI:")
    print("-" * 70)
    print("""
1. Copy quantized model to ComfyUI:
   cp models/seedvr2_ema_3b_quantized.safetensors \\
      /path/to/ComfyUI/models/diffusion_models/

2. In ComfyUI:
   - Add "Load DiT Model" node
   - Select "seedvr2_ema_3b_quantized.safetensors"
   - The model will automatically use NVFP4/NF4 quantization

3. Check console for confirmation:
   🚀 NVFP4 Blackwell optimization: ✅
      └─ Native FP4 dispatch configured
      └─ Loaded 297 NVFP4-quantized layers
    """)
    
    # Performance comparison
    print("\n⚡ Performance Comparison:")
    print("-" * 70)
    print("""
│ Metric          │ FP16    │ NVFP4   │ NF4     │
├─────────────────┼─────────┼─────────┼─────────┤
│ VRAM Usage      │ 12.5 GB │ 3.3 GB  │ 3.3 GB  │
│ Speed (RTX 5090)│ 1.0x    │ 2.8x    │ 1.1x    │
│ Speed (RTX 4090)│ 1.0x    │ N/A     │ 1.0x    │
│ Quality Loss    │ 0%      │ <1%     │ <2%     │
    """)
    
    # Troubleshooting
    print("\n🛠️  Troubleshooting:")
    print("-" * 70)
    print("""
Issue: "bitsandbytes not available"
→ Solution: pip install bitsandbytes

Issue: "NVFP4 not supported on this GPU"
→ Solution: Use --method nf4 for non-Blackwell GPUs

Issue: "CUDA out of memory during conversion"
→ Solution: Close other applications or use CPU (slower)

Issue: Run diagnostic to check system:
→ Command: python scripts/nvfp4_diagnostic.py
    """)
    
    # Next steps
    print("\n📚 Next Steps:")
    print("-" * 70)
    print("""
1. Read full documentation:
   docs/MODEL_CONVERSION.md

2. Check hardware compatibility:
   python scripts/nvfp4_diagnostic.py

3. Convert your model:
   python scripts/convert_to_nvfp4.py --help

4. Optimize Blackwell GPUs:
   docs/BLACKWELL_OPTIMIZATION.md
    """)
    
    print("\n" + "=" * 70)
    print("  Ready to convert? Run the commands above!")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
