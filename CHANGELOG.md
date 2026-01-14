# Changelog

All notable changes to SeedVR2 VideoUpscaler will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [2.6.0] - SpargeAttn/Sage2 Blackwell Integration

### Summary

This release integrates the SpargeAttn library (SageAttention) with Sage2 implementation optimized for NVIDIA RTX 50-series (Blackwell) GPUs. The integration provides significant performance improvements through specialized Triton kernels and FP4/FP8 acceleration.

### Added

#### SageAttention Integration
- **SageAttention 2 support** (`sageattn_2`): Variable-length attention optimized for NVIDIA GPUs via the `sageattention` package
- **SageAttention 3 / Blackwell support** (`sageattn_3`): Maximum performance on RTX 50xx series GPUs via the `sageattn3` package
- **Automatic fallback chain**: SA3 → SA2 → SDPA when required packages are unavailable
- **Hybrid GPU detection**: Automatic detection of Blackwell architecture (compute capability 10.0+)

#### Blackwell-Specific Optimizations
- **NVFP4 quantization**: Native 4-bit floating point for Blackwell Tensor Cores (E2M1 format)
- **Block-wise scaling**: E4M3 scale factors per 16-weight block for accuracy preservation
- **Preserved layers**: Bias, Norm, Embeddings remain in FP16 for quality
- **Native FP4 dispatch**: TF32 and cuDNN benchmark configuration for optimal kernel selection

#### Async Offloading Infrastructure  
- **PinnedMemoryPool**: Reusable pinned memory buffer pool with LRU eviction
- **CUDAStreamManager**: Dedicated streams for H2D/D2H/Compute operations
- **AsyncModelOffloader**: Layer-by-layer prefetching for BlockSwap-style loading
- **Overlapped transfers**: Data movement concurrent with computation

#### Benchmark & Validation Scripts
- **Attention parity test** (`scripts/attention_parity_test.py`): Validates output consistency across attention backends
- **Performance benchmark** (`scripts/attention_benchmark.py`): Measures VRAM, throughput, and latency
- **NVFP4 diagnostic** (`scripts/nvfp4_diagnostic.py`): Pre-flight system verification

### Changed

#### Attention Backend Selection
- DiT model loader now includes `attention_mode` parameter with options:
  - `sdpa`: PyTorch scaled_dot_product_attention (default, stable)
  - `flash_attn_2`: Flash Attention 2 (Ampere+)
  - `flash_attn_3`: Flash Attention 3 (Hopper+)
  - `sageattn_2`: SageAttention 2
  - `sageattn_3`: SageAttention 3 (Blackwell/RTX 50xx)

#### Compatibility Layer (`src/optimization/compatibility.py`)
- Added SageAttention import with graceful fallback
- Added `call_sage_attn_2_varlen()` wrapper for SA2
- Added `call_sage_attn_3_varlen()` wrapper for SA3 with varlen-to-batched conversion
- Enhanced `validate_attention_mode()` with SA3 → SA2 → SDPA fallback chain

### Performance Expectations

#### RTX 5090 (32GB VRAM)
- DiT 7B: ~2-3x faster than FP16 with NVFP4
- Full video upscaling: ~40-50% faster end-to-end
- SageAttention 3: Additional 10-20% attention speedup

#### RTX 5080/5070 Ti (16GB VRAM)
- DiT 3B: Optimal performance with NVFP4
- DiT 7B: May require BlockSwap
- Async offloading essential for large models

### Requirements

#### Hardware
- **NVFP4/SA3**: NVIDIA RTX 50-series (Blackwell, compute capability 10.0+)
- **SA2**: Any modern NVIDIA GPU
- **SDPA**: All platforms (CPU, CUDA, MPS)

#### Software
- **NVFP4**: PyTorch 2.6+ with CUDA 12.8+
- **SA3**: `pip install sageattn3` or `sageattention>=2.0` with Blackwell support
- **SA2**: `pip install sageattention`
- **Flash Attention**: `pip install flash-attn`

### Files Modified

#### Core Attention
- `src/models/dit_3b/attention.py`: FlashAttentionVarlen with SA2/SA3 support
- `src/models/dit_7b/attention.py`: FlashAttentionVarlen with SA2/SA3 support

#### Optimization
- `src/optimization/compatibility.py`: SageAttention imports and wrappers
- `src/optimization/nvfp4.py`: NVFP4 quantization and async offloading

#### Interfaces
- `src/interfaces/dit_model_loader.py`: attention_mode parameter, NVFP4 options

#### Documentation
- `docs/BLACKWELL_OPTIMIZATION.md`: Blackwell optimization guide
- `CHANGELOG.md`: This file

#### Scripts
- `scripts/nvfp4_diagnostic.py`: System verification
- `scripts/attention_parity_test.py`: Output validation (new)
- `scripts/attention_benchmark.py`: Performance benchmarking (new)

### Known Issues & Patches

Based on SageAttention GitHub issues:

1. **Variable-length sequences with SA3**: SA3/Blackwell uses batched attention, not varlen. The wrapper automatically detects non-uniform sequence lengths and falls back to SA2.

2. **FP32 input handling**: SageAttention requires FP16/BF16 inputs. The wrappers automatically convert FP32 inputs to BF16 and convert outputs back.

3. **Softmax scale**: The wrappers compute and pass the correct softmax scale (1/sqrt(head_dim)) to prevent numerical issues.

### Migration Guide

#### From Previous Versions
1. **No code changes required**: Default attention mode is `sdpa` (unchanged behavior)
2. **To enable SageAttention**: Set `attention_mode="sageattn_2"` or `"sageattn_3"` in DiT model loader
3. **Install packages**: `pip install sageattention` (SA2) or `pip install sageattn3` (SA3)

#### For Blackwell Users
1. Run `python scripts/nvfp4_diagnostic.py` to verify system configuration
2. Set `enable_nvfp4=True` (default) in DiT model loader
3. Set `attention_mode="sageattn_3"` for maximum attention performance
4. The system will automatically use optimal settings

### References

- [SageAttention GitHub](https://github.com/thu-ml/SageAttention)
- [NVIDIA Blackwell Architecture](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)
- [PyTorch FP8 Documentation](https://pytorch.org/docs/stable/generated/torch.float8_e4m3fn.html)
- [CUDA 12.8 Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/)

---

## [2.5.24] - 2025-12-24

### Fixed
- **MPS memory leak regression**: Restored MPS cache clearing after VAE encode/decode operations

## [2.5.23] - 2025-12-24

### Fixed
- **Security: Prevent code execution in model loading**: Added protection against malicious .pth files
- **FFmpeg video writer reliability**: Resolved ffmpeg process hanging issues
- **GGUF VAE model support**: Enabled automatic weight dequantization for convolution operations
- **VAE slicing edge cases**: Protected against division by zero crashes
- **LAB color transfer precision**: Resolved dtype mismatch errors
- **PyTorch 2.9+ compatibility**: Extended Conv3d memory workaround
- **Bitsandbytes compatibility**: Added ValueError exception handling

## [2.5.22] - 2025-12-13

### Added
- **CLI: FFmpeg video backend with 10-bit support**: New `--video_backend ffmpeg` and `--10bit` flags

### Fixed
- **MPS bicubic upscaling compatibility**: Added CPU fallback for bicubic+antialias
- **Cross-platform histogram matching**: Replaced scatter_ with argsort+index_select

## [2.5.21] - 2025-12-12

### Fixed
- **GGUF dequantization error on MPS**: Resolved shape mismatch error
- **MPS sync overhead**: Skip unnecessary CPU tensor offload on unified memory

### Added
- **MPS: Preload text embeddings**: Load before Phase 1 to avoid sync stalls

## [2.5.20] - 2025-12-12

### Added
- **Expanded attention backends**: Full support for Flash Attention 2/3, SageAttention 2/3
- **macOS/Apple Silicon compatibility**: Replaced MPS autocast with explicit dtype conversion
- **Flash Attention graceful fallback**: Compatibility shims for corrupted DLLs
- **AMD ROCm: bitsandbytes conflict fix**: Prevent kernel registration errors

---

*For older versions, see the [Release Notes](README.md#-release-notes) in the README.*
