# WAN2.1 3D Causal VAE Integration

## Overview

The WAN2.1 3D Causal VAE integration adds support for WAN2.1's advanced temporal video encoding architecture to SeedVR2. This integration maintains full backward compatibility with existing VAE models while adding powerful new capabilities.

## Key Features

### 1. Temporal Padding (T = 4n + 1 Constraint)
WAN2.1 requires the number of frames to satisfy the formula **T = 4n + 1** (e.g., 1, 5, 9, 13, 17, 21, ...).

- **Automatic Padding**: The wrapper automatically pads input videos to satisfy this constraint
- **Reflection Padding**: Uses reflection padding to maintain temporal continuity
- **Transparent Unpadding**: Automatically removes padding after encoding/decoding

### 2. 3D Temporal Tiling
Process long videos efficiently by splitting them into temporal chunks with overlap:

- **Default Chunk Size**: 17 frames (satisfies 4n + 1 where n = 4)
- **Configurable Overlap**: Default 4 frames to prevent temporal seams
- **Linear Blending**: Smooth transitions between chunks using weighted blending
- **Memory Efficient**: Reduces VRAM usage for long video sequences

### 3. Tensor Format Conversion
Automatic conversion between SeedVR2 and WAN2.1 formats:

- **SeedVR2 Format**: `[B, T, H, W, C]` (batch, time, height, width, channels)
- **WAN2.1 Format**: `[B, C, T, H, W]` (batch, channels, time, height, width)
- **Bidirectional**: Handles conversion in both directions automatically

### 4. Architecture Specifications

- **Temporal Compression**: 4x (4 input frames → 1 latent frame)
- **Spatial Compression**: 8x (standard VAE spatial downsampling)
- **Causal Convolution**: Temporal convolutions only look backward or at current frame
- **Latent Channels**: 16 channels (vs. 4 for standard VAEs)
- **Scaling Factor**: 0.18215 (WAN2.1-specific, extracted from config)

## Usage

### Automatic Detection
The system automatically detects WAN2.1 models by examining the state dictionary:

```python
# Detection looks for:
# - 3D convolution layers (conv3d)
# - Temporal/causal markers in keys
# - WAN2.1-specific patterns (v_blocks, temporal_conv, inflation)
```

### Manual Wrapping
You can also manually wrap a VAE model:

```python
from src.vae.wan2_1_wrapper import Wan21VAEWrapper, wrap_vae_if_wan21

# Automatic wrapping based on detection
wrapped_vae = wrap_vae_if_wan21(base_vae, state_dict=state_dict)

# Manual wrapping
wrapped_vae = Wan21VAEWrapper(
    base_vae=base_vae,
    scaling_factor=0.18215,
    shift_factor=0.0,
    temporal_compression=4,
    spatial_compression=8,
)
```

### Encoding with Tiling

```python
# Encode with 3D temporal tiling enabled
latent = vae.encode(
    video_tensor,
    tiled=True,
    tile_size=(512, 512),          # Spatial tile size
    tile_overlap=(64, 64),         # Spatial overlap
    # Temporal tiling parameters handled automatically:
    # - temporal_tile_size=17 frames (4*4 + 1)
    # - temporal_overlap=4 frames
)
```

### Decoding with Tiling

```python
# Decode with 3D temporal tiling enabled
video = vae.decode(
    latent_tensor,
    tiled=True,
    tile_size=(512, 512),          # Spatial tile size
    tile_overlap=(64, 64),         # Spatial overlap
    # Temporal tiling parameters handled automatically:
    # - temporal_tile_size=5 latent frames (expands to 17 pixel frames)
    # - temporal_overlap=1 latent frame
)
```

## Configuration

### VAE Configuration Manager

```python
from src.vae.vae_config import get_wan21_config, get_vae_config_manager

# Get default WAN2.1 configuration
config = get_wan21_config()

# Customize configuration
config_manager = get_vae_config_manager()
config_manager.update_config("wan2.1", {
    "architecture": {
        "scaling_factor": 0.18215,
        "latent_channels": 16,
    },
    "encoding": {
        "precision": "bf16",
        "use_tiling": True,
    },
    "custom_params": {
        "temporal_compression": 4,
        "spatial_compression": 8,
    }
})
```

## Technical Implementation

### Temporal Padding Algorithm

```python
def calculate_valid_temporal_size(num_frames: int) -> int:
    """Calculate nearest valid size satisfying T = 4n + 1"""
    if num_frames <= 1:
        return 1
    n = math.ceil((num_frames - 1) / 4.0)
    return 4 * n + 1

# Example:
# Input: 15 frames → Output: 17 frames (n=4, 4*4+1=17)
# Input: 18 frames → Output: 21 frames (n=5, 4*5+1=21)
```

### Temporal Chunk Blending

The blending algorithm uses linear interpolation in overlap regions:

1. **Fade In**: At the start of each chunk (except first), weights linearly increase from 0 to 1
2. **Fade Out**: At the end of each chunk (except last), weights linearly decrease from 1 to 0
3. **Normalization**: Final output is divided by sum of weights for proper blending

This prevents visible seams or discontinuities at chunk boundaries.

## Backward Compatibility

The integration maintains full backward compatibility:

- ✅ **Legacy VAE Models**: Continue to work without modification
- ✅ **Existing Workflows**: No changes required to existing workflows
- ✅ **API Compatibility**: Same encode/decode interface as standard VAEs
- ✅ **Automatic Fallback**: Non-WAN2.1 models bypass the wrapper

## Performance Considerations

### Memory Usage
- **With Tiling**: Significantly reduced VRAM usage for long videos
- **Without Tiling**: Similar to standard VAE (with temporal padding overhead)
- **Recommended**: Enable tiling for videos longer than 17 frames

### Precision
- **Recommended**: BFloat16 for optimal performance on modern GPUs (RTX 30xx+)
- **Fallback**: Float16 for older GPUs
- **Compatibility**: Float32 supported but not recommended (higher memory usage)

### Temporal Overhead
- **Padding**: Minimal overhead (~1-5 additional frames maximum)
- **Blending**: Linear cost with number of chunks
- **Overall Impact**: Negligible for most use cases

## Known Limitations

1. **Spatial Tiling**: Full spatial tiling for 3D tensors not yet implemented
   - Currently falls back to full spatial encoding/decoding
   - Temporal tiling works as expected

2. **From Pretrained**: Loading WAN2.1 models from scratch requires the actual WAN2.1 architecture
   - Current implementation wraps existing loaded models
   - Future: Direct WAN2.1 model construction

3. **State Dict Detection**: Detection relies on key patterns
   - May need adjustment for different WAN2.1 variants
   - Manual wrapping available as fallback

## Future Enhancements

- [ ] Full spatial tiling implementation for 3D tensors
- [ ] WAN2.1 model construction from config
- [ ] Adaptive temporal chunk sizing based on VRAM
- [ ] Multi-resolution temporal tiling
- [ ] Optimized blending with GPU kernels

## References

- **WAN2.1 Architecture**: 3D Causal VAE with 4x temporal and 8x spatial compression
- **Temporal Constraint**: T = 4n + 1 frame requirement
- **Causal Convolution**: Forward-only temporal dependencies

## Support

For issues or questions:
1. Check existing GitHub issues
2. Review this documentation
3. Examine the wrapper code in `src/vae/wan2_1_wrapper.py`
4. Create a new issue with detailed information
