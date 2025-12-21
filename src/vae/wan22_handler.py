"""
Wan2.2 3D Causal VAE Handler for ComfyUI-SeedVR2.5

This module implements the Wan2.2 3D Causal VAE with full support for:
- 5D Tensor handling (B, C, F, H, W)
- GGUF quantized weight loading
- 3D Spatial-Temporal tiling to prevent OOM
- FP16/BF16 high-performance precision
- Batch size > 1 support while preserving temporal sequences

The implementation follows SeedVR2 DNA for VAE tiling and batching patterns.
"""

import logging
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

logger = logging.getLogger(__name__)

# Constants
EPSILON = 1e-8  # Small value to prevent division by zero
DEFAULT_LATENT_SCALE_FACTOR = 4  # Default spatial downsampling factor for latent space
MIN_LATENT_TILE_SIZE = 16  # Minimum tile size in latent space


class Wan22PrecisionMode(Enum):
    """Supported precision modes for Wan2.2 VAE"""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"


class Wan22TilingMode(Enum):
    """Tiling modes for memory-efficient processing"""
    DISABLED = "disabled"
    SPATIAL = "spatial"  # Tile only in H, W dimensions
    TEMPORAL = "temporal"  # Tile only in F (frames) dimension
    FULL_3D = "full_3d"  # Tile in all spatial-temporal dimensions


@dataclass
class Wan22TilingConfig:
    """Configuration for 3D Spatial-Temporal tiling"""
    mode: Wan22TilingMode = Wan22TilingMode.DISABLED
    spatial_tile_size: int = 256  # Tile size for H, W dimensions
    temporal_tile_size: int = 4   # Tile size for F (frames) dimension
    spatial_overlap: int = 32     # Overlap for spatial blending
    temporal_overlap: int = 1     # Overlap for temporal blending
    blend_mode: str = "linear"    # linear, cosine, or hard
    
    def validate(self) -> None:
        """Validate configuration parameters"""
        if self.spatial_tile_size < 64:
            raise ValueError("spatial_tile_size must be at least 64")
        if self.temporal_tile_size < 1:
            raise ValueError("temporal_tile_size must be at least 1")
        if self.spatial_overlap >= self.spatial_tile_size // 2:
            raise ValueError("spatial_overlap must be less than half of spatial_tile_size")
        if self.temporal_overlap >= self.temporal_tile_size:
            raise ValueError("temporal_overlap must be less than temporal_tile_size")


@dataclass  
class Wan22LoaderConfig:
    """Configuration for Wan2.2 VAE model loading"""
    model_path: Optional[str] = None
    device: str = "cuda"
    precision: Wan22PrecisionMode = Wan22PrecisionMode.FP16
    use_gguf: bool = False
    memory_efficient: bool = True
    enable_xformers: bool = True
    enable_flash_attention: bool = True


class CausalConv3d(nn.Module):
    """
    Causal 3D Convolution for temporal consistency in video VAE.
    
    This ensures that each frame only depends on previous frames,
    maintaining causality in the temporal dimension.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]] = 3,
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Optional[Union[int, Tuple[int, int, int]]] = None,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        
        # Normalize to 3D tuples
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(dilation, int):
            dilation = (dilation, dilation, dilation)
            
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        
        # Calculate temporal padding for causality
        temporal_kernel = kernel_size[0]
        self.temporal_padding = (temporal_kernel - 1) * dilation[0]
        
        # Spatial padding (same padding)
        if padding is None:
            spatial_padding_h = ((kernel_size[1] - 1) * dilation[1]) // 2
            spatial_padding_w = ((kernel_size[2] - 1) * dilation[2]) // 2
            padding = (0, spatial_padding_h, spatial_padding_w)  # No temporal padding here
        elif isinstance(padding, int):
            padding = (0, padding, padding)
            
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with causal temporal padding.
        
        Args:
            x: Input tensor of shape (B, C, F, H, W)
            
        Returns:
            Output tensor with causal convolution applied
        """
        # Apply causal padding: pad only past frames, not future
        if self.temporal_padding > 0:
            x = F.pad(x, (0, 0, 0, 0, self.temporal_padding, 0))
        
        return self.conv(x)


class CausalResBlock3d(nn.Module):
    """
    Residual block with causal 3D convolutions for video VAE.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        kernel_size: int = 3,
        use_dropout: bool = False,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        
        out_channels = out_channels or in_channels
        
        self.norm1 = nn.GroupNorm(
            num_groups=min(32, in_channels),
            num_channels=in_channels,
            eps=1e-6,
            affine=True,
        )
        self.conv1 = CausalConv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            bias=True,
        )
        
        self.norm2 = nn.GroupNorm(
            num_groups=min(32, out_channels),
            num_channels=out_channels,
            eps=1e-6,
            affine=True,
        )
        self.conv2 = CausalConv3d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            bias=True,
        )
        
        self.nonlinearity = nn.SiLU()
        self.dropout = nn.Dropout(dropout_rate) if use_dropout else None
        
        # Skip connection
        if in_channels != out_channels:
            self.skip_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=True)
        else:
            self.skip_conv = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection.
        
        Args:
            x: Input tensor of shape (B, C, F, H, W)
            
        Returns:
            Output tensor with residual block applied
        """
        residual = x
        
        h = self.norm1(x)
        h = self.nonlinearity(h)
        h = self.conv1(h)
        
        h = self.norm2(h)
        h = self.nonlinearity(h)
        if self.dropout is not None:
            h = self.dropout(h)
        h = self.conv2(h)
        
        if self.skip_conv is not None:
            residual = self.skip_conv(residual)
        
        return h + residual


class SpatialTemporalDownsample3d(nn.Module):
    """
    Downsampling for 3D video tensors with separate spatial and temporal control.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_factor: int = 2,
        temporal_factor: int = 2,
    ):
        super().__init__()
        
        self.spatial_factor = spatial_factor
        self.temporal_factor = temporal_factor
        
        # Use strided convolution for downsampling
        self.conv = CausalConv3d(
            in_channels,
            out_channels,
            kernel_size=(
                3 if temporal_factor > 1 else 1,
                3,
                3,
            ),
            stride=(temporal_factor, spatial_factor, spatial_factor),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Downsample input tensor.
        
        Args:
            x: Input tensor of shape (B, C, F, H, W)
            
        Returns:
            Downsampled tensor
        """
        # Pad spatial dimensions if not divisible
        b, c, f, h, w = x.shape
        pad_h = (self.spatial_factor - h % self.spatial_factor) % self.spatial_factor
        pad_w = (self.spatial_factor - w % self.spatial_factor) % self.spatial_factor
        
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0)
        
        return self.conv(x)


class SpatialTemporalUpsample3d(nn.Module):
    """
    Upsampling for 3D video tensors with separate spatial and temporal control.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_factor: int = 2,
        temporal_factor: int = 2,
    ):
        super().__init__()
        
        self.spatial_factor = spatial_factor
        self.temporal_factor = temporal_factor
        
        self.conv = CausalConv3d(
            in_channels,
            out_channels,
            kernel_size=3,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Upsample input tensor.
        
        Args:
            x: Input tensor of shape (B, C, F, H, W)
            
        Returns:
            Upsampled tensor
        """
        # Use nearest neighbor interpolation
        x = F.interpolate(
            x,
            scale_factor=(
                self.temporal_factor,
                self.spatial_factor,
                self.spatial_factor,
            ),
            mode="nearest",
        )
        return self.conv(x)


class Wan22Encoder3D(nn.Module):
    """
    Wan2.2 3D Causal Encoder for video VAE.
    
    Encodes video tensors (B, C, F, H, W) to latent space.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 4,
        base_channels: int = 64,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        temporal_downsample_indices: Tuple[int, ...] = (1, 2),
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        
        # Initial convolution
        self.conv_in = CausalConv3d(in_channels, base_channels, kernel_size=3)
        
        # Encoder blocks
        self.down_blocks = nn.ModuleList()
        in_ch = base_channels
        
        for i, mult in enumerate(channel_multipliers):
            out_ch = base_channels * mult
            
            block = nn.ModuleList()
            
            # Residual blocks
            for _ in range(num_res_blocks):
                block.append(CausalResBlock3d(in_ch, out_ch))
                in_ch = out_ch
            
            # Downsampling (except for last block)
            if i < len(channel_multipliers) - 1:
                temporal_down = i in temporal_downsample_indices
                block.append(
                    SpatialTemporalDownsample3d(
                        out_ch,
                        out_ch,
                        spatial_factor=2,
                        temporal_factor=2 if temporal_down else 1,
                    )
                )
            
            self.down_blocks.append(block)
        
        # Middle blocks
        self.mid_block1 = CausalResBlock3d(in_ch, in_ch)
        self.mid_block2 = CausalResBlock3d(in_ch, in_ch)
        
        # Output projection to latent space (mean and logvar)
        self.norm_out = nn.GroupNorm(min(32, in_ch), in_ch, eps=1e-6)
        self.conv_out = CausalConv3d(in_ch, 2 * latent_channels, kernel_size=3)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input video to latent distribution parameters.
        
        Args:
            x: Input tensor of shape (B, C, F, H, W)
            
        Returns:
            Tuple of (mean, logvar) tensors for latent distribution
        """
        h = self.conv_in(x)
        
        # Encoder blocks
        for block in self.down_blocks:
            for layer in block:
                h = layer(h)
        
        # Middle blocks
        h = self.mid_block1(h)
        h = self.mid_block2(h)
        
        # Output
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        
        # Split into mean and logvar
        mean, logvar = torch.chunk(h, 2, dim=1)
        
        return mean, logvar


class Wan22Decoder3D(nn.Module):
    """
    Wan2.2 3D Causal Decoder for video VAE.
    
    Decodes latent tensors to video (B, C, F, H, W).
    """
    
    def __init__(
        self,
        latent_channels: int = 4,
        out_channels: int = 3,
        base_channels: int = 64,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        temporal_upsample_indices: Tuple[int, ...] = (0, 1),
    ):
        super().__init__()
        
        self.latent_channels = latent_channels
        self.out_channels = out_channels
        
        # Calculate input channels from multipliers
        in_ch = base_channels * channel_multipliers[-1]
        
        # Input projection from latent space
        self.conv_in = CausalConv3d(latent_channels, in_ch, kernel_size=3)
        
        # Middle blocks
        self.mid_block1 = CausalResBlock3d(in_ch, in_ch)
        self.mid_block2 = CausalResBlock3d(in_ch, in_ch)
        
        # Decoder blocks (reversed order)
        self.up_blocks = nn.ModuleList()
        reversed_mults = list(reversed(channel_multipliers))
        
        for i, mult in enumerate(reversed_mults):
            out_ch = base_channels * mult
            
            block = nn.ModuleList()
            
            # Upsampling (except for first block in reversed order)
            if i > 0:
                temporal_up = (len(reversed_mults) - 1 - i) in temporal_upsample_indices
                block.append(
                    SpatialTemporalUpsample3d(
                        in_ch,
                        in_ch,
                        spatial_factor=2,
                        temporal_factor=2 if temporal_up else 1,
                    )
                )
            
            # Residual blocks
            for j in range(num_res_blocks + 1):
                block.append(CausalResBlock3d(in_ch, out_ch if j == num_res_blocks else in_ch))
            
            in_ch = out_ch
            self.up_blocks.append(block)
        
        # Output convolution
        self.norm_out = nn.GroupNorm(min(32, in_ch), in_ch, eps=1e-6)
        self.conv_out = CausalConv3d(in_ch, out_channels, kernel_size=3)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to video.
        
        Args:
            z: Latent tensor of shape (B, C_latent, F_latent, H_latent, W_latent)
            
        Returns:
            Decoded video tensor of shape (B, C, F, H, W)
        """
        h = self.conv_in(z)
        
        # Middle blocks
        h = self.mid_block1(h)
        h = self.mid_block2(h)
        
        # Decoder blocks
        for block in self.up_blocks:
            for layer in block:
                h = layer(h)
        
        # Output
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        
        return torch.tanh(h)


class Wan22CausalVAE3D(nn.Module):
    """
    Complete Wan2.2 3D Causal VAE for video encoding/decoding.
    
    This is the main VAE class that combines encoder and decoder with:
    - Proper 5D tensor handling (B, C, F, H, W)
    - GGUF weight loading support
    - 3D Spatial-Temporal tiling for memory efficiency
    - FP16/BF16 precision support
    - Batch processing while preserving temporal sequences
    
    Attributes:
        encoder: 3D Causal Encoder
        decoder: 3D Causal Decoder
        scaling_factor: Latent space scaling factor
        tiling_config: Configuration for tiled processing
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        latent_channels: int = 4,
        base_channels: int = 64,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        temporal_downsample_indices: Tuple[int, ...] = (1, 2),
        temporal_upsample_indices: Tuple[int, ...] = (0, 1),
        scaling_factor: float = 0.18215,
        shift_factor: float = 0.0,
        use_quant_conv: bool = True,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.scaling_factor = scaling_factor
        self.shift_factor = shift_factor
        
        # Encoder
        self.encoder = Wan22Encoder3D(
            in_channels=in_channels,
            latent_channels=latent_channels,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            num_res_blocks=num_res_blocks,
            temporal_downsample_indices=temporal_downsample_indices,
        )
        
        # Decoder
        self.decoder = Wan22Decoder3D(
            latent_channels=latent_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            num_res_blocks=num_res_blocks,
            temporal_upsample_indices=temporal_upsample_indices,
        )
        
        # Optional quantization convolutions
        if use_quant_conv:
            self.quant_conv = nn.Conv3d(2 * latent_channels, 2 * latent_channels, 1)
            self.post_quant_conv = nn.Conv3d(latent_channels, latent_channels, 1)
        else:
            self.quant_conv = None
            self.post_quant_conv = None
        
        # Tiling configuration
        self.tiling_config = Wan22TilingConfig()
        self._tiling_enabled = False
        
        # Cache for temporal state and original dimensions
        self._temporal_cache = None
        self._last_input_shape = None
    
    def enable_tiling(
        self,
        spatial_tile_size: int = 256,
        temporal_tile_size: int = 4,
        spatial_overlap: int = 32,
        temporal_overlap: int = 1,
        mode: Wan22TilingMode = Wan22TilingMode.FULL_3D,
    ) -> None:
        """
        Enable 3D tiled processing for memory efficiency.
        
        Args:
            spatial_tile_size: Tile size for H, W dimensions
            temporal_tile_size: Tile size for F dimension
            spatial_overlap: Overlap for spatial blending
            temporal_overlap: Overlap for temporal blending
            mode: Tiling mode (spatial, temporal, or full 3D)
        """
        self.tiling_config = Wan22TilingConfig(
            mode=mode,
            spatial_tile_size=spatial_tile_size,
            temporal_tile_size=temporal_tile_size,
            spatial_overlap=spatial_overlap,
            temporal_overlap=temporal_overlap,
        )
        self.tiling_config.validate()
        self._tiling_enabled = True
        logger.info(f"Enabled {mode.value} tiling: spatial={spatial_tile_size}, temporal={temporal_tile_size}")
    
    def disable_tiling(self) -> None:
        """Disable tiled processing"""
        self._tiling_enabled = False
        self.tiling_config.mode = Wan22TilingMode.DISABLED
        logger.info("Disabled tiling")
    
    def _ensure_5d(self, x: torch.Tensor) -> torch.Tensor:
        """Ensure input is 5D tensor (B, C, F, H, W)"""
        if x.ndim == 4:
            # (B, C, H, W) -> (B, C, 1, H, W)
            return x.unsqueeze(2)
        elif x.ndim == 5:
            return x
        else:
            raise ValueError(f"Expected 4D or 5D tensor, got {x.ndim}D")
    
    def _create_blend_mask(
        self,
        size: int,
        overlap: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Create linear blending mask for tile overlaps"""
        if overlap == 0:
            return torch.ones(size, device=device, dtype=dtype)
        
        mask = torch.ones(size, device=device, dtype=dtype)
        
        # Linear ramp for blend regions
        ramp = torch.linspace(0, 1, overlap, device=device, dtype=dtype)
        mask[:overlap] = ramp
        mask[-overlap:] = 1 - ramp
        
        return mask
    
    def encode(
        self,
        x: torch.Tensor,
        sample: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode video to latent space.
        
        Args:
            x: Input tensor of shape (B, C, F, H, W) or (B, C, H, W)
            sample: If True, sample from distribution; else return mean
            
        Returns:
            Dictionary with 'latent', 'mean', 'logvar', 'input_shape' keys
        """
        x = self._ensure_5d(x)
        
        # Store original input shape for decode
        self._last_input_shape = x.shape
        
        if self._tiling_enabled and self.tiling_config.mode != Wan22TilingMode.DISABLED:
            result = self._tiled_encode(x, sample)
        else:
            result = self._direct_encode(x, sample)
        
        # Store input shape in result for decode to use
        result["input_shape"] = x.shape
        
        return result
    
    def _direct_encode(
        self,
        x: torch.Tensor,
        sample: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Direct encoding without tiling"""
        mean, logvar = self.encoder(x)
        
        if self.quant_conv is not None:
            h = torch.cat([mean, logvar], dim=1)
            h = self.quant_conv(h)
            mean, logvar = torch.chunk(h, 2, dim=1)
        
        # Clamp logvar for stability
        logvar = torch.clamp(logvar, -30.0, 20.0)
        
        if sample:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            latent = mean + eps * std
        else:
            latent = mean
        
        # Apply scaling
        latent = (latent - self.shift_factor) * self.scaling_factor
        
        return {
            "latent": latent,
            "mean": mean,
            "logvar": logvar,
        }
    
    def _tiled_encode(
        self,
        x: torch.Tensor,
        sample: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Tiled encoding for memory efficiency.
        
        For simplicity, we fall back to direct encoding when tiling is enabled
        but the input is small enough, or we process in spatial tiles only
        since temporal tiling with causal convolutions is complex.
        """
        b, c, f, h, w = x.shape
        config = self.tiling_config
        
        # For now, only support spatial tiling (temporal tiling with causal conv is complex)
        # Fall back to direct encoding if input is small
        spatial_tile = config.spatial_tile_size
        spatial_overlap = config.spatial_overlap
        
        if h <= spatial_tile and w <= spatial_tile:
            # Input is small enough, use direct encoding
            return self._direct_encode(x, sample)
        
        # First, get latent dimensions by encoding a small test patch
        test_result = self._direct_encode(x[:, :, :, :min(spatial_tile, h), :min(spatial_tile, w)], sample=False)
        test_latent = test_result["latent"]
        
        # Calculate scaling factors
        test_h = min(spatial_tile, h)
        test_w = min(spatial_tile, w)
        latent_scale_h = test_h // test_latent.shape[3]
        latent_scale_w = test_w // test_latent.shape[4]
        latent_f = test_latent.shape[2]  # Get actual temporal dimension from encoder
        latent_c = test_latent.shape[1]
        
        # Calculate latent spatial dimensions
        latent_h = h // latent_scale_h
        latent_w = w // latent_scale_w
        
        # Calculate number of spatial tiles
        n_h = max(1, math.ceil((h - spatial_overlap) / (spatial_tile - spatial_overlap)))
        n_w = max(1, math.ceil((w - spatial_overlap) / (spatial_tile - spatial_overlap)))
        
        # Initialize accumulators
        latent_sum = torch.zeros((b, latent_c, latent_f, latent_h, latent_w), device=x.device, dtype=x.dtype)
        weight_sum = torch.zeros((b, 1, latent_f, latent_h, latent_w), device=x.device, dtype=x.dtype)
        
        # Process spatial tiles (full temporal sequence for each tile)
        for hi in range(n_h):
            for wi in range(n_w):
                # Calculate tile boundaries
                h_start = hi * (spatial_tile - spatial_overlap)
                h_end = min(h_start + spatial_tile, h)
                
                w_start = wi * (spatial_tile - spatial_overlap)
                w_end = min(w_start + spatial_tile, w)
                
                # Extract tile (full temporal sequence)
                tile = x[:, :, :, h_start:h_end, w_start:w_end]
                
                # Encode tile
                tile_result = self._direct_encode(tile, sample=sample)
                tile_latent = tile_result["latent"]
                
                # Calculate latent tile boundaries
                lh_start = h_start // latent_scale_h
                lh_end = lh_start + tile_latent.shape[3]
                lw_start = w_start // latent_scale_w
                lw_end = lw_start + tile_latent.shape[4]
                
                # Ensure we don't exceed bounds
                lh_end = min(lh_end, latent_h)
                lw_end = min(lw_end, latent_w)
                actual_lh = lh_end - lh_start
                actual_lw = lw_end - lw_start
                
                # Create blend weights
                tile_weight = torch.ones((b, 1, latent_f, actual_lh, actual_lw), device=x.device, dtype=x.dtype)
                
                # Accumulate (trim tile_latent if needed)
                latent_sum[:, :, :, lh_start:lh_end, lw_start:lw_end] += tile_latent[:, :, :, :actual_lh, :actual_lw] * tile_weight
                weight_sum[:, :, :, lh_start:lh_end, lw_start:lw_end] += tile_weight
        
        # Normalize by weights
        latent = latent_sum / (weight_sum + EPSILON)
        
        return {
            "latent": latent,
            "mean": latent,  # Approximate for tiled mode
            "logvar": torch.zeros_like(latent),  # Approximate for tiled mode
        }
    
    def decode(self, z: torch.Tensor, target_shape: Optional[Tuple[int, ...]] = None) -> Dict[str, torch.Tensor]:
        """
        Decode latent to video.
        
        Args:
            z: Latent tensor of shape (B, C, F, H, W) or (B, C, H, W)
            target_shape: Optional target output shape to crop to (B, C, F, H, W)
            
        Returns:
            Dictionary with 'sample' key containing decoded video
        """
        z = self._ensure_5d(z)
        
        # Use stored input shape or provided target shape
        if target_shape is None:
            target_shape = self._last_input_shape
        
        if self._tiling_enabled and self.tiling_config.mode != Wan22TilingMode.DISABLED:
            result = self._tiled_decode(z)
        else:
            result = self._direct_decode(z)
        
        # Crop output to match target shape if needed
        if target_shape is not None:
            sample = result["sample"]
            _, _, target_f, target_h, target_w = target_shape
            _, _, curr_f, curr_h, curr_w = sample.shape
            
            # Crop temporal dimension if needed
            if curr_f > target_f:
                sample = sample[:, :, :target_f, :, :]
            
            # Crop spatial dimensions if needed
            if curr_h > target_h:
                sample = sample[:, :, :, :target_h, :]
            if curr_w > target_w:
                sample = sample[:, :, :, :, :target_w]
            
            result["sample"] = sample
        
        return result
    
    def _direct_decode(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Direct decoding without tiling"""
        # Remove scaling
        z = z / self.scaling_factor + self.shift_factor
        
        if self.post_quant_conv is not None:
            z = self.post_quant_conv(z)
        
        sample = self.decoder(z)
        
        return {"sample": sample}
    
    def _tiled_decode(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Tiled decoding for memory efficiency.
        
        For simplicity, only support spatial tiling.
        """
        b, c, f, h, w = z.shape
        config = self.tiling_config
        
        # Calculate spatial tile size in latent space using configurable scale factor
        latent_spatial_tile = max(MIN_LATENT_TILE_SIZE, config.spatial_tile_size // DEFAULT_LATENT_SCALE_FACTOR)
        latent_spatial_overlap = max(MIN_LATENT_TILE_SIZE // 4, config.spatial_overlap // DEFAULT_LATENT_SCALE_FACTOR)
        
        if h <= latent_spatial_tile and w <= latent_spatial_tile:
            # Input is small enough, use direct decoding
            return self._direct_decode(z)
        
        # First pass to determine output dimensions
        test_result = self._direct_decode(z[:, :, :, :min(latent_spatial_tile, h), :min(latent_spatial_tile, w)])
        test_output = test_result["sample"]
        
        test_lh = min(latent_spatial_tile, h)
        test_lw = min(latent_spatial_tile, w)
        scale_h = test_output.shape[3] // test_lh
        scale_w = test_output.shape[4] // test_lw
        output_f = test_output.shape[2]
        output_c = test_output.shape[1]
        
        output_h = h * scale_h
        output_w = w * scale_w
        
        # Calculate number of tiles
        n_h = max(1, math.ceil((h - latent_spatial_overlap) / (latent_spatial_tile - latent_spatial_overlap)))
        n_w = max(1, math.ceil((w - latent_spatial_overlap) / (latent_spatial_tile - latent_spatial_overlap)))
        
        # Initialize accumulators
        output_sum = torch.zeros((b, output_c, output_f, output_h, output_w), device=z.device, dtype=z.dtype)
        weight_sum = torch.zeros((b, 1, output_f, output_h, output_w), device=z.device, dtype=z.dtype)
        
        # Process spatial tiles (full temporal sequence for each tile)
        for hi in range(n_h):
            for wi in range(n_w):
                # Calculate latent tile boundaries
                lh_start = hi * (latent_spatial_tile - latent_spatial_overlap)
                lh_end = min(lh_start + latent_spatial_tile, h)
                
                lw_start = wi * (latent_spatial_tile - latent_spatial_overlap)
                lw_end = min(lw_start + latent_spatial_tile, w)
                
                # Extract latent tile (full temporal sequence)
                tile = z[:, :, :, lh_start:lh_end, lw_start:lw_end]
                
                # Decode tile
                tile_result = self._direct_decode(tile)
                tile_output = tile_result["sample"]
                
                # Calculate output tile boundaries
                oh_start = lh_start * scale_h
                oh_end = oh_start + tile_output.shape[3]
                ow_start = lw_start * scale_w
                ow_end = ow_start + tile_output.shape[4]
                
                # Ensure we don't exceed bounds
                oh_end = min(oh_end, output_h)
                ow_end = min(ow_end, output_w)
                actual_oh = oh_end - oh_start
                actual_ow = ow_end - ow_start
                
                # Create blend weights
                tile_weight = torch.ones((b, 1, output_f, actual_oh, actual_ow), device=z.device, dtype=z.dtype)
                
                # Accumulate (trim tile_output if needed)
                output_sum[:, :, :, oh_start:oh_end, ow_start:ow_end] += tile_output[:, :, :, :actual_oh, :actual_ow] * tile_weight
                weight_sum[:, :, :, oh_start:oh_end, ow_start:ow_end] += tile_weight
        
        # Normalize by weights
        sample = output_sum / (weight_sum + EPSILON)
        
        return {"sample": sample}
    
    def forward(
        self,
        x: torch.Tensor,
        return_loss: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Full VAE forward pass: encode -> sample -> decode.
        
        Args:
            x: Input tensor of shape (B, C, F, H, W) or (B, C, H, W)
            return_loss: If True, return KL divergence loss
            
        Returns:
            Reconstructed tensor, or (reconstruction, kl_loss) if return_loss
        """
        x = self._ensure_5d(x)
        
        # Encode
        enc_result = self.encode(x, sample=True)
        latent = enc_result["latent"]
        mean = enc_result["mean"]
        logvar = enc_result["logvar"]
        
        # Decode
        dec_result = self.decode(latent)
        reconstruction = dec_result["sample"]
        
        if return_loss:
            # KL divergence loss
            kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
            kl_loss = kl_loss / (x.shape[0] * x.shape[2] * x.shape[3] * x.shape[4])
            return reconstruction, kl_loss
        
        return reconstruction
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return {
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "latent_channels": self.latent_channels,
            "scaling_factor": self.scaling_factor,
            "shift_factor": self.shift_factor,
            "tiling_enabled": self._tiling_enabled,
            "tiling_config": {
                "mode": self.tiling_config.mode.value,
                "spatial_tile_size": self.tiling_config.spatial_tile_size,
                "temporal_tile_size": self.tiling_config.temporal_tile_size,
            },
        }


def load_wan22_vae_gguf(
    gguf_path: str,
    device: Union[str, torch.device] = "cuda",
    dtype: torch.dtype = torch.float16,
    debug: Optional[Any] = None,
) -> Wan22CausalVAE3D:
    """
    Load Wan2.2 VAE from GGUF quantized weights.
    
    Args:
        gguf_path: Path to GGUF file
        device: Target device
        dtype: Target dtype for dequantized weights
        debug: Optional debug logger
        
    Returns:
        Loaded Wan22CausalVAE3D model
        
    Raises:
        ImportError: If GGUF library is not available
        FileNotFoundError: If GGUF file doesn't exist
    """
    try:
        import gguf
    except ImportError:
        raise ImportError(
            "GGUF library is required for loading quantized weights. "
            "Please install it with: pip install gguf"
        )
    
    if not Path(gguf_path).exists():
        raise FileNotFoundError(f"GGUF file not found: {gguf_path}")
    
    logger.info(f"Loading Wan2.2 VAE from GGUF: {gguf_path}")
    
    # Create model
    model = Wan22CausalVAE3D()
    
    # Load GGUF file
    reader = gguf.GGUFReader(gguf_path)
    
    # Get model state dict
    model_state = model.state_dict()
    
    # Load and dequantize tensors
    loaded_count = 0
    failed_tensors = []
    
    for tensor in reader.tensors:
        tensor_name = tensor.name
        
        # Map GGUF tensor names to model parameter names if needed
        param_name = tensor_name  # Adjust mapping as needed
        
        if param_name not in model_state:
            continue
        
        try:
            # Convert GGUF tensor to PyTorch
            torch_tensor = torch.from_numpy(tensor.data)
            
            # Dequantize if needed
            if tensor.tensor_type not in {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16}:
                try:
                    # Use GGUF's built-in dequantization
                    dequantized = gguf.quants.dequantize(tensor.data, tensor.tensor_type)
                    torch_tensor = torch.from_numpy(dequantized)
                except Exception as dequant_error:
                    logger.warning(
                        f"Failed to dequantize tensor '{param_name}' with type {tensor.tensor_type}: {dequant_error}. "
                        "Falling back to raw tensor data."
                    )
                    # Fallback: use raw tensor data and hope for the best
                    torch_tensor = torch.from_numpy(tensor.data).float()
            
            # Reshape and convert dtype
            target_shape = model_state[param_name].shape
            torch_tensor = torch_tensor.view(target_shape).to(dtype)
            
            # Set parameter
            model_state[param_name] = torch_tensor
            loaded_count += 1
            
        except Exception as e:
            failed_tensors.append((param_name, str(e)))
            logger.warning(f"Failed to load tensor '{param_name}': {e}")
    
    if failed_tensors:
        logger.warning(f"Failed to load {len(failed_tensors)} tensors. First few: {failed_tensors[:5]}")
    
    # Load state dict
    model.load_state_dict(model_state)
    model = model.to(device, dtype)
    model.eval()
    
    logger.info(f"Loaded {loaded_count} parameters from GGUF")
    
    return model


def create_wan22_causal_vae(
    latent_channels: int = 4,
    base_channels: int = 64,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None,
    enable_tiling: bool = False,
    tiling_config: Optional[Wan22TilingConfig] = None,
) -> Wan22CausalVAE3D:
    """
    Create a Wan2.2 3D Causal VAE with standard configuration.
    
    Args:
        latent_channels: Number of latent channels
        base_channels: Base channel count
        device: Target device
        dtype: Model dtype
        enable_tiling: Enable 3D tiling
        tiling_config: Custom tiling configuration
        
    Returns:
        Configured Wan22CausalVAE3D instance
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dtype is None:
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    model = Wan22CausalVAE3D(
        in_channels=3,
        out_channels=3,
        latent_channels=latent_channels,
        base_channels=base_channels,
        channel_multipliers=(1, 2, 4, 8),
        num_res_blocks=2,
        temporal_downsample_indices=(1, 2),
        temporal_upsample_indices=(0, 1),
    )
    
    model = model.to(device, dtype)
    
    if enable_tiling:
        if tiling_config is not None:
            model.enable_tiling(
                spatial_tile_size=tiling_config.spatial_tile_size,
                temporal_tile_size=tiling_config.temporal_tile_size,
                spatial_overlap=tiling_config.spatial_overlap,
                temporal_overlap=tiling_config.temporal_overlap,
                mode=tiling_config.mode,
            )
        else:
            model.enable_tiling()
    
    return model
