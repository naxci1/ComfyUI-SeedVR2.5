"""
WAN2.1 3D Causal VAE Wrapper for SeedVR2 Integration

This module provides a wrapper that integrates WAN2.1's 3D Causal VAE architecture
with SeedVR2's existing VAE infrastructure. The wrapper handles:
- Temporal padding to satisfy T = 4n + 1 constraint
- Tensor format conversions between SeedVR2 and WAN2.1
- 3D temporal tiling with overlap for memory efficiency
- Proper scaling factor management

Key Features:
- Temporal padding/unpadding (T = 4n + 1 requirement)
- Tensor permutations: [B, T, H, W, C] <-> [B, C, T, H, W]
- 3D temporal tiling with configurable overlap
- BFloat16/Float16 precision support
- Backward compatible with legacy VAE loading

Architecture:
- Temporal compression: 4x
- Spatial compression: 8x
- Causal temporal convolution (looks back only)
- Scaling factor extracted from config
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, Dict, Any
import logging
import math

logger = logging.getLogger(__name__)


def calculate_valid_temporal_size(num_frames: int) -> int:
    """
    Calculate the nearest valid temporal size for WAN2.1 (T = 4n + 1).
    
    Args:
        num_frames: Input number of frames
        
    Returns:
        Nearest valid frame count >= num_frames that satisfies T = 4n + 1
    """
    if num_frames <= 1:
        return 1
    
    # Find n such that 4n + 1 >= num_frames
    # Solving: 4n + 1 >= num_frames => n >= (num_frames - 1) / 4
    n = math.ceil((num_frames - 1) / 4.0)
    return 4 * n + 1


def pad_temporal_to_valid(x: torch.Tensor, dim: int = 2) -> Tuple[torch.Tensor, int]:
    """
    Pad tensor temporally to satisfy T = 4n + 1 constraint.
    Uses reflection padding to maintain temporal continuity.
    
    Args:
        x: Input tensor with temporal dimension at position `dim`
        dim: Dimension index for temporal axis (default: 2 for [B, C, T, H, W])
        
    Returns:
        Tuple of (padded_tensor, original_temporal_size)
    """
    original_size = x.shape[dim]
    valid_size = calculate_valid_temporal_size(original_size)
    
    if original_size == valid_size:
        return x, original_size
    
    pad_amount = valid_size - original_size
    
    # Create padding specification for F.pad
    # F.pad expects padding in reverse order: (pad_last_dim_left, pad_last_dim_right, ...)
    ndim = x.ndim
    pad_spec = [0, 0] * ndim
    
    # Set padding for the temporal dimension
    # We pad at the end to maintain causality
    pad_idx = (ndim - dim - 1) * 2  # Convert dim index to pad_spec index
    pad_spec[pad_idx + 1] = pad_amount  # Pad right side
    
    # Use reflection padding if possible, otherwise replicate
    try:
        x_padded = F.pad(x, pad_spec, mode='reflect')
    except RuntimeError:
        # Reflection might fail if temporal size is too small
        x_padded = F.pad(x, pad_spec, mode='replicate')
    
    return x_padded, original_size


def unpad_temporal(x: torch.Tensor, original_size: int, dim: int = 2) -> torch.Tensor:
    """
    Remove temporal padding to restore original size.
    
    Args:
        x: Padded tensor
        original_size: Original temporal size before padding
        dim: Dimension index for temporal axis
        
    Returns:
        Unpadded tensor with original temporal size
    """
    current_size = x.shape[dim]
    if current_size == original_size:
        return x
    
    # Create slicing to remove padding
    slices = [slice(None)] * x.ndim
    slices[dim] = slice(0, original_size)
    
    return x[tuple(slices)]


class Wan21VAEWrapper(nn.Module):
    """
    Wrapper for WAN2.1 3D Causal VAE that integrates with SeedVR2 infrastructure.
    
    This wrapper:
    1. Handles temporal padding (T = 4n + 1 constraint)
    2. Manages tensor format conversions
    3. Implements 3D temporal tiling
    4. Provides SeedVR2-compatible interface
    
    Attributes:
        base_vae: Underlying VAE model (can be WAN2.1 or compatible architecture)
        temporal_compression: Temporal compression factor (default: 4)
        spatial_compression: Spatial compression factor (default: 8)
        scaling_factor: Latent scaling factor from config
        shift_factor: Optional latent shift factor
    """
    
    def __init__(
        self,
        base_vae: nn.Module,
        scaling_factor: float = 0.18215,
        shift_factor: float = 0.0,
        temporal_compression: int = 4,
        spatial_compression: int = 8,
    ):
        """
        Initialize WAN2.1 VAE wrapper.
        
        Args:
            base_vae: Base VAE model to wrap
            scaling_factor: Scaling factor for latent space (from config.json)
            shift_factor: Shift factor for latent space (optional)
            temporal_compression: Temporal compression ratio (default: 4x)
            spatial_compression: Spatial compression ratio (default: 8x)
        """
        super().__init__()
        
        self.base_vae = base_vae
        self.scaling_factor = scaling_factor
        self.shift_factor = shift_factor
        self.temporal_compression = temporal_compression
        self.spatial_compression = spatial_compression
        
        # Check if base VAE has encoder/decoder attributes
        if not hasattr(base_vae, 'encoder') or not hasattr(base_vae, 'decoder'):
            raise ValueError("Base VAE must have 'encoder' and 'decoder' attributes")
        
        logger.info(
            f"Initialized WAN2.1 VAE Wrapper: "
            f"temporal_compression={temporal_compression}x, "
            f"spatial_compression={spatial_compression}x, "
            f"scaling_factor={scaling_factor}"
        )
    
    @property
    def device(self) -> torch.device:
        """Get device of the underlying VAE."""
        return next(self.base_vae.parameters()).device
    
    @property
    def dtype(self) -> torch.dtype:
        """Get dtype of the underlying VAE."""
        return next(self.base_vae.parameters()).dtype
    
    def to(self, *args, **kwargs):
        """Override to() to move base VAE."""
        self.base_vae = self.base_vae.to(*args, **kwargs)
        return self
    
    def permute_to_wan21_format(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert from SeedVR2 format [B, T, H, W, C] to WAN2.1 format [B, C, T, H, W].
        
        Args:
            x: Input tensor in SeedVR2 format
            
        Returns:
            Tensor in WAN2.1 format
        """
        # Check if already in correct format
        if x.ndim == 5:
            # Assume [B, T, H, W, C] if last dim is small (channels)
            # or [B, C, T, H, W] if already correct
            if x.shape[1] < x.shape[-1]:  # Likely [B, T, H, W, C]
                return x.permute(0, 4, 1, 2, 3)  # -> [B, C, T, H, W]
            else:  # Already [B, C, T, H, W]
                return x
        elif x.ndim == 4:
            # Single frame: [B, H, W, C] -> [B, C, 1, H, W]
            if x.shape[1] > x.shape[-1]:  # [B, H, W, C]
                x = x.permute(0, 3, 1, 2)  # -> [B, C, H, W]
            return x.unsqueeze(2)  # -> [B, C, 1, H, W]
        else:
            raise ValueError(f"Unexpected tensor shape: {x.shape}")
    
    def permute_from_wan21_format(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert from WAN2.1 format [B, C, T, H, W] to SeedVR2 format [B, T, H, W, C].
        
        Args:
            x: Input tensor in WAN2.1 format
            
        Returns:
            Tensor in SeedVR2 format
        """
        if x.ndim == 5:
            # [B, C, T, H, W] -> [B, T, H, W, C]
            return x.permute(0, 2, 3, 4, 1)
        elif x.ndim == 4:
            # [B, C, H, W] -> [B, H, W, C] (single frame)
            return x.permute(0, 2, 3, 1)
        else:
            raise ValueError(f"Unexpected tensor shape: {x.shape}")
    
    def encode_with_temporal_padding(
        self,
        x: torch.Tensor,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Encode with temporal padding to satisfy T = 4n + 1.
        
        Args:
            x: Input tensor in WAN2.1 format [B, C, T, H, W]
            return_dict: Whether to return as dict or tuple
            
        Returns:
            Encoded latent (potentially wrapped in output object)
        """
        # Apply temporal padding
        x_padded, original_t = pad_temporal_to_valid(x, dim=2)
        
        # Encode using base VAE
        if hasattr(self.base_vae, 'encode'):
            output = self.base_vae.encode(x_padded, return_dict=return_dict)
        else:
            # Fallback: call encoder directly
            h = self.base_vae.encoder(x_padded)
            
            # Handle quantization if present
            if hasattr(self.base_vae, 'quant_conv') and self.base_vae.quant_conv is not None:
                h = self.base_vae.quant_conv(h)
            
            # Apply scaling
            h = h * self.scaling_factor
            if self.shift_factor != 0.0:
                h = h + self.shift_factor
            
            output = h if not return_dict else type('Output', (), {'latent': h})()
        
        # Extract latent from output
        if hasattr(output, 'latent_dist'):
            latent = output.latent_dist.sample() if hasattr(output.latent_dist, 'sample') else output.latent_dist.mode()
        elif hasattr(output, 'latent'):
            latent = output.latent
        elif isinstance(output, torch.Tensor):
            latent = output
        else:
            latent = output[0] if isinstance(output, tuple) else output
        
        # Unpad temporal dimension in latent space (compressed by temporal_compression)
        original_t_latent = (original_t - 1) // self.temporal_compression + 1
        latent = unpad_temporal(latent, original_t_latent, dim=2)
        
        if return_dict:
            return type('Output', (), {'latent': latent})()
        return latent
    
    def decode_with_temporal_padding(
        self,
        z: torch.Tensor,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Decode with temporal padding to satisfy T = 4n + 1.
        
        Args:
            z: Latent tensor in WAN2.1 format [B, C, T, H, W]
            return_dict: Whether to return as dict or tuple
            
        Returns:
            Decoded sample (potentially wrapped in output object)
        """
        # Apply temporal padding in latent space
        z_padded, original_t_latent = pad_temporal_to_valid(z, dim=2)
        
        # Unapply scaling
        if self.shift_factor != 0.0:
            z_padded = z_padded - self.shift_factor
        z_padded = z_padded / self.scaling_factor
        
        # Decode using base VAE
        if hasattr(self.base_vae, 'decode'):
            output = self.base_vae.decode(z_padded, return_dict=return_dict)
        else:
            # Fallback: call decoder directly
            if hasattr(self.base_vae, 'post_quant_conv') and self.base_vae.post_quant_conv is not None:
                z_padded = self.base_vae.post_quant_conv(z_padded)
            
            sample = self.base_vae.decoder(z_padded)
            output = sample if not return_dict else type('Output', (), {'sample': sample})()
        
        # Extract sample from output
        if hasattr(output, 'sample'):
            sample = output.sample
        elif isinstance(output, torch.Tensor):
            sample = output
        else:
            sample = output[0] if isinstance(output, tuple) else output
        
        # Unpad temporal dimension (expanded by temporal_compression)
        original_t = (original_t_latent - 1) * self.temporal_compression + 1
        sample = unpad_temporal(sample, original_t, dim=2)
        
        if return_dict:
            return type('Output', (), {'sample': sample})()
        return sample
    
    def encode_with_3d_tiling(
        self,
        x: torch.Tensor,
        tile_size: Tuple[int, int] = (512, 512),
        tile_overlap: Tuple[int, int] = (64, 64),
        temporal_tile_size: int = 17,
        temporal_overlap: int = 4,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Encode with 3D tiling (temporal + spatial).
        
        Args:
            x: Input tensor in WAN2.1 format [B, C, T, H, W]
            tile_size: Spatial tile size (height, width)
            tile_overlap: Spatial tile overlap (height, width)
            temporal_tile_size: Number of frames per temporal chunk (should satisfy T = 4n + 1)
            temporal_overlap: Frame overlap between temporal chunks
            return_dict: Whether to return as dict
            
        Returns:
            Encoded latent (wrapped in output object if return_dict=True)
        """
        B, C, T, H, W = x.shape
        
        # Validate temporal tile size
        if (temporal_tile_size - 1) % 4 != 0:
            logger.warning(
                f"Temporal tile size {temporal_tile_size} doesn't satisfy T = 4n + 1, "
                f"adjusting to {calculate_valid_temporal_size(temporal_tile_size)}"
            )
            temporal_tile_size = calculate_valid_temporal_size(temporal_tile_size)
        
        # If input is small enough, use non-tiled encoding
        if T <= temporal_tile_size and H <= tile_size[0] and W <= tile_size[1]:
            return self.encode_with_temporal_padding(x, return_dict=return_dict)
        
        # Calculate temporal chunks
        temporal_stride = temporal_tile_size - temporal_overlap
        num_temporal_chunks = math.ceil((T - temporal_overlap) / temporal_stride)
        
        # Process temporal chunks
        latent_chunks = []
        for t_idx in range(num_temporal_chunks):
            t_start = t_idx * temporal_stride
            t_end = min(t_start + temporal_tile_size, T)
            
            # Extract temporal chunk
            x_chunk = x[:, :, t_start:t_end, :, :]
            
            # Pad temporal dimension if needed
            x_chunk, original_t = pad_temporal_to_valid(x_chunk, dim=2)
            
            # Encode chunk (with spatial tiling if needed)
            if H > tile_size[0] or W > tile_size[1]:
                # Use spatial tiling on this temporal chunk
                latent_chunk = self._encode_with_spatial_tiling(
                    x_chunk, tile_size, tile_overlap, return_dict=False
                )
            else:
                # Encode without spatial tiling
                if hasattr(self.base_vae, 'encode'):
                    output = self.base_vae.encode(x_chunk, return_dict=False)
                    latent_chunk = output[0] if isinstance(output, tuple) else output
                    if hasattr(latent_chunk, 'sample'):
                        latent_chunk = latent_chunk.sample()
                    elif hasattr(latent_chunk, 'mode'):
                        latent_chunk = latent_chunk.mode()
                else:
                    h = self.base_vae.encoder(x_chunk)
                    if hasattr(self.base_vae, 'quant_conv') and self.base_vae.quant_conv is not None:
                        h = self.base_vae.quant_conv(h)
                    latent_chunk = h * self.scaling_factor
                    if self.shift_factor != 0.0:
                        latent_chunk = latent_chunk + self.shift_factor
            
            # Unpad temporal dimension in latent space
            original_t_latent = (original_t - 1) // self.temporal_compression + 1
            latent_chunk = unpad_temporal(latent_chunk, original_t_latent, dim=2)
            
            latent_chunks.append(latent_chunk)
        
        # Blend overlapping temporal chunks
        if num_temporal_chunks > 1:
            latent = self._blend_temporal_chunks(
                latent_chunks, temporal_stride // self.temporal_compression, 
                temporal_overlap // self.temporal_compression, T
            )
        else:
            latent = latent_chunks[0]
        
        if return_dict:
            return type('Output', (), {'latent': latent})()
        return latent
    
    def _encode_with_spatial_tiling(
        self,
        x: torch.Tensor,
        tile_size: Tuple[int, int],
        tile_overlap: Tuple[int, int],
        return_dict: bool = False,
    ) -> torch.Tensor:
        """
        Encode with spatial tiling only (no temporal tiling).
        
        Args:
            x: Input tensor [B, C, T, H, W]
            tile_size: Spatial tile size
            tile_overlap: Spatial tile overlap
            return_dict: Whether to return as dict
            
        Returns:
            Encoded latent
        """
        # For now, use simple non-overlapping tiling
        # A more sophisticated implementation would handle overlaps and blending
        logger.warning("Spatial tiling for 3D tensors not fully implemented, using full encoding")
        return self.encode_with_temporal_padding(x, return_dict=return_dict)
    
    def _blend_temporal_chunks(
        self,
        chunks: list,
        stride: int,
        overlap: int,
        target_length: int
    ) -> torch.Tensor:
        """
        Blend overlapping temporal chunks with linear interpolation in overlap regions.
        
        Args:
            chunks: List of latent chunks
            stride: Stride between chunks in latent space
            overlap: Overlap size in latent space
            target_length: Target temporal length in latent space
            
        Returns:
            Blended latent tensor
        """
        if len(chunks) == 1:
            return chunks[0]
        
        B, C, _, H, W = chunks[0].shape
        target_t_latent = (target_length - 1) // self.temporal_compression + 1
        
        # Initialize output tensor
        output = torch.zeros((B, C, target_t_latent, H, W), 
                           dtype=chunks[0].dtype, device=chunks[0].device)
        weight_sum = torch.zeros((B, 1, target_t_latent, 1, 1), 
                                dtype=chunks[0].dtype, device=chunks[0].device)
        
        for i, chunk in enumerate(chunks):
            t_start = i * stride
            t_end = t_start + chunk.shape[2]
            
            # Clip to target length
            if t_end > target_t_latent:
                chunk = chunk[:, :, :(target_t_latent - t_start), :, :]
                t_end = target_t_latent
            
            # Create weight mask for blending
            chunk_len = t_end - t_start
            weight = torch.ones((B, 1, chunk_len, 1, 1), 
                              dtype=chunk.dtype, device=chunk.device)
            
            if overlap > 0:
                # Linear fade in/out in overlap regions
                if i > 0:  # Fade in at start
                    fade_in = torch.linspace(0, 1, min(overlap, chunk_len), 
                                           dtype=chunk.dtype, device=chunk.device)
                    weight[:, :, :len(fade_in), :, :] = fade_in.view(1, 1, -1, 1, 1)
                
                if i < len(chunks) - 1:  # Fade out at end
                    fade_out = torch.linspace(1, 0, min(overlap, chunk_len),
                                            dtype=chunk.dtype, device=chunk.device)
                    weight[:, :, -len(fade_out):, :, :] = fade_out.view(1, 1, -1, 1, 1)
            
            # Accumulate weighted chunks
            output[:, :, t_start:t_end, :, :] += chunk * weight
            weight_sum[:, :, t_start:t_end, :, :] += weight
        
        # Normalize by weight sum
        output = output / (weight_sum + 1e-8)
        
        return output
    
    def encode(
        self,
        x: torch.Tensor,
        return_dict: bool = True,
        tiled: bool = False,
        tile_size: Tuple[int, int] = (512, 512),
        tile_overlap: Tuple[int, int] = (64, 64),
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Encode input to latent space with SeedVR2-compatible interface.
        
        Args:
            x: Input tensor (can be SeedVR2 or WAN2.1 format)
            return_dict: Whether to return as dict
            tiled: Whether to use tiled encoding (3D temporal+spatial tiling)
            tile_size: Tile size for spatial tiling
            tile_overlap: Tile overlap for spatial tiling
            
        Returns:
            Encoded latent (wrapped in output object if return_dict=True)
        """
        # Convert to WAN2.1 format if needed
        x_wan21 = self.permute_to_wan21_format(x)
        
        if tiled:
            # Use 3D temporal+spatial tiling
            output = self.encode_with_3d_tiling(
                x_wan21, 
                tile_size=tile_size, 
                tile_overlap=tile_overlap,
                temporal_tile_size=17,  # Default: 17 frames (4*4 + 1)
                temporal_overlap=4,
                return_dict=return_dict
            )
        else:
            # Encode with temporal padding only
            output = self.encode_with_temporal_padding(x_wan21, return_dict=return_dict)
        
        return output
    
    def decode_with_3d_tiling(
        self,
        z: torch.Tensor,
        tile_size: Tuple[int, int] = (512, 512),
        tile_overlap: Tuple[int, int] = (64, 64),
        temporal_tile_size: int = 5,  # In latent space (will expand to 17 frames)
        temporal_overlap: int = 1,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Decode with 3D tiling (temporal + spatial).
        
        Args:
            z: Latent tensor in WAN2.1 format [B, C, T, H, W]
            tile_size: Spatial tile size (height, width)
            tile_overlap: Spatial tile overlap (height, width)
            temporal_tile_size: Number of latent frames per temporal chunk
            temporal_overlap: Latent frame overlap between temporal chunks
            return_dict: Whether to return as dict
            
        Returns:
            Decoded sample (wrapped in output object if return_dict=True)
        """
        B, C, T, H, W = z.shape
        
        # If input is small enough, use non-tiled decoding
        if T <= temporal_tile_size and H <= tile_size[0] and W <= tile_size[1]:
            return self.decode_with_temporal_padding(z, return_dict=return_dict)
        
        # Calculate temporal chunks
        temporal_stride = temporal_tile_size - temporal_overlap
        num_temporal_chunks = math.ceil((T - temporal_overlap) / temporal_stride)
        
        # Process temporal chunks
        sample_chunks = []
        for t_idx in range(num_temporal_chunks):
            t_start = t_idx * temporal_stride
            t_end = min(t_start + temporal_tile_size, T)
            
            # Extract temporal chunk
            z_chunk = z[:, :, t_start:t_end, :, :]
            
            # Pad temporal dimension if needed
            z_chunk, original_t_latent = pad_temporal_to_valid(z_chunk, dim=2)
            
            # Unapply scaling
            if self.shift_factor != 0.0:
                z_chunk = z_chunk - self.shift_factor
            z_chunk = z_chunk / self.scaling_factor
            
            # Decode chunk (with spatial tiling if needed)
            if H > tile_size[0] or W > tile_size[1]:
                # Use spatial tiling on this temporal chunk
                sample_chunk = self._decode_with_spatial_tiling(
                    z_chunk, tile_size, tile_overlap, return_dict=False
                )
            else:
                # Decode without spatial tiling
                if hasattr(self.base_vae, 'decode'):
                    output = self.base_vae.decode(z_chunk, return_dict=False)
                    sample_chunk = output[0] if isinstance(output, tuple) else output
                    if hasattr(sample_chunk, 'sample'):
                        sample_chunk = sample_chunk.sample
                else:
                    if hasattr(self.base_vae, 'post_quant_conv') and self.base_vae.post_quant_conv is not None:
                        z_chunk = self.base_vae.post_quant_conv(z_chunk)
                    sample_chunk = self.base_vae.decoder(z_chunk)
            
            # Unpad temporal dimension (expanded by temporal_compression)
            original_t = (original_t_latent - 1) * self.temporal_compression + 1
            sample_chunk = unpad_temporal(sample_chunk, original_t, dim=2)
            
            sample_chunks.append(sample_chunk)
        
        # Blend overlapping temporal chunks
        if num_temporal_chunks > 1:
            sample = self._blend_temporal_chunks_output(
                sample_chunks, temporal_stride * self.temporal_compression, 
                temporal_overlap * self.temporal_compression
            )
        else:
            sample = sample_chunks[0]
        
        if return_dict:
            return type('Output', (), {'sample': sample})()
        return sample
    
    def _decode_with_spatial_tiling(
        self,
        z: torch.Tensor,
        tile_size: Tuple[int, int],
        tile_overlap: Tuple[int, int],
        return_dict: bool = False,
    ) -> torch.Tensor:
        """
        Decode with spatial tiling only (no temporal tiling).
        
        Args:
            z: Latent tensor [B, C, T, H, W]
            tile_size: Spatial tile size
            tile_overlap: Spatial tile overlap
            return_dict: Whether to return as dict
            
        Returns:
            Decoded sample
        """
        # For now, use simple non-overlapping tiling
        # A more sophisticated implementation would handle overlaps and blending
        logger.warning("Spatial tiling for 3D tensors not fully implemented, using full decoding")
        return self.decode_with_temporal_padding(z, return_dict=return_dict)
    
    def _blend_temporal_chunks_output(
        self,
        chunks: list,
        stride: int,
        overlap: int
    ) -> torch.Tensor:
        """
        Blend overlapping temporal chunks in output space (similar to latent blending).
        
        Args:
            chunks: List of sample chunks
            stride: Stride between chunks in output space
            overlap: Overlap size in output space
            
        Returns:
            Blended sample tensor
        """
        if len(chunks) == 1:
            return chunks[0]
        
        B, C, _, H, W = chunks[0].shape
        
        # Calculate total output length
        total_length = stride * (len(chunks) - 1) + chunks[-1].shape[2]
        
        # Initialize output tensor
        output = torch.zeros((B, C, total_length, H, W), 
                           dtype=chunks[0].dtype, device=chunks[0].device)
        weight_sum = torch.zeros((B, 1, total_length, 1, 1), 
                                dtype=chunks[0].dtype, device=chunks[0].device)
        
        for i, chunk in enumerate(chunks):
            t_start = i * stride
            t_end = t_start + chunk.shape[2]
            
            # Create weight mask for blending
            chunk_len = chunk.shape[2]
            weight = torch.ones((B, 1, chunk_len, 1, 1), 
                              dtype=chunk.dtype, device=chunk.device)
            
            if overlap > 0:
                # Linear fade in/out in overlap regions
                if i > 0:  # Fade in at start
                    fade_in = torch.linspace(0, 1, min(overlap, chunk_len), 
                                           dtype=chunk.dtype, device=chunk.device)
                    weight[:, :, :len(fade_in), :, :] = fade_in.view(1, 1, -1, 1, 1)
                
                if i < len(chunks) - 1:  # Fade out at end
                    fade_out = torch.linspace(1, 0, min(overlap, chunk_len),
                                            dtype=chunk.dtype, device=chunk.device)
                    weight[:, :, -len(fade_out):, :, :] = fade_out.view(1, 1, -1, 1, 1)
            
            # Accumulate weighted chunks
            output[:, :, t_start:t_end, :, :] += chunk * weight
            weight_sum[:, :, t_start:t_end, :, :] += weight
        
        # Normalize by weight sum
        output = output / (weight_sum + 1e-8)
        
        return output
    
    def decode(
        self,
        z: torch.Tensor,
        return_dict: bool = True,
        tiled: bool = False,
        tile_size: Tuple[int, int] = (512, 512),
        tile_overlap: Tuple[int, int] = (64, 64),
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Decode latent to sample space with SeedVR2-compatible interface.
        
        Args:
            z: Latent tensor (can be SeedVR2 or WAN2.1 format)
            return_dict: Whether to return as dict
            tiled: Whether to use tiled decoding (3D temporal+spatial tiling)
            tile_size: Tile size for spatial tiling
            tile_overlap: Tile overlap for spatial tiling
            
        Returns:
            Decoded sample (wrapped in output object if return_dict=True)
        """
        # Convert to WAN2.1 format if needed
        z_wan21 = self.permute_to_wan21_format(z)
        
        if tiled:
            # Use 3D temporal+spatial tiling
            output = self.decode_with_3d_tiling(
                z_wan21, 
                tile_size=tile_size, 
                tile_overlap=tile_overlap,
                temporal_tile_size=5,  # Default: 5 latent frames (expands to 17 pixel frames)
                temporal_overlap=1,
                return_dict=return_dict
            )
        else:
            # Decode with temporal padding only
            output = self.decode_with_temporal_padding(z_wan21, return_dict=return_dict)
        
        return output
    
    @staticmethod
    def detect_wan21_state_dict(state_dict: Dict[str, torch.Tensor]) -> bool:
        """
        Detect if a state dict is from a WAN2.1 model.
        
        Looks for characteristic keys like:
        - Temporal convolution layers (conv3d, causal patterns)
        - WAN2.1-specific architecture markers
        
        Args:
            state_dict: Model state dictionary
            
        Returns:
            True if state dict appears to be from WAN2.1 model
        """
        # Check for 3D convolution keys
        has_conv3d = any('conv3d' in k.lower() for k in state_dict.keys())
        
        # Check for causal/temporal markers
        has_temporal = any(
            any(marker in k.lower() for marker in ['temporal', 'causal', 'time'])
            for k in state_dict.keys()
        )
        
        # Check for WAN2.1-specific patterns
        has_wan21_pattern = any(
            'v_blocks' in k or 'temporal_conv' in k or 'inflation' in k
            for k in state_dict.keys()
        )
        
        return has_conv3d and (has_temporal or has_wan21_pattern)
    
    @staticmethod
    def from_pretrained(
        pretrained_path: str,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        **kwargs
    ) -> 'Wan21VAEWrapper':
        """
        Load pretrained WAN2.1 VAE and wrap it.
        
        Args:
            pretrained_path: Path to checkpoint
            device: Target device
            dtype: Target dtype
            **kwargs: Additional arguments for VAE construction
            
        Returns:
            Wrapped WAN2.1 VAE
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if dtype is None:
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        
        # Load checkpoint
        checkpoint = torch.load(pretrained_path, map_location=device)
        
        # Extract config and scaling factor
        config = checkpoint.get('config', {})
        scaling_factor = config.get('scaling_factor', 0.18215)
        shift_factor = config.get('shift_factor', 0.0)
        
        # TODO: Construct base VAE from config
        # For now, this is a placeholder
        raise NotImplementedError(
            "WAN2.1 VAE construction from checkpoint not yet implemented. "
            "This requires the actual WAN2.1 model architecture."
        )


def wrap_vae_if_wan21(
    vae: nn.Module,
    state_dict: Optional[Dict[str, torch.Tensor]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> nn.Module:
    """
    Wrap VAE with WAN2.1 wrapper if it's a WAN2.1 model.
    
    Args:
        vae: VAE model to potentially wrap
        state_dict: State dict for detection (optional)
        config: Configuration dict (optional)
        
    Returns:
        Wrapped VAE if WAN2.1, otherwise original VAE
    """
    # Check if already wrapped
    if isinstance(vae, Wan21VAEWrapper):
        return vae
    
    # Detect WAN2.1
    is_wan21 = False
    
    if state_dict is not None:
        is_wan21 = Wan21VAEWrapper.detect_wan21_state_dict(state_dict)
    
    if config is not None and not is_wan21:
        # Check config for WAN2.1 markers
        model_type = config.get('model_type', '').lower()
        architecture = config.get('architecture', {})
        is_wan21 = 'wan2.1' in model_type or architecture.get('temporal_compression') == 4
    
    if is_wan21:
        logger.info("Detected WAN2.1 model, applying wrapper")
        
        scaling_factor = 0.18215
        shift_factor = 0.0
        temporal_compression = 4
        spatial_compression = 8
        
        if config is not None:
            # Extract scaling factors from config
            scaling_factor = config.get('scaling_factor', scaling_factor)
            shift_factor = config.get('shift_factor', shift_factor)
            
            # Extract compression ratios
            architecture = config.get('architecture', {})
            temporal_compression = architecture.get('temporal_compression', temporal_compression)
            spatial_compression = architecture.get('spatial_compression', spatial_compression)
        
        return Wan21VAEWrapper(
            base_vae=vae,
            scaling_factor=scaling_factor,
            shift_factor=shift_factor,
            temporal_compression=temporal_compression,
            spatial_compression=spatial_compression,
        )
    
    return vae


def load_vae_config_json(config_path: str) -> Dict[str, Any]:
    """
    Load VAE configuration from a config.json file.
    
    This is useful for loading WAN2.1 models that include a config.json
    file alongside the checkpoint with model-specific parameters like
    scaling_factor, temporal_compression, etc.
    
    Args:
        config_path: Path to config.json file
        
    Returns:
        Configuration dictionary
        
    Example:
        >>> config = load_vae_config_json("/path/to/wan2.1/config.json")
        >>> scaling_factor = config.get('scaling_factor', 0.18215)
    """
    import json
    import os
    
    if not os.path.exists(config_path):
        logger.warning(f"Config file not found: {config_path}")
        return {}
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded VAE config from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading config from {config_path}: {e}")
        return {}
