# Wan2.1 VAE Wrapper
# Integrates Wan2.1 3D Causal VAE into SeedVR2

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union
from .video_vae import VideoAutoencoderKLWrapper, VideoAutoencoderKL, CausalEncoderOutput, CausalDecoderOutput
from ....common.logger import get_logger
from ....common.distributed.advanced import get_sequence_parallel_world_size
from .types import MemoryState

logger = get_logger(__name__)

class WanVAEWrapper(VideoAutoencoderKLWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Wan2.1 scaling factor
        self._scaling_factor = 0.18215

    @property
    def scaling_factor(self):
        return self._scaling_factor

    def _pad_temporal(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Pad input tensor to satisfy T = 4n + 1 requirement for Wan2.1
        Args:
            x: Input tensor [B, C, T, H, W]
        Returns:
            padded_x: Padded tensor
            pad_amount: Amount padded (to crop later if needed)
        """
        t = x.shape[2]
        # Formula: T = 4n + 1
        # We need to find the next valid T
        remainder = (t - 1) % 4
        if remainder == 0:
            return x, 0

        pad_amount = 4 - remainder
        # Reflect pad the last frames. Replicate is safer for causality at boundaries.
        padding = (0, 0, 0, 0, 0, pad_amount)
        padded_x = F.pad(x, padding, mode='replicate')
        return padded_x, pad_amount

    def slicing_encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        3D Temporal Tiling for Encoding with Overlap and Blending
        Args:
            x: [B, C, T, H, W]
        """
        sp_size = get_sequence_parallel_world_size()

        # Use config values for tile/overlap if available, else defaults
        tile_size = self.slicing_sample_min_size * sp_size
        overlap = getattr(self, 'encode_tile_overlap', 4) # Default overlap 4 frames

        # Temporal compression ratio
        t_compress = self.temporal_downsample_factor

        # Stride
        stride = tile_size - overlap

        # Adjust stride to be divisible by t_compress
        if stride % t_compress != 0:
            new_stride = (stride // t_compress) * t_compress
            if new_stride == 0:
                new_stride = t_compress

            if tile_size < new_stride:
                tile_size = new_stride

            overlap = tile_size - new_stride
            stride = new_stride

        # If input is too short, use base method
        if not self.use_slicing or x.shape[2] <= tile_size:
            return self._encode(x, memory_state=MemoryState.DISABLED)

        # Sliding window encoding
        B, C, T, H, W = x.shape

        chunks = []
        for i in range(0, T, stride):
            start = i
            end = min(i + tile_size, T)

            if (end - start) < tile_size and start > 0:
                start = max(0, end - tile_size)
                # Align start to t_compress
                start = (start // t_compress) * t_compress

            chunk_input = x[:, :, start:end, :, :]

            # Encode chunk independently
            chunk_latent = self._encode(chunk_input, memory_state=MemoryState.DISABLED)

            chunks.append((start, end, chunk_latent))

            if end == T:
                break

        # Initialize output buffer
        total_latent_t = (T - 1) // t_compress + 1
        _, C_lat, _, H_lat, W_lat = chunks[0][2].shape

        output_latent = torch.zeros(B, C_lat, total_latent_t, H_lat, W_lat,
                                   device=x.device, dtype=chunks[0][2].dtype)
        output_weight = torch.zeros(B, C_lat, total_latent_t, H_lat, W_lat,
                                   device=x.device, dtype=chunks[0][2].dtype)

        for start_pixel, end_pixel, latent in chunks:
            latent_start = start_pixel // t_compress
            latent_len = latent.shape[2]

            target_slice = slice(latent_start, latent_start + latent_len)

            if latent_start + latent_len > total_latent_t:
                 valid_len = total_latent_t - latent_start
                 if valid_len <= 0: continue
                 latent = latent[:, :, :valid_len, :, :]
                 target_slice = slice(latent_start, total_latent_t)

            # Use constant weights for averaging overlap
            weight = 1.0

            output_latent[:, :, target_slice, :, :] += latent * weight
            output_weight[:, :, target_slice, :, :] += weight

        # Normalize
        mask = output_weight > 0
        output_latent[mask] /= output_weight[mask]

        return output_latent

    def slicing_decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        3D Temporal Tiling for Decoding with Overlap and Blending
        Args:
            z: [B, C, T_lat, H_lat, W_lat]
        """
        # Use base decoding if z is small
        if not self.use_slicing:
             return self._decode(z, memory_state=MemoryState.DISABLED)

        tile_size = self.slicing_latent_min_size
        overlap = getattr(self, 'decode_tile_overlap', 1)

        if z.shape[2] <= tile_size:
             return self._decode(z, memory_state=MemoryState.DISABLED)

        B, C, T, H, W = z.shape
        stride = tile_size - overlap
        if stride <= 0: stride = tile_size // 2

        chunks = []
        for i in range(0, T, stride):
            start = i
            end = min(i + tile_size, T)
            if (end - start) < tile_size and start > 0:
                start = max(0, end - tile_size)

            chunk_input = z[:, :, start:end, :, :]
            chunk_pixel = self._decode(chunk_input, memory_state=MemoryState.DISABLED)
            chunks.append((start, end, chunk_pixel))

            if end == T: break

        # Output buffer
        t_compress = self.temporal_downsample_factor
        total_pixel_t = (T - 1) * t_compress + 1

        _, C_pix, _, H_pix, W_pix = chunks[0][2].shape

        output_pixel = torch.zeros(B, C_pix, total_pixel_t, H_pix, W_pix,
                                  device=z.device, dtype=chunks[0][2].dtype)
        output_weight = torch.zeros(B, C_pix, total_pixel_t, H_pix, W_pix,
                                   device=z.device, dtype=chunks[0][2].dtype)

        for start_lat, end_lat, pixel in chunks:
            # Latent start i corresponds to Pixel start i*4
            pixel_start = start_lat * t_compress
            pixel_len = pixel.shape[2]

            target_slice = slice(pixel_start, pixel_start + pixel_len)

            if pixel_start + pixel_len > total_pixel_t:
                valid_len = total_pixel_t - pixel_start
                pixel = pixel[:, :, :valid_len, :, :]
                target_slice = slice(pixel_start, total_pixel_t)

            # Use constant weights for averaging overlap
            weight = 1.0

            output_pixel[:, :, target_slice, :, :] += pixel * weight
            output_weight[:, :, target_slice, :, :] += weight

        mask = output_weight > 0
        output_pixel[mask] /= output_weight[mask]

        return output_pixel

    def encode(self, x: torch.Tensor) -> CausalEncoderOutput:
        """
        Encode video frames to latent space.
        Args:
            x: Input tensor [B, T, H, W, C] (SeedVR2 default)
        Returns:
            CausalEncoderOutput with latent [B, T, H, W, C] (SeedVR2 default)
        """
        # 1. Permute [B, T, H, W, C] -> [B, C, T, H, W]
        x = x.permute(0, 4, 1, 2, 3)

        # 2. Temporal Padding
        x_padded, pad_amount = self._pad_temporal(x)

        # 3. Encode
        if self.use_slicing:
             h = self.slicing_encode(x_padded)
        else:
             h = self._encode(x_padded, memory_state=MemoryState.DISABLED)

        # 4. Sample
        from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
        p = DiagonalGaussianDistribution(h)
        z = p.sample()

        # 5. Permute back [B, C, T, H, W] -> [B, T, H, W, C]
        z = z.permute(0, 2, 3, 4, 1) # [B, T, H, W, C]

        return CausalEncoderOutput(z, p)

    def decode(self, z: torch.Tensor) -> CausalDecoderOutput:
        """
        Decode latents to video frames.
        Args:
            z: Latent tensor [B, T, H, W, C]
        Returns:
            CausalDecoderOutput with video [B, T, H, W, C]
        """
        # 1. Permute [B, T, H, W, C] -> [B, C, T, H, W]
        z = z.permute(0, 4, 1, 2, 3)

        # 2. Decode
        if self.use_slicing:
            x = self.slicing_decode(z)
        else:
            x = self._decode(z, memory_state=MemoryState.DISABLED)

        # 3. Permute back [B, C, T, H, W] -> [B, T, H, W, C]
        x = x.permute(0, 2, 3, 4, 1)

        return CausalDecoderOutput(x)
