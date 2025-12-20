# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Adapted for ComfyUI-SeedVR2.5 integration
"""
Wan2.1 VAE Implementation with ComfyUI Integration
Includes encoder/decoder blocks and WanVAE wrapper class with tiling support
"""

import logging
import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional, Tuple, List, NamedTuple

__all__ = [
    'WanVAE',
    'WanVAE_',
    'WanVAEWrapper',
    'WanVAEEncoderOutput',
    'WanVAEDecoderOutput',
]


class WanVAEEncoderOutput(NamedTuple):
    """Output from WanVAE encoder to maintain compatibility with existing VAE interface"""
    latent: torch.Tensor
    posterior: Optional[object] = None


class WanVAEDecoderOutput(NamedTuple):
    """Output from WanVAE decoder to maintain compatibility with existing VAE interface"""
    sample: torch.Tensor

CACHE_T = 2


class CausalConv3d(nn.Conv3d):
    """
    Causal 3d convolution.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._padding = (self.padding[2], self.padding[2], self.padding[1],
                         self.padding[1], 2 * self.padding[0], 0)
        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)

        return super().forward(x)


class RMS_norm(nn.Module):

    def __init__(self, dim, channel_first=True, images=True, bias=False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.

    def forward(self, x):
        return F.normalize(
            x, dim=(1 if self.channel_first else
                    -1)) * self.scale * self.gamma + self.bias


class Upsample(nn.Upsample):

    def forward(self, x):
        """
        Fix bfloat16 support for nearest neighbor interpolation.
        """
        return super().forward(x.float()).type_as(x)


class Resample(nn.Module):

    def __init__(self, dim, mode):
        assert mode in ('none', 'upsample2d', 'upsample3d', 'downsample2d',
                        'downsample3d')
        super().__init__()
        self.dim = dim
        self.mode = mode

        # layers
        if mode == 'upsample2d':
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2., 2.), mode='nearest-exact'),
                nn.Conv2d(dim, dim // 2, 3, padding=1))
        elif mode == 'upsample3d':
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2., 2.), mode='nearest-exact'),
                nn.Conv2d(dim, dim // 2, 3, padding=1))
            self.time_conv = CausalConv3d(
                dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))

        elif mode == 'downsample2d':
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)))
        elif mode == 'downsample3d':
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)))
            self.time_conv = CausalConv3d(
                dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))

        else:
            self.resample = nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        b, c, t, h, w = x.size()
        if self.mode == 'upsample3d':
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = 'Rep'
                    feat_idx[0] += 1
                else:

                    cache_x = x[:, :, -CACHE_T:, :, :].clone()
                    if cache_x.shape[2] < 2 and feat_cache[
                            idx] is not None and feat_cache[idx] != 'Rep':
                        # cache last frame of last two chunk
                        cache_x = torch.cat([
                            feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                                cache_x.device), cache_x
                        ],
                                            dim=2)
                    if cache_x.shape[2] < 2 and feat_cache[
                            idx] is not None and feat_cache[idx] == 'Rep':
                        cache_x = torch.cat([
                            torch.zeros_like(cache_x).to(cache_x.device),
                            cache_x
                        ],
                                            dim=2)
                    if feat_cache[idx] == 'Rep':
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1

                    x = x.reshape(b, 2, c, t, h, w)
                    x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]),
                                    3)
                    x = x.reshape(b, c, t * 2, h, w)
        t = x.shape[2]
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.resample(x)
        x = rearrange(x, '(b t) c h w -> b c t h w', t=t)

        if self.mode == 'downsample3d':
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x.clone()
                    feat_idx[0] += 1
                else:

                    cache_x = x[:, :, -1:, :, :].clone()
                    # if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx]!='Rep':
                    #     # cache last frame of last two chunk
                    #     cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)

                    x = self.time_conv(
                        torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
        return x

    def init_weight(self, conv):
        conv_weight = conv.weight
        nn.init.zeros_(conv_weight)
        c1, c2, t, h, w = conv_weight.size()
        one_matrix = torch.eye(c1, c2)
        init_matrix = one_matrix
        nn.init.zeros_(conv_weight)
        #conv_weight.data[:,:,-1,1,1] = init_matrix * 0.5
        conv_weight.data[:, :, 1, 0, 0] = init_matrix  #* 0.5
        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def init_weight2(self, conv):
        conv_weight = conv.weight.data
        nn.init.zeros_(conv_weight)
        c1, c2, t, h, w = conv_weight.size()
        init_matrix = torch.eye(c1 // 2, c2)
        #init_matrix = repeat(init_matrix, 'o ... -> (o 2) ...').permute(1,0,2).contiguous().reshape(c1,c2)
        conv_weight[:c1 // 2, :, -1, 0, 0] = init_matrix
        conv_weight[c1 // 2:, :, -1, 0, 0] = init_matrix
        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)


class ResidualBlock(nn.Module):

    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # layers
        self.residual = nn.Sequential(
            RMS_norm(in_dim, images=False), nn.SiLU(),
            CausalConv3d(in_dim, out_dim, 3, padding=1),
            RMS_norm(out_dim, images=False), nn.SiLU(), nn.Dropout(dropout),
            CausalConv3d(out_dim, out_dim, 3, padding=1))
        self.shortcut = CausalConv3d(in_dim, out_dim, 1) \
            if in_dim != out_dim else nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        h = self.shortcut(x)
        for layer in self.residual:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                            cache_x.device), cache_x
                    ],
                                        dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x + h


class AttentionBlock(nn.Module):
    """
    Causal self-attention with a single head.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # layers
        self.norm = RMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

        # zero out the last layer params
        nn.init.zeros_(self.proj.weight)

    def forward(self, x):
        identity = x
        b, c, t, h, w = x.size()
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.norm(x)
        # compute query, key, value
        q, k, v = self.to_qkv(x).reshape(b * t, 1, c * 3,
                                         -1).permute(0, 1, 3,
                                                     2).contiguous().chunk(
                                                         3, dim=-1)

        # apply attention
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
        )
        x = x.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)

        # output
        x = self.proj(x)
        x = rearrange(x, '(b t) c h w-> b c t h w', t=t)
        return x + identity


class Encoder3d(nn.Module):

    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_downsample=[True, True, False],
                 dropout=0.0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample

        # dimensions
        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0

        # init block
        self.conv1 = CausalConv3d(3, dims[0], 3, padding=1)

        # downsample blocks
        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            for _ in range(num_res_blocks):
                downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    downsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            # downsample block
            if i != len(dim_mult) - 1:
                mode = 'downsample3d' if temperal_downsample[
                    i] else 'downsample2d'
                downsamples.append(Resample(out_dim, mode=mode))
                scale /= 2.0
        self.downsamples = nn.Sequential(*downsamples)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(out_dim, out_dim, dropout), AttentionBlock(out_dim),
            ResidualBlock(out_dim, out_dim, dropout))

        # output blocks
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False), nn.SiLU(),
            CausalConv3d(out_dim, z_dim, 3, padding=1))

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([
                    feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                        cache_x.device), cache_x
                ],
                                    dim=2)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        ## downsamples
        for layer in self.downsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## middle
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## head
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                            cache_x.device), cache_x
                    ],
                                        dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x


class Decoder3d(nn.Module):

    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_upsample=[False, True, True],
                 dropout=0.0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample

        # dimensions
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        scale = 1.0 / 2**(len(dim_mult) - 2)

        # init block
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0], dropout), AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0], dropout))

        # upsample blocks
        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            if i == 1 or i == 2 or i == 3:
                in_dim = in_dim // 2
            for _ in range(num_res_blocks + 1):
                upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    upsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            # upsample block
            if i != len(dim_mult) - 1:
                mode = 'upsample3d' if temperal_upsample[i] else 'upsample2d'
                upsamples.append(Resample(out_dim, mode=mode))
                scale *= 2.0
        self.upsamples = nn.Sequential(*upsamples)

        # output blocks
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False), nn.SiLU(),
            CausalConv3d(out_dim, 3, 3, padding=1))

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        ## conv1
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([
                    feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                        cache_x.device), cache_x
                ],
                                    dim=2)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        ## middle
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## upsamples
        for layer in self.upsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## head
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                            cache_x.device), cache_x
                    ],
                                        dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x


def count_conv3d(model):
    count = 0
    for m in model.modules():
        if isinstance(m, CausalConv3d):
            count += 1
    return count


class WanVAE_(nn.Module):

    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_downsample=[True, True, False],
                 dropout=0.0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_downsample[::-1]

        # modules
        self.encoder = Encoder3d(dim, z_dim * 2, dim_mult, num_res_blocks,
                                 attn_scales, self.temperal_downsample, dropout)
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(dim, z_dim, dim_mult, num_res_blocks,
                                 attn_scales, self.temperal_upsample, dropout)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var

    def encode(self, x, scale):
        # Fixed: Use isolated cache for each encode operation
        # The cache mechanism is designed for temporal consistency WITHIN an encode operation,
        # not ACROSS encode/decode operations
        self.clear_cache()
        
        t = x.shape[2]
        iter_ = 1 + (t - 1) // 4
        
        # CRITICAL FIX: Initialize fresh cache list for this encode operation
        encode_feat_map = [None] * len(list(self.encoder.modules()))
        # DON'T reset conv_idx in loop - it should persist across chunks!
        encode_conv_idx = [0]
        
        # Collect chunks in list to avoid repeated concatenation (memory efficient)
        chunks = []
        
        # Process temporal chunks with isolated caching
        for i in range(iter_):
            if i == 0:
                # First chunk: 1 frame
                chunk = self.encoder(
                    x[:, :, :1, :, :],
                    feat_cache=encode_feat_map,
                    feat_idx=encode_conv_idx)
            else:
                # Subsequent chunks: 4 frames each
                chunk = self.encoder(
                    x[:, :, 1 + 4 * (i - 1):1 + 4 * i, :, :],
                    feat_cache=encode_feat_map,
                    feat_idx=encode_conv_idx)
            chunks.append(chunk)
        
        # Concatenate all chunks once at the end (more memory efficient)
        out = torch.cat(chunks, dim=2)
        del chunks
        
        # Apply conv1 and split into mean and log_var
        mu, log_var = self.conv1(out).chunk(2, dim=1)
        
        # Apply scaling
        if isinstance(scale[0], torch.Tensor):
            mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(
                1, self.z_dim, 1, 1, 1)
        else:
            mu = (mu - scale[0]) * scale[1]
        
        self.clear_cache()
        return mu

    def decode(self, z, scale):
        # Fixed: Decode using fresh cache for each decode operation
        # The cache mechanism is designed for temporal consistency WITHIN a decode operation,
        # not ACROSS encode/decode operations
        self.clear_cache()
        
        # Apply reverse scaling
        if isinstance(scale[0], torch.Tensor):
            z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(
                1, self.z_dim, 1, 1, 1)
        else:
            z = z / scale[1] + scale[0]
        
        # Apply conv2
        x = self.conv2(z)
        
        # Decode frame-by-frame with isolated cache (iterate over latent frames)
        # CRITICAL FIX: Initialize fresh cache list for this decode operation
        decode_feat_map = [None] * len(list(self.decoder.modules()))
        # DON'T reset conv_idx in loop - it should persist across frames!
        decode_conv_idx = [0]
        iter_ = x.shape[2]
        
        # Collect frames in list to avoid repeated concatenation (memory efficient)
        frames = []
        
        for i in range(iter_):
            # Decode each latent frame, cache persists across iterations
            frame = self.decoder(
                x[:, :, i:i + 1, :, :],
                feat_cache=decode_feat_map,
                feat_idx=decode_conv_idx)
            frames.append(frame)
        
        # Concatenate all frames once at the end (more memory efficient)
        out = torch.cat(frames, dim=2)
        del frames
        
        self.clear_cache()
        return out

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def sample(self, imgs, deterministic=False):
        mu, log_var = self.encode(imgs)
        if deterministic:
            return mu
        std = torch.exp(0.5 * log_var.clamp(-30.0, 20.0))
        return mu + std * torch.randn_like(std)

    def clear_cache(self):
        self._conv_num = count_conv3d(self.decoder)
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num
        # Cache encode
        self._enc_conv_num = count_conv3d(self.encoder)
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num


def _video_vae(pretrained_path=None, z_dim=None, device='cpu', **kwargs):
    """
    Autoencoder3d adapted from Stable Diffusion 1.x, 2.x and XL.
    """
    # params
    cfg = dict(
        dim=96,
        z_dim=z_dim,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        dropout=0.0)
    cfg.update(**kwargs)

    # init model
    with torch.device('meta'):
        model = WanVAE_(**cfg)

    # load checkpoint
    logging.info(f'loading {pretrained_path}')
    model.load_state_dict(
        torch.load(pretrained_path, map_location=device), assign=True)

    return model


class WanVAE:

    def __init__(self,
                 z_dim=16,
                 vae_pth='cache/vae_step_411000.pth',
                 dtype=torch.float,
                 device="cuda"):
        self.dtype = dtype
        self.device = device

        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]
        self.mean = torch.tensor(mean, dtype=dtype, device=device)
        self.std = torch.tensor(std, dtype=dtype, device=device)
        self.scale = [self.mean, 1.0 / self.std]

        # init model
        self.model = _video_vae(
            pretrained_path=vae_pth,
            z_dim=z_dim,
        ).eval().requires_grad_(False).to(device)

    def encode(self, videos):
        """
        videos: A list of videos each with shape [C, T, H, W].
        
        Memory-efficient encoding with temporal chunking at wrapper level.
        For large videos (>32 frames), splits into smaller chunks to prevent OOM.
        Each chunk is encoded separately, then results are concatenated.
        """
        latents = []
        max_frames_per_chunk = 32  # Process max 32 frames at a time to avoid OOM
        
        # Use CUDA autocast for compatibility with PyTorch 2.7.1
        if self.device.type == 'cuda':
            with torch.cuda.amp.autocast(enabled=True, dtype=self.dtype):
                # Process each video separately
                for u in videos:
                    T = u.shape[1]  # Temporal dimension
                    
                    if T <= max_frames_per_chunk:
                        # Small video: process directly
                        latent = self.model.encode(u.unsqueeze(0), self.scale).float().squeeze(0)
                        latents.append(latent)
                    else:
                        # Large video: split into temporal chunks
                        chunk_latents = []
                        for start_idx in range(0, T, max_frames_per_chunk):
                            end_idx = min(start_idx + max_frames_per_chunk, T)
                            chunk = u[:, start_idx:end_idx, :, :]
                            
                            # Encode chunk
                            chunk_latent = self.model.encode(chunk.unsqueeze(0), self.scale).float().squeeze(0)
                            chunk_latents.append(chunk_latent)
                            
                            # Free GPU memory immediately after each chunk
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        
                        # Concatenate chunks along temporal dimension
                        latent = torch.cat(chunk_latents, dim=1)  # Concat on T dimension
                        latents.append(latent)
                    
                    # Free GPU memory after each video
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        else:
            # For non-CUDA devices, run without autocast (same chunking logic)
            for u in videos:
                T = u.shape[1]
                
                if T <= max_frames_per_chunk:
                    latent = self.model.encode(u.unsqueeze(0), self.scale).float().squeeze(0)
                    latents.append(latent)
                else:
                    chunk_latents = []
                    for start_idx in range(0, T, max_frames_per_chunk):
                        end_idx = min(start_idx + max_frames_per_chunk, T)
                        chunk = u[:, start_idx:end_idx, :, :]
                        chunk_latent = self.model.encode(chunk.unsqueeze(0), self.scale).float().squeeze(0)
                        chunk_latents.append(chunk_latent)
                    latent = torch.cat(chunk_latents, dim=1)
                    latents.append(latent)
        
        return latents

    def decode(self, zs):
        """
        Decode latent tensors to pixel space with temporal chunking.
        
        For large latents (>8 temporal frames after 4x downsampling),
        splits into smaller chunks to prevent OOM during decoding.
        """
        samples = []
        max_latent_frames_per_chunk = 8  # Process max 8 latent frames at a time (= 32 video frames)
        
        # Use CUDA autocast for compatibility with PyTorch 2.7.1
        if self.device.type == 'cuda':
            with torch.cuda.amp.autocast(enabled=True, dtype=self.dtype):
                # Process each latent separately
                for u in zs:
                    T = u.shape[1]  # Temporal dimension in latent space
                    
                    if T <= max_latent_frames_per_chunk:
                        # Small latent: decode directly
                        sample = self.model.decode(u.unsqueeze(0),
                                                 self.scale).float().clamp_(-1, 1).squeeze(0)
                        samples.append(sample)
                    else:
                        # Large latent: split into temporal chunks
                        chunk_samples = []
                        for start_idx in range(0, T, max_latent_frames_per_chunk):
                            end_idx = min(start_idx + max_latent_frames_per_chunk, T)
                            chunk = u[:, start_idx:end_idx, :, :]
                            
                            # Decode chunk
                            chunk_sample = self.model.decode(chunk.unsqueeze(0),
                                                           self.scale).float().clamp_(-1, 1).squeeze(0)
                            chunk_samples.append(chunk_sample)
                            
                            # Free GPU memory immediately after each chunk
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        
                        # Concatenate chunks along temporal dimension
                        sample = torch.cat(chunk_samples, dim=1)  # Concat on T dimension
                        samples.append(sample)
                    
                    # Free GPU memory after each latent
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        else:
            # For non-CUDA devices, run without autocast (same chunking logic)
            for u in zs:
                T = u.shape[1]
                
                if T <= max_latent_frames_per_chunk:
                    sample = self.model.decode(u.unsqueeze(0),
                                             self.scale).float().clamp_(-1, 1).squeeze(0)
                    samples.append(sample)
                else:
                    chunk_samples = []
                    for start_idx in range(0, T, max_latent_frames_per_chunk):
                        end_idx = min(start_idx + max_latent_frames_per_chunk, T)
                        chunk = u[:, start_idx:end_idx, :, :]
                        chunk_sample = self.model.decode(chunk.unsqueeze(0),
                                                       self.scale).float().clamp_(-1, 1).squeeze(0)
                        chunk_samples.append(chunk_sample)
                    sample = torch.cat(chunk_samples, dim=1)
                    samples.append(sample)
        
        return samples


class WanVAEWrapper(nn.Module):
    """
    Wrapper for Wan2.1 VAE with ComfyUI integration
    
    Based on FlashVSR's approach: uses Wan2.1 VAE as-is without spatial tiling.
    The VAE's internal temporal chunking (1+4+4+4 frames) handles memory efficiently.
    
    Provides:
    - Device management (device, offload_device)
    - Sequential batch processing
    - Compatible interface with existing VAE loaders
    """
    
    def __init__(self, vae_model: WanVAE, device: torch.device, 
                 offload_device: Optional[torch.device] = None):
        super().__init__()
        self.vae = vae_model
        self.device = device
        self.offload_device = offload_device
        
    def to(self, device):
        """Move model to device"""
        self.device = device
        self.vae.device = device
        self.vae.model = self.vae.model.to(device)
        self.vae.mean = self.vae.mean.to(device)
        self.vae.std = self.vae.std.to(device)
        self.vae.scale = [self.vae.mean, 1.0 / self.vae.std]
        return self
    
    def _encode_single(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a single video (no batch dimension) using VAE's internal temporal chunking"""
        # x shape: (C, T, H, W)
        # Returns: (Z, T//4, H//8, W//8)
        batch_list = [x]
        latents = self.vae.encode(batch_list)
        return latents[0]
    
    def encode(self, x: torch.Tensor, tiled: bool = False, 
               tile_size: Optional[Tuple[int, int]] = None,
               tile_overlap: Optional[Tuple[int, int]] = None) -> WanVAEEncoderOutput:
        """
        Encode input tensor to latent space with spatial tiling support (like SeedVR2 VAE)
        
        Args:
            x: Input tensor of shape (B, C, T, H, W)
            tiled: Whether to use spatial tiling
            tile_size: Tile size (H, W) in pixel space
            tile_overlap: Overlap size (H, W) in pixel space
            
        Returns:
            WanVAEEncoderOutput with latent tensor of shape (B, Z, T//4, H//8, W//8)
        """
        B, C, T, H, W = x.shape
        
        # If not tiled or small enough, encode directly
        if not tiled or (tile_size and H <= tile_size[0] and W <= tile_size[1]):
            batch_list = [x[i] for i in range(B)]
            latents = self.vae.encode(batch_list)
            latent = torch.stack(latents, dim=0)
            return WanVAEEncoderOutput(latent=latent, posterior=None)
        
        # Spatial tiling implementation (matching SeedVR2 VAE pattern)
        tile_size = tile_size or (512, 512)
        tile_overlap = tile_overlap or (64, 64)
        
        scale_factor = 8  # Wan2.1 VAE spatial downsampling
        tile_h, tile_w = tile_size
        overlap_h, overlap_w = tile_overlap
        
        # Convert to latent space
        latent_tile_h = max(1, tile_h // scale_factor)
        latent_tile_w = max(1, tile_w // scale_factor)
        latent_overlap_h = max(0, min(overlap_h // scale_factor, latent_tile_h - 1))
        latent_overlap_w = max(0, min(overlap_w // scale_factor, latent_tile_w - 1))
        
        stride_h = max(1, latent_tile_h - latent_overlap_h)
        stride_w = max(1, latent_tile_w - latent_overlap_w)
        
        H_lat_total = (H + scale_factor - 1) // scale_factor
        W_lat_total = (W + scale_factor - 1) // scale_factor
        
        # Pre-compute blending ramps
        ramp_h = None
        ramp_w = None
        if latent_overlap_h > 0:
            t_h = torch.linspace(0, 1, steps=latent_overlap_h, device=x.device, dtype=x.dtype)
            ramp_h = 0.5 - 0.5 * torch.cos(t_h * torch.pi)
        if latent_overlap_w > 0:
            t_w = torch.linspace(0, 1, steps=latent_overlap_w, device=x.device, dtype=x.dtype)
            ramp_w = 0.5 - 0.5 * torch.cos(t_w * torch.pi)
        
        # Process each batch item separately
        batch_latents = []
        for b in range(B):
            result = None
            count = None
            
            # Tile the spatial dimensions
            for y_lat in range(0, H_lat_total, stride_h):
                y_lat_end = min(y_lat + latent_tile_h, H_lat_total)
                for x_lat in range(0, W_lat_total, stride_w):
                    x_lat_end = min(x_lat + latent_tile_w, W_lat_total)
                    
                    # Skip tiny overlap tiles
                    if (y_lat > 0 and (y_lat_end - y_lat) <= latent_overlap_h) or \
                       (x_lat > 0 and (x_lat_end - x_lat) <= latent_overlap_w):
                        continue
                    
                    # Map to pixel space
                    y_out = y_lat * scale_factor
                    x_out = x_lat * scale_factor
                    y_out_end = min(y_lat_end * scale_factor, H)
                    x_out_end = min(x_lat_end * scale_factor, W)
                    
                    # Extract tile
                    tile_sample = x[b:b+1, :, :, y_out:y_out_end, x_out:x_out_end]
                    
                    # Encode tile
                    tile_latent = self._encode_single(tile_sample[0])
                    
                    # Initialize result on first tile
                    if result is None:
                        Z, T_lat, _, _ = tile_latent.shape
                        result = torch.zeros((Z, T_lat, H_lat_total, W_lat_total), 
                                           device=tile_latent.device, dtype=tile_latent.dtype)
                        count = torch.zeros((Z, T_lat, H_lat_total, W_lat_total), 
                                          device=tile_latent.device, dtype=tile_latent.dtype)
                    
                    # Compute blending weights
                    tile_h_actual = tile_latent.shape[2]
                    tile_w_actual = tile_latent.shape[3]
                    weight = torch.ones((Z, T_lat, tile_h_actual, tile_w_actual), 
                                       device=tile_latent.device, dtype=tile_latent.dtype)
                    
                    # Apply blending ramps
                    if latent_overlap_h > 0 and y_lat > 0:
                        blend_h = min(latent_overlap_h, tile_h_actual)
                        weight[:, :, :blend_h, :] *= ramp_h[:blend_h].view(1, 1, -1, 1)
                    if latent_overlap_h > 0 and y_lat_end < H_lat_total:
                        blend_h = min(latent_overlap_h, tile_h_actual)
                        weight[:, :, -blend_h:, :] *= (1.0 - ramp_h[:blend_h]).view(1, 1, -1, 1)
                    
                    if latent_overlap_w > 0 and x_lat > 0:
                        blend_w = min(latent_overlap_w, tile_w_actual)
                        weight[:, :, :, :blend_w] *= ramp_w[:blend_w].view(1, 1, 1, -1)
                    if latent_overlap_w > 0 and x_lat_end < W_lat_total:
                        blend_w = min(latent_overlap_w, tile_w_actual)
                        weight[:, :, :, -blend_w:] *= (1.0 - ramp_w[:blend_w]).view(1, 1, 1, -1)
                    
                    # Accumulate weighted tile
                    result[:, :, y_lat:y_lat+tile_h_actual, x_lat:x_lat+tile_w_actual] += tile_latent * weight
                    count[:, :, y_lat:y_lat+tile_h_actual, x_lat:x_lat+tile_w_actual] += weight
                    
                    # Free memory
                    del tile_latent, weight
                    torch.cuda.empty_cache()
            
            # Normalize by weights
            latent_item = result / count.clamp(min=1e-8)
            batch_latents.append(latent_item)
            
            # Free memory
            del result, count
            torch.cuda.empty_cache()
        
        # Stack batch
        latent = torch.stack(batch_latents, dim=0)
        return WanVAEEncoderOutput(latent=latent, posterior=None)
    
    def _decode_single(self, z: torch.Tensor) -> torch.Tensor:
        """Decode a single latent (no batch dimension) using VAE's internal frame-by-frame processing"""
        # z shape: (Z, T//4, H//8, W//8)
        # Returns: (C, T, H, W)
        batch_list = [z]
        decoded = self.vae.decode(batch_list)
        return decoded[0]
    
    def decode(self, z: torch.Tensor, tiled: bool = False,
               tile_size: Optional[Tuple[int, int]] = None,
               tile_overlap: Optional[Tuple[int, int]] = None) -> WanVAEDecoderOutput:
        """
        Decode latent tensor to pixel space with spatial tiling support (like SeedVR2 VAE)
        
        Args:
            z: Latent tensor of shape (B, Z, T//4, H//8, W//8)
            tiled: Whether to use spatial tiling
            tile_size: Tile size (H, W) in pixel space
            tile_overlap: Overlap size (H, W) in pixel space
            
        Returns:
            WanVAEDecoderOutput with sample tensor of shape (B, C, T, H, W)
        """
        B, Z, T_lat, H_lat, W_lat = z.shape
        
        # If not tiled or small enough, decode directly
        if not tiled or (tile_size and H_lat * 8 <= tile_size[0] and W_lat * 8 <= tile_size[1]):
            batch_list = [z[i] for i in range(B)]
            decoded = self.vae.decode(batch_list)
            sample = torch.stack(decoded, dim=0)
            return WanVAEDecoderOutput(sample=sample)
        
        # Spatial tiling implementation (matching SeedVR2 VAE pattern)
        tile_size = tile_size or (512, 512)
        tile_overlap = tile_overlap or (64, 64)
        
        scale_factor = 8  # Wan2.1 VAE spatial upsampling
        tile_h, tile_w = tile_size
        overlap_h, overlap_w = tile_overlap
        
        # Convert to latent space
        latent_tile_h = max(1, tile_h // scale_factor)
        latent_tile_w = max(1, tile_w // scale_factor)
        latent_overlap_h = max(0, min(overlap_h // scale_factor, latent_tile_h - 1))
        latent_overlap_w = max(0, min(overlap_w // scale_factor, latent_tile_w - 1))
        
        stride_h = max(1, latent_tile_h - latent_overlap_h)
        stride_w = max(1, latent_tile_w - latent_overlap_w)
        
        H_out = H_lat * scale_factor
        W_out = W_lat * scale_factor
        
        # Pre-compute blending ramps
        ramp_h = None
        ramp_w = None
        if overlap_h > 0:
            t_h = torch.linspace(0, 1, steps=overlap_h, device=z.device, dtype=z.dtype)
            ramp_h = 0.5 - 0.5 * torch.cos(t_h * torch.pi)
        if overlap_w > 0:
            t_w = torch.linspace(0, 1, steps=overlap_w, device=z.device, dtype=z.dtype)
            ramp_w = 0.5 - 0.5 * torch.cos(t_w * torch.pi)
        
        # Process each batch item separately
        batch_samples = []
        for b in range(B):
            result = None
            count = None
            
            # Tile the spatial dimensions in latent space
            for y_lat in range(0, H_lat, stride_h):
                y_lat_end = min(y_lat + latent_tile_h, H_lat)
                for x_lat in range(0, W_lat, stride_w):
                    x_lat_end = min(x_lat + latent_tile_w, W_lat)
                    
                    # Skip tiny overlap tiles
                    if (y_lat > 0 and (y_lat_end - y_lat) <= latent_overlap_h) or \
                       (x_lat > 0 and (x_lat_end - x_lat) <= latent_overlap_w):
                        continue
                    
                    # Extract latent tile
                    tile_latent = z[b:b+1, :, :, y_lat:y_lat_end, x_lat:x_lat_end]
                    
                    # Decode tile
                    tile_sample = self._decode_single(tile_latent[0])
                    
                    # Map to pixel space
                    y_out = y_lat * scale_factor
                    x_out = x_lat * scale_factor
                    y_out_end = min(y_lat_end * scale_factor, H_out)
                    x_out_end = min(x_lat_end * scale_factor, W_out)
                    
                    # Initialize result on first tile
                    if result is None:
                        C, T, _, _ = tile_sample.shape
                        result = torch.zeros((C, T, H_out, W_out), 
                                           device=tile_sample.device, dtype=tile_sample.dtype)
                        count = torch.zeros((C, T, H_out, W_out), 
                                          device=tile_sample.device, dtype=tile_sample.dtype)
                    
                    # Compute blending weights
                    tile_h_actual = tile_sample.shape[2]
                    tile_w_actual = tile_sample.shape[3]
                    weight = torch.ones((C, T, tile_h_actual, tile_w_actual), 
                                       device=tile_sample.device, dtype=tile_sample.dtype)
                    
                    # Apply blending ramps in pixel space
                    if overlap_h > 0 and y_out > 0:
                        blend_h = min(overlap_h, tile_h_actual)
                        weight[:, :, :blend_h, :] *= ramp_h[:blend_h].view(1, 1, -1, 1)
                    if overlap_h > 0 and y_out_end < H_out:
                        blend_h = min(overlap_h, tile_h_actual)
                        weight[:, :, -blend_h:, :] *= (1.0 - ramp_h[:blend_h]).view(1, 1, -1, 1)
                    
                    if overlap_w > 0 and x_out > 0:
                        blend_w = min(overlap_w, tile_w_actual)
                        weight[:, :, :, :blend_w] *= ramp_w[:blend_w].view(1, 1, 1, -1)
                    if overlap_w > 0 and x_out_end < W_out:
                        blend_w = min(overlap_w, tile_w_actual)
                        weight[:, :, :, -blend_w:] *= (1.0 - ramp_w[:blend_w]).view(1, 1, 1, -1)
                    
                    # Accumulate weighted tile
                    result[:, :, y_out:y_out+tile_h_actual, x_out:x_out+tile_w_actual] += tile_sample * weight
                    count[:, :, y_out:y_out+tile_h_actual, x_out:x_out+tile_w_actual] += weight
                    
                    # Free memory
                    del tile_sample, weight
                    torch.cuda.empty_cache()
            
            # Normalize by weights
            sample_item = result / count.clamp(min=1e-8)
            batch_samples.append(sample_item)
            
            # Free memory
            del result, count
            torch.cuda.empty_cache()
        
        # Stack batch
        sample = torch.stack(batch_samples, dim=0)
        return WanVAEDecoderOutput(sample=sample)
    
    def clear_cache(self):
        """Clear internal cache"""
        if hasattr(self.vae, 'model') and hasattr(self.vae.model, 'clear_cache'):
            self.vae.model.clear_cache()
    
    def parameters(self):
        """Return model parameters"""
        return self.vae.model.parameters()
