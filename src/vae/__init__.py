"""VAE modules for ComfyUI-SeedVR2.5"""

from .wan2_1_wrapper import (
    Wan21VAEWrapper, 
    wrap_vae_if_wan21, 
    calculate_valid_temporal_size,
    load_vae_config_json,
)
from .vae_config import (
    VAEModelVersion,
    VAEEncodingType,
    VAEArchitectureConfig,
    VAEEncodingConfig,
    VAEModelConfig,
    VAEConfigManager,
    get_vae_config_manager,
    get_wan21_config,
    get_wan22_config,
)

__all__ = [
    "Wan21VAEWrapper",
    "wrap_vae_if_wan21",
    "calculate_valid_temporal_size",
    "load_vae_config_json",
    "VAEModelVersion",
    "VAEEncodingType",
    "VAEArchitectureConfig",
    "VAEEncodingConfig",
    "VAEModelConfig",
    "VAEConfigManager",
    "get_vae_config_manager",
    "get_wan21_config",
    "get_wan22_config",
]
