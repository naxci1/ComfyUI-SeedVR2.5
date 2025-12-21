"""VAE modules for ComfyUI-SeedVR2.5"""

from .wan2_1_vae import Wan2_1_VAE, create_wan2_1_vae  # noqa: F401
from .wan2_2_vae import Wan2_2_VAE, create_wan2_2_vae  # noqa: F401
from .vae_config import (  # noqa: F401
    VAEConfigManager,
    VAEModelConfig,
    get_vae_config_manager,
    get_wan21_config,
    get_wan22_config,
)
from .wan22_handler import (  # noqa: F401
    Wan22CausalVAE3D,
    Wan22TilingConfig,
    Wan22TilingMode,
    Wan22PrecisionMode,
    create_wan22_causal_vae,
    load_wan22_vae_gguf,
)

__all__ = [
    # Wan2.1 VAE
    "Wan2_1_VAE",
    "create_wan2_1_vae",
    # Wan2.2 Basic VAE
    "Wan2_2_VAE",
    "create_wan2_2_vae",
    # Wan2.2 3D Causal VAE (new)
    "Wan22CausalVAE3D",
    "Wan22TilingConfig",
    "Wan22TilingMode",
    "Wan22PrecisionMode",
    "create_wan22_causal_vae",
    "load_wan22_vae_gguf",
    # Configuration
    "VAEConfigManager",
    "VAEModelConfig",
    "get_vae_config_manager",
    "get_wan21_config",
    "get_wan22_config",
]
