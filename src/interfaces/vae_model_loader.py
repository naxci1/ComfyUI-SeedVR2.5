"""
SeedVR2 VAE Model Loader Node
Configure VAE (Variational Autoencoder) model with tiling support and GGUF loading
"""

from comfy_api.latest import io
from comfy_execution.utils import get_executing_context
from typing import Dict, Any, Tuple
from ..utils.model_registry import get_available_vae_models, DEFAULT_VAE
from ..optimization.memory_manager import get_device_list


class SeedVR2LoadVAEModel(io.ComfyNode):
    """
    Configure VAE (Variational Autoencoder) model loader with tiling support and GGUF loading
    
    Provides configuration for:
    - Model selection and device placement (including GGUF models)
    - Tiled encoding/decoding for VRAM reduction
    - Tile size and overlap control
    - Model caching between runs
    - Optional torch.compile integration
    - Automatic Blackwell sm_120 FP8 optimization for GGUF models
    
    Returns:
        SEEDVR2_VAE configuration dictionary for main upscaler node
    """
    
    @classmethod
    def define_schema(cls) -> io.Schema:        
        devices = get_device_list()
        vae_models = get_available_vae_models()
        
        return io.Schema(
            node_id="SeedVR2LoadVAEModel",
            display_name="SeedVR2 (Down)Load VAE Model",
            category="SEEDVR2",
            description=(
                "Load and configure SeedVR2 VAE (Variational Autoencoder) for encoding/decoding video frames to/from latent space. "
                "Supports tiled processing to handle high resolutions on limited VRAM, model caching, "
                "multi-GPU offloading, and torch.compile acceleration. \n\n"
                "GGUF models are automatically optimized for Blackwell sm_120 with FP8 precision.\n\n"
                "Connect to Video Upscaler node."
            ),
            inputs=[
                io.Combo.Input("vae_name",
                    options=vae_models,
                    default=DEFAULT_VAE,
                    tooltip=(
                        "VAE (Variational Autoencoder) model for encoding/decoding.\n"
                        "Models automatically download on first use.\n"
                        "GGUF models (.gguf extension) are automatically optimized for Blackwell (sm_120) with FP8.\n"
                        "Additional models can be added to the ComfyUI models/SEEDVR2 folder."
                    )
                ),
                io.Combo.Input("device",
                    options=devices,
                    default=devices[0],
                    tooltip="GPU device for VAE model inference (encoding/decoding phases)"
                ),
                io.Boolean.Input("enable_blackwell_optimization",
                    default=True,
                    optional=True,
                    tooltip=(
                        "Enable Blackwell sm_120 optimizations for GGUF models:\n"
                        "• FP8 native inference (torch.float8_e4m3fn)\n"
                        "• Smart dynamic tiling (13GB VRAM target for 16GB cards)\n"
                        "• Channels Last 3D memory format\n"
                        "• CUDA Graph capture for Windows latency fix\n"
                        "• Flash Attention integration\n"
                        "\n"
                        "Only applies to GGUF (.gguf) models. Ignored for safetensors."
                    )
                ),
                io.Boolean.Input("encode_tiled",
                    default=False,
                    optional=True,
                    tooltip="Enable tiled encoding to reduce VRAM usage during the encoding phase"
                ),
                io.Int.Input("encode_tile_size",
                    default=1024,
                    min=64,
                    step=32,
                    optional=True,
                    tooltip=(
                        "Encoding tile size in pixels (default: 1024).\n"
                        "Applied to both height and width.\n"
                        "Lower values reduce VRAM usage but may increase processing time.\n"
                        "Only used when encode_tiled is enabled."
                    )
                ),
                io.Int.Input("encode_tile_overlap",
                    default=128,
                    min=0,
                    step=32,
                    optional=True,
                    tooltip=(
                        "Pixel overlap between encoding tiles (default: 128).\n"
                        "Reduces visible seams between tiles through blending.\n"
                        "Higher values improve quality but slow processing.\n"
                        "Only used when encode_tiled is enabled."
                    )
                ),
                io.Boolean.Input("decode_tiled",
                    default=False,
                    optional=True,
                    tooltip="Enable tiled decoding to reduce VRAM usage during the decoding phase"
                ),
                io.Int.Input("decode_tile_size",
                    default=1024,
                    min=64,
                    step=32,
                    optional=True,
                    tooltip=(
                        "Decoding tile size in pixels (default: 1024).\n"
                        "Applied to both height and width.\n"
                        "Lower values reduce VRAM usage but may increase processing time.\n"
                        "Only used when decode_tiled is enabled."
                    )
                ),
                io.Int.Input("decode_tile_overlap",
                    default=128,
                    min=0,
                    step=32,
                    optional=True,
                    tooltip=(
                        "Pixel overlap between decoding tiles (default: 128).\n"
                        "Reduces visible seams between tiles through blending.\n"
                        "Higher values improve quality but slow processing.\n"
                        "Only used when decode_tiled is enabled."
                    )
                ),
                io.Combo.Input("tile_debug",
                    options=["false", "encode", "decode"],
                    default="false",
                    optional=True,
                    tooltip=(
                        "Tile debug visualization mode:\n"
                        "• 'false': No visualization overlay (default)\n"
                        "• 'encode': Show encoding tile boundaries\n"
                        "• 'decode': Show decoding tile boundaries\n"
                        "\n"
                        "Only works when respective tiling is enabled."
                    )
                ),
                io.Combo.Input("offload_device",
                    options=get_device_list(include_none=True, include_cpu=True),
                    default="none",
                    optional=True,
                    tooltip=(
                        "Device to offload VAE model when not actively processing.\n"
                        "• 'none': Keep model on inference device (default, fastest)\n"
                        "• 'cpu': Offload to system RAM (reduces VRAM usage)\n"
                        "• 'cuda:X': Offload to another GPU (good balance if available)"
                    )
                ),
                io.Boolean.Input("cache_model",
                    default=False,
                    optional=True,
                    tooltip=(
                        "Keep VAE model loaded on offload_device between workflow runs.\n"
                        "Useful for batch processing to avoid repeated loading.\n"
                        "Requires offload_device to be set."
                    )
                ),
                io.Custom("TORCH_COMPILE_ARGS").Input("torch_compile_args",
                    optional=True,
                    tooltip=(
                        "Optional torch.compile optimization settings from SeedVR2 Torch Compile Settings node.\n"
                        "Provides 15-25% speedup with compatible PyTorch 2.0+ and Triton installation."
                    )
                ),
            ],
            outputs=[
                io.Custom("SEEDVR2_VAE").Output(
                    tooltip="VAE model configuration containing model path, device settings, tiling parameters, and compilation options. Connect to Video Upscaler node."
                )
            ]
        )
    
    @classmethod
    def execute(cls, vae_name: str, device: str, enable_blackwell_optimization: bool = True,
                     offload_device: str = "none", cache_model: bool = False, 
                     encode_tiled: bool = False, encode_tile_size: int = 512, 
                     encode_tile_overlap: int = 64, decode_tiled: bool = False, 
                     decode_tile_size: int = 512, decode_tile_overlap: int = 64, 
                     tile_debug: str = "false", torch_compile_args: Dict[str, Any] = None
                     ) -> io.NodeOutput:
        """
        Create VAE model configuration for SeedVR2 main node with GGUF support
        
        Args:
            vae_name: Model filename to load (supports .safetensors and .gguf)
            device: Target device for model execution
            enable_blackwell_optimization: Enable Blackwell sm_120 FP8 optimizations for GGUF
            offload_device: Device to offload model to when not in use
            cache_model: Whether to keep model loaded between runs
            encode_tiled: Enable tiled encoding
            encode_tile_size: Tile size for encoding
            encode_tile_overlap: Tile overlap for encoding
            decode_tiled: Enable tiled decoding
            decode_tile_size: Tile size for decoding
            decode_tile_overlap: Tile overlap for decoding
            tile_debug: Tile visualization mode (false/encode/decode)
            torch_compile_args: Optional torch.compile configuration from settings node
            
        Returns:
            NodeOutput containing configuration dictionary for SeedVR2 main node
            
        Raises:
            ValueError: If cache_model is enabled but offload_device is invalid
        """
        # Validate cache_model configuration
        if cache_model and offload_device == "none":
            raise ValueError(
                "Model caching (cache_model=True) requires offload_device to be set. "
                f"Current: offload_device='{offload_device}'. "
                "Please set offload_device to specify where the cached VAE model should be stored "
                "(e.g., 'cpu' or another device). Set cache_model=False if you don't want to cache the model."
            )
        
        # Detect if this is a GGUF model
        is_gguf = vae_name.endswith('.gguf')
        
        config = {
            "model": vae_name,
            "device": device,
            "offload_device": offload_device,
            "cache_model": cache_model,
            "encode_tiled": encode_tiled,
            "encode_tile_size": encode_tile_size,
            "encode_tile_overlap": encode_tile_overlap,
            "decode_tiled": decode_tiled,
            "decode_tile_size": decode_tile_size,
            "decode_tile_overlap": decode_tile_overlap,
            "tile_debug": tile_debug,
            "torch_compile_args": torch_compile_args,
            "node_id": get_executing_context().node_id,
            # GGUF-specific configuration
            "is_gguf": is_gguf,
            "enable_blackwell_optimization": enable_blackwell_optimization and is_gguf,
            "blackwell_vram_gb": 16.0,  # Target RTX 5070 Ti with 16GB VRAM
            "blackwell_target_vram_usage": 0.8125,  # 13GB of 16GB (81.25%)
        }
        
        # Log GGUF detection and optimization status
        if is_gguf:
            import logging
            logger = logging.getLogger(__name__)
            if enable_blackwell_optimization:
                logger.info(f"[BLACKWELL_ENGINE] GGUF VAE detected: {vae_name}")
                logger.info(f"[BLACKWELL_ENGINE] Blackwell sm_120 optimizations will be applied on load:")
                logger.info(f"[BLACKWELL_ENGINE]   - FP8 native inference (torch.float8_e4m3fn)")
                logger.info(f"[BLACKWELL_ENGINE]   - Smart dynamic tiling (13GB VRAM target)")
                logger.info(f"[BLACKWELL_ENGINE]   - Channels Last 3D memory format")
                logger.info(f"[BLACKWELL_ENGINE]   - CUDA Graph capture for Windows")
            else:
                logger.info(f"[BLACKWELL_ENGINE] GGUF VAE detected but optimization disabled: {vae_name}")
        
        return io.NodeOutput(config)