"""
SeedVR2 VAE Model Loader Node
Configure VAE (Variational Autoencoder) model with tiling support and Blackwell optimization
"""

from comfy_api.latest import io
from comfy_execution.utils import get_executing_context
from typing import Dict, Any, Tuple
from ..utils.model_registry import get_available_vae_models, DEFAULT_VAE
from ..optimization.memory_manager import get_device_list


class SeedVR2LoadVAEModel(io.ComfyNode):
    """
    Configure VAE (Variational Autoencoder) model loader with tiling support
    
    Provides configuration for:
    - Model selection and device placement
    - Tiled encoding/decoding for VRAM reduction
    - Tile size and overlap control
    - Model caching between runs
    - Optional torch.compile integration
    - Blackwell optimization with Sparge attention (RTX 50xx)
    
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
                "multi-GPU offloading, Blackwell GPU optimization (RTX 50xx), and torch.compile acceleration. \n\n"
                "Connect to Video Upscaler node."
            ),
            inputs=[
                io.Combo.Input("model",
                    options=vae_models,
                    default=DEFAULT_VAE,
                    tooltip=(
                        "VAE (Variational Autoencoder) model for encoding/decoding.\n"
                        "Models automatically download on first use.\n"
                        "Additional models can be added to the ComfyUI models folder.\n"
                        "\n"
                        "FP8 models (e.g., ema_vae_fp8.safetensors) are automatically\n"
                        "detected and loaded in native FP8 format for Blackwell GPUs."
                    )
                ),
                io.Combo.Input("device",
                    options=devices,
                    default=devices[0],
                    tooltip="GPU device for VAE model inference (encoding/decoding phases)"
                ),
                io.Boolean.Input("encode_tiled",
                    default=True,
                    optional=True,
                    tooltip=(
                        "Enable tiled encoding to reduce VRAM usage during the encoding phase.\n"
                        "IMPORTANT: Enable this for 16GB GPUs (Blackwell RTX 50xx) to prevent OOM.\n"
                        "Required for large videos (e.g., 81 frames at 960x720).\n"
                        "Default: True for memory safety."
                    )
                ),
                io.Int.Input("encode_tile_size",
                    default=512,
                    min=64,
                    step=32,
                    optional=True,
                    tooltip=(
                        "Encoding tile size in pixels (default: 512).\n"
                        "Applied to both height and width.\n"
                        "Lower values reduce VRAM usage but may increase processing time.\n"
                        "Recommended: 512 for 16GB VRAM GPUs (Blackwell), 736 for 24GB+.\n"
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
                    default=True,
                    optional=True,
                    tooltip=(
                        "Enable tiled decoding to reduce VRAM usage during the decoding phase.\n"
                        "IMPORTANT: Enable this for 16GB GPUs (Blackwell RTX 50xx) to prevent OOM.\n"
                        "Required for large videos (e.g., 81 frames at 960x720).\n"
                        "Default: True for memory safety."
                    )
                ),
                io.Int.Input("decode_tile_size",
                    default=512,
                    min=64,
                    step=32,
                    optional=True,
                    tooltip=(
                        "Decoding tile size in pixels (default: 512).\n"
                        "Applied to both height and width.\n"
                        "Lower values reduce VRAM usage but may increase processing time.\n"
                        "Recommended: 512 for 16GB VRAM GPUs (Blackwell), 736 for 24GB+.\n"
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
                io.Boolean.Input("enable_sparge_attention",
                    default=True,
                    optional=True,
                    tooltip=(
                        "Enable Sparge + SageAttention 2 for VAE (Blackwell optimized).\n"
                        "\n"
                        "• Dynamic head splitting: 512-dim → 8 heads × 64 dim in forward pass\n"
                        "• Preserves original model weights while enabling Sparge/SageAttn\n"
                        "• Sparge Top-K selection (threshold=0.3) zeros out unimportant pixels\n"
                        "• SageAttention 2 with INT8/FP8 for Blackwell Tensor Cores\n"
                        "• torch.cuda.synchronize() after each tile to prevent scheduler hang\n"
                        "\n"
                        "RECOMMENDED: Keep enabled (True) for RTX 50xx series GPUs.\n"
                        "Falls back to sliced SDPA if Sparge kernel fails."
                    )
                ),
                io.Combo.Input("performance_mode",
                    options=["Fast", "Balanced", "High Quality"],
                    default="Fast",
                    optional=True,
                    tooltip=(
                        "Performance tuning mode for Sparge attention (Blackwell GPUs only).\n"
                        "Controls the sparsity threshold for VAE attention blocks:\n"
                        "\n"
                        "• Fast: Maximum speed, sparsity threshold 0.3 (default for 16GB GPUs)\n"
                        "  Zeros out 70% of attention weights to save massive VRAM\n"
                        "• Balanced: Speed/quality balance, sparsity threshold 0.5\n"
                        "• High Quality: Best quality, sparsity threshold 0.7\n"
                        "\n"
                        "This setting only affects VAE when enable_sparge_attention is True."
                    )
                ),
                io.Combo.Input("vae_precision",
                    options=["auto", "fp16", "bf16", "fp8_e4m3fn"],
                    default="bf16",
                    optional=True,
                    tooltip=(
                        "VAE model precision override:\n"
                        "\n"
                        "• auto: Detect precision from filename\n"
                        "  - Files with 'fp8' or 'e4m3fn' → FP8 loading\n"
                        "  - Other files → Use compute dtype (fp16/bf16)\n"
                        "• fp16: Force FP16 (16-bit floating point)\n"
                        "• bf16: Force BF16 (Brain Float 16) - DEFAULT for Blackwell\n"
                        "  - More memory-efficient on Blackwell GPUs\n"
                        "  - Avoids 'forced to FP16' overhead\n"
                        "• fp8_e4m3fn: Force FP8 E4M3 format (8-bit float)\n"
                        "  - E4M3FN = 4 exponent bits, 3 mantissa bits, no infinity\n"
                        "  - Optimized for RTX 50xx (Blackwell) Tensor Cores\n"
                        "  - Also works on RTX 40xx (Ada) with reduced precision\n"
                        "\n"
                        "RECOMMENDED: Use 'bf16' for best stability on RTX 50xx."
                    )
                ),
                io.Custom("TORCH_COMPILE_ARGS").Input("torch_compile_args",
                    optional=True,
                    tooltip=(
                        "Optional torch.compile optimization settings from SeedVR2 Torch Compile Settings node.\n"
                        "Provides 15-25% speedup with compatible PyTorch 2.0+ and Triton installation.\n"
                        "\n"
                        "Note: torch.compile may be unstable on Windows. Consider disabling if you experience issues."
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
    def execute(cls, model: str, device: str, offload_device: str = "none",
                     cache_model: bool = False, encode_tiled: bool = True,
                     encode_tile_size: int = 512, encode_tile_overlap: int = 64,
                     decode_tiled: bool = True, decode_tile_size: int = 512, 
                     decode_tile_overlap: int = 64, tile_debug: str = "false",
                     enable_sparge_attention: bool = True, performance_mode: str = "Balanced",
                     vae_precision: str = "auto", torch_compile_args: Dict[str, Any] = None
                     ) -> io.NodeOutput:
        """
        Create VAE model configuration for SeedVR2 main node
        
        Args:
            model: Model filename to load
            device: Target device for model execution
            offload_device: Device to offload model to when not in use
            cache_model: Whether to keep model loaded between runs
            encode_tiled: Enable tiled encoding
            encode_tile_size: Tile size for encoding
            encode_tile_overlap: Tile overlap for encoding
            decode_tiled: Enable tiled decoding
            decode_tile_size: Tile size for decoding
            decode_tile_overlap: Tile overlap for decoding
            tile_debug: Tile visualization mode (false/encode/decode)
            enable_sparge_attention: Enable Sparge block-sparse attention for VAE
            performance_mode: Performance tuning for Sparge ('Fast', 'Balanced', 'High Quality')
            vae_precision: Precision override ('auto', 'fp16', 'bf16', 'fp8_e4m3fn')
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
        
        # Lazy import to avoid loading torch at module level
        from ..optimization.compatibility import BLACKWELL_GPU_DETECTED
        from ..optimization.vae_attention import is_vae_sparge_available
        
        # Check if Sparge attention is actually available
        sparge_available = is_vae_sparge_available() and BLACKWELL_GPU_DETECTED
        sparge_active = enable_sparge_attention and sparge_available
        
        # Map performance_mode to sparsity_threshold for VAE Sparge attention
        # Uses same mapping as DiT for consistency
        performance_mode_map = {
            "Fast": 0.3,        # Maximum speed, 30% attention weights kept
            "Balanced": 0.5,    # Optimal speed/quality balance (default)
            "High Quality": 0.7 # Best quality, 70% attention weights kept
        }
        vae_sparsity_threshold = performance_mode_map.get(performance_mode, 0.5)
        
        # Determine FP8 model status based on precision setting
        # Priority: manual precision override > filename detection
        if vae_precision == "fp8_e4m3fn":
            # Force FP8 path regardless of filename
            is_fp8_model = True
        elif vae_precision in ("fp16", "bf16"):
            # Force non-FP8 path regardless of filename
            is_fp8_model = False
        else:
            # Auto-detect from filename
            is_fp8_model = "fp8" in model.lower() or "e4m3fn" in model.lower()
        
        config = {
            "model": model,
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
            # Blackwell optimization settings
            "enable_sparge_attention": sparge_active,
            "vae_sparsity_threshold": vae_sparsity_threshold,
            "performance_mode": performance_mode,
            "vae_precision": vae_precision,
            "is_fp8_model": is_fp8_model,
            "blackwell_detected": BLACKWELL_GPU_DETECTED,
        }
        return io.NodeOutput(config)