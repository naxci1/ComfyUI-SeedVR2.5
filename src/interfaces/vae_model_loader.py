"""
SeedVR2 VAE Model Loader Node
Configure VAE (Variational Autoencoder) model with tiling support and GGUF loading
"""

from comfy_api.latest import io
from comfy_execution.utils import get_executing_context
from typing import Dict, Any, Tuple, Optional
import torch
import os
from ..utils.model_registry import get_available_vae_models, DEFAULT_VAE, get_model_repo
from ..utils.constants import find_model_file
from ..optimization.memory_manager import get_device_list
from ..models.video_vae_v3.modules.attn_video_vae import VideoAutoencoderKLWrapper


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
                    default=512,
                    min=32,
                    step=32,
                    optional=True,
                    tooltip=(
                        "Encoding tile size in pixels (default: 512).\n"
                        "Applied to both height and width.\n"
                        "Lower values reduce VRAM usage but may increase processing time.\n"
                        "Only used when encode_tiled is enabled."
                    )
                ),
                io.Int.Input("encode_tile_overlap",
                    default=64,
                    min=0,
                    step=32,
                    optional=True,
                    tooltip=(
                        "Pixel overlap between encoding tiles (default: 64).\n"
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
                    default=512,
                    min=32,
                    step=32,
                    optional=True,
                    tooltip=(
                        "Decoding tile size in pixels (default: 512).\n"
                        "Applied to both height and width.\n"
                        "Lower values reduce VRAM usage but may increase processing time.\n"
                        "Only used when decode_tiled is enabled."
                    )
                ),
                io.Int.Input("decode_tile_overlap",
                    default=64,
                    min=0,
                    step=32,
                    optional=True,
                    tooltip=(
                        "Pixel overlap between decoding tiles (default: 64).\n"
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
                     encode_tiled: bool = False, encode_tile_size: int = 512, 
                     encode_tile_overlap: int = 64, decode_tiled: bool = False, 
                     decode_tile_size: int = 512, decode_tile_overlap: int = 64, 
                     tile_debug: str = "false", offload_device: str = "none", 
                     cache_model: bool = False, torch_compile_args: Dict[str, Any] = None
                     ) -> io.NodeOutput:
        """
        Load VAE model and apply Blackwell optimizations immediately
        
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
            NodeOutput containing loaded and optimized VAE model with configuration
            
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
        
        # LOAD THE VAE MODEL NOW (not later)
        vae_model = cls._load_vae_model(vae_name, device, is_gguf)
        
        # FORCE BLACKWELL OPTIMIZATION IMMEDIATELY
        if is_gguf and enable_blackwell_optimization:
            print("\n" + "="*60)
            print("[BLACKWELL_ENGINE] !!! FORCING SM_120 FP8 OPTIMIZATION !!!")
            print("="*60)
            
            # Enable Blackwell backends
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
            torch.backends.cudnn.benchmark = True
            print("[BLACKWELL_ENGINE] ✓ TF32 and cuDNN benchmark enabled")
            
            # Convert weights to FP8 for Blackwell sm_120
            print("[BLACKWELL_ENGINE] Converting weights to FP8 (torch.float8_e4m3fn)...")
            param_count = 0
            for param in vae_model.parameters():
                if param.dtype in [torch.float16, torch.float32, torch.bfloat16]:
                    param.data = param.data.to(torch.float8_e4m3fn)
                    param_count += 1
            print(f"[BLACKWELL_ENGINE] ✓ Converted {param_count} parameters to FP8")
            
            print("[BLACKWELL_ENGINE] Dynamic tiling: PRESERVED (user-controlled)")
            print("="*60 + "\n")
        
        config = {
            "model": vae_model,  # Return the actual loaded model, not just name
            "model_name": vae_name,
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
            "is_gguf": is_gguf,
            "is_optimized": is_gguf and enable_blackwell_optimization,
        }
        
        return io.NodeOutput(config)
    
    @classmethod
    def _load_vae_model(cls, vae_name: str, device: str, is_gguf: bool) -> Any:
        """
        Load VAE model from file
        
        Args:
            vae_name: Model filename
            device: Target device
            is_gguf: Whether this is a GGUF model
            
        Returns:
            Loaded VAE model
        """
        # Get model path
        model_path = find_model_file(vae_name)
        
        # Check if file exists
        if not os.path.exists(model_path):
            # Try to download from HuggingFace
            repo = get_model_repo(vae_name)
            print(f"[VAE Loader] Downloading {vae_name} from {repo}...")
            # This will download automatically via VideoAutoencoderKLWrapper
        
        print(f"[VAE Loader] Loading VAE model: {vae_name}")
        print(f"[VAE Loader] Path: {model_path}")
        print(f"[VAE Loader] Device: {device}")
        print(f"[VAE Loader] Format: {'GGUF' if is_gguf else 'SafeTensors'}")
        
        # Load the model
        try:
            if is_gguf:
                # GGUF: Load as binary file using safetensors
                print(f"[VAE Loader] Loading GGUF as binary file...")
                from safetensors.torch import load_file
                
                # Create model instance first
                vae = VideoAutoencoderKLWrapper.from_pretrained(
                    get_model_repo(vae_name),
                    subfolder="vae",
                    torch_dtype=torch.bfloat16,
                )
                
                # Load GGUF weights as binary
                if os.path.exists(model_path):
                    state_dict = load_file(model_path)
                    vae.load_state_dict(state_dict, strict=False)
                    print(f"[VAE Loader] ✓ Loaded {len(state_dict)} tensors from GGUF file")
            else:
                # Regular safetensors: Use standard loading
                vae = VideoAutoencoderKLWrapper.from_pretrained(
                    model_path if os.path.exists(model_path) else get_model_repo(vae_name),
                    subfolder="vae" if not os.path.exists(model_path) else None,
                    torch_dtype=torch.bfloat16,
                )
            
            # Move to device
            if device != "cpu":
                vae = vae.to(device)
            
            print(f"[VAE Loader] ✓ Model loaded successfully")
            return vae
            
        except Exception as e:
            print(f"[VAE Loader] ✗ Error loading model: {e}")
            raise