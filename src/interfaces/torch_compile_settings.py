"""
SeedVR2 Torch Compile Settings Node
Configure torch.compile optimization for DiT and VAE models
"""

from comfy_api.latest import io
from typing import Dict, Any, Tuple


class SeedVR2TorchCompileSettings(io.ComfyNode):
    """Configure torch.compile optimization for DiT and VAE models"""
    
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SeedVR2TorchCompileSettings",
            display_name="SeedVR2 Torch Compile Settings",
            category="SEEDVR2",
            description=(
                "Configure SeedVR2 torch.compile optimization for 20-40% DiT speedup and 15-25% VAE speedup. "
                "Trades longer first-run compilation time for faster inference.\n\n"
                "Connect to DiT and/or VAE model loaders. Requires PyTorch 2.0+ and Triton for inductor backend."
            ),
            inputs=[
                io.Combo.Input("backend", 
                    options=["inductor", "cudagraphs"],
                    default="inductor",
                    tooltip=(
                        "Compilation backend:\n"
                        "• inductor: Full optimization with Triton kernel generation and fusion (recommended)\n"
                        "• cudagraphs: Lightweight wrapper using CUDA graphs, no kernel optimization"
                    )
                ),
                io.Combo.Input("mode",
                    options=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
                    default="default",
                    tooltip=(
                        "Optimization level (compilation time vs runtime performance):\n"
                        "• default: Fast compilation with good speedup (recommended for development)\n"
                        "• reduce-overhead: Lower overhead, optimized for smaller models\n"
                        "• max-autotune: Slowest compilation, best runtime performance (recommended for production)\n"
                        "• max-autotune-no-cudagraphs: Like max-autotune but without CUDA graphs"
                    )
                ),
                io.Boolean.Input("fullgraph",
                    default=False,
                    tooltip=(
                        "Compile entire model as single graph without breaks.\n"
                        "• False: Allow graph breaks for better compatibility (default)\n"
                        "• True: Enforce no breaks for maximum optimization (may fail with dynamic shapes)"
                    )
                ),
                io.Boolean.Input("dynamic",
                    default=False,
                    tooltip=(
                        "Handle varying input shapes without recompilation.\n"
                        "• False: Specialize for exact input shapes (default)\n"
                        "• True: Create dynamic kernels that adapt to shape variations\n"
                        "\n"
                        "Enable when processing different resolutions or batch sizes."
                    )
                ),
                io.Int.Input("dynamo_cache_size_limit",
                    default=64,
                    min=0,
                    max=1024,
                    step=1,
                    tooltip=(
                        "Maximum cached compiled versions per function (default: 64).\n"
                        "Controls how many shape variations to compile before stopping.\n"
                        "\n"
                        "• Increase: When processing many different input shapes (more memory usage)\n"
                        "• Decrease: When recompilation cost outweighs benefits (faster fallback to eager)"
                    )
                ),
                io.Int.Input("dynamo_recompile_limit",
                    default=128,
                    min=0,
                    max=1024,
                    step=1,
                    tooltip=(
                        "Maximum recompilation attempts before fallback to eager mode (default: 128).\n"
                        "Safety limit to prevent infinite compilation loops.\n"
                        "\n"
                        "Only increase if you see 'hit config.recompile_limit' warnings and have bounded shape variations."
                    )
                ),
            ],
            outputs=[
                io.Custom("TORCH_COMPILE_ARGS").Output(
                    tooltip="torch.compile optimization settings including backend, mode, and Dynamo configuration. Connect to DiT and/or VAE model loader nodes."
                )
            ]
        )
    
    @classmethod
    def execute(cls, backend: str, mode: str, fullgraph: bool, dynamic: bool, 
                   dynamo_cache_size_limit: int, dynamo_recompile_limit: int) -> io.NodeOutput:
        """
        Create torch.compile configuration for model optimization
        
        Args:
            backend: Compilation backend ("inductor" or "cudagraphs")
            mode: Optimization mode ("default", "reduce-overhead", "max-autotune", etc.)
            fullgraph: Whether to compile entire model as single graph
            dynamic: Whether to handle varying input shapes without recompilation
            dynamo_cache_size_limit: Maximum cached compiled versions per function
            dynamo_recompile_limit: Maximum recompilation attempts before fallback
            
        Returns:
            NodeOutput containing torch.compile configuration dictionary
        """
        import platform
        import torch
        
        # Windows-specific inductor configuration
        # These settings improve compatibility with Triton-Windows
        is_windows = platform.system() == 'Windows'
        
        # Configure inductor settings for Windows + Triton-Windows compatibility
        if is_windows and backend == "inductor":
            try:
                import torch._inductor.config as inductor_config
                
                # Windows-optimized inductor settings:
                # cpp_wrapper=True: Use C++ wrapper instead of Python for reduced overhead
                inductor_config.cpp_wrapper = True
                
                # fallback_random=True: Use PyTorch random instead of Triton random
                # This avoids Windows-specific issues with Triton random generation
                inductor_config.fallback_random = True
                
                # disable_progress=True: Avoid Windows console issues with progress bars
                if hasattr(inductor_config, 'disable_progress'):
                    inductor_config.disable_progress = True
                
                # triton.cudagraphs=False: Disable CUDA graphs in Triton to avoid
                # Windows WDDM driver issues with graph capture
                if hasattr(inductor_config, 'triton'):
                    inductor_config.triton.cudagraphs = False
                
                # coordinate_descent_tuning=False: Disable for faster compilation
                # (tuning can be slow on Windows due to subprocess overhead)
                if hasattr(inductor_config, 'coordinate_descent_tuning'):
                    inductor_config.coordinate_descent_tuning = False
                    
                print("🔧 Windows torch.compile optimizations applied: cpp_wrapper=True, fallback_random=True")
                
            except (ImportError, AttributeError) as e:
                print(f"⚠️ Could not apply Windows inductor optimizations: {e}")
        
        # Apply Dynamo configuration
        try:
            torch._dynamo.config.cache_size_limit = dynamo_cache_size_limit
            torch._dynamo.config.accumulated_cache_size_limit = dynamo_recompile_limit
            
            # Windows-specific Dynamo settings
            if is_windows:
                # Suppress excessive recompilation warnings on Windows
                torch._dynamo.config.verbose = False
                torch._dynamo.config.suppress_errors = True
                
        except (ImportError, AttributeError):
            pass
        
        compile_args = {
            "backend": backend,
            "mode": mode,
            "fullgraph": fullgraph,
            "dynamic": dynamic,
            "dynamo_cache_size_limit": dynamo_cache_size_limit,
            "dynamo_recompile_limit": dynamo_recompile_limit,
            "is_windows": is_windows,
        }
        return io.NodeOutput(compile_args)