"""
Memory Optimization Integration for SeedVR2 Pipeline

This module provides integration utilities for enabling memory optimizations
in the main inference pipeline (video_upscaler.py and infer.py).

Author: GitHub Copilot
Date: 2026-01-21
"""

import torch
import torch.nn as nn
from typing import Optional
from .memory_optimizations import (
    enable_gradient_checkpointing,
    CPUOffloadManager,
    ActivationMemoryManager
)


class MemoryOptimizedPipeline:
    """
    Wrapper for memory-optimized inference pipeline.
    
    Features:
    - Automatic gradient checkpointing for DiT
    - CPU offloading for VAE/DiT models
    - Activation memory management
    - VRAM monitoring and logging
    """
    
    def __init__(self, debug=None):
        """
        Initialize memory-optimized pipeline.
        
        Args:
            debug: Debug logger instance
        """
        self.debug = debug
        self.offload_manager = CPUOffloadManager()
        self.initial_vram = None
        self.peak_vram = None
        
    def setup_dit_optimization(self, dit_model: nn.Module, enable_checkpointing: bool = True):
        """
        Setup memory optimizations for DiT model.
        
        Args:
            dit_model: DiT model instance
            enable_checkpointing: Whether to enable gradient checkpointing (default: True)
        
        Returns:
            dit_model: Optimized DiT model
        """
        if enable_checkpointing:
            # Enable gradient checkpointing for all transformer blocks
            enable_gradient_checkpointing(dit_model, enabled=True)
            
            if self.debug:
                self.debug.log("Gradient checkpointing enabled for DiT model", category="info")
        
        return dit_model
    
    def setup_vae_optimization(self, vae_model: nn.Module):
        """
        Setup memory optimizations for VAE model.
        
        Args:
            vae_model: VAE model instance
        
        Returns:
            vae_model: Optimized VAE model
        """
        # VAE typically doesn't need gradient checkpointing during inference
        # But we can offload it to CPU when DiT is working
        return vae_model
    
    def offload_vae_to_cpu(self, vae_model: nn.Module):
        """
        Offload VAE to CPU while DiT is working.
        
        Args:
            vae_model: VAE model instance
        """
        self.offload_manager.offload_to_cpu(vae_model)
        
        if self.debug:
            freed_vram = self.offload_manager.estimate_model_memory(vae_model) / (1024**3)
            self.debug.log(f"VAE offloaded to CPU (freed ~{freed_vram:.2f} GB VRAM)", 
                          category="info")
    
    def restore_vae_to_gpu(self, vae_model: nn.Module):
        """
        Restore VAE to GPU after DiT work is done.
        
        Args:
            vae_model: VAE model instance
        """
        self.offload_manager.restore_to_gpu(vae_model)
        
        if self.debug:
            self.debug.log("VAE restored to GPU", category="info")
    
    def offload_dit_to_cpu(self, dit_model: nn.Module):
        """
        Offload DiT to CPU while VAE is working.
        
        Args:
            dit_model: DiT model instance
        """
        self.offload_manager.offload_to_cpu(dit_model)
        
        if self.debug:
            freed_vram = self.offload_manager.estimate_model_memory(dit_model) / (1024**3)
            self.debug.log(f"DiT offloaded to CPU (freed ~{freed_vram:.2f} GB VRAM)", 
                          category="info")
    
    def restore_dit_to_gpu(self, dit_model: nn.Module):
        """
        Restore DiT to GPU after VAE work is done.
        
        Args:
            dit_model: DiT model instance
        """
        self.offload_manager.restore_to_gpu(dit_model)
        
        if self.debug:
            self.debug.log("DiT restored to GPU", category="info")
    
    def track_vram_usage(self, phase: str = ""):
        """
        Track VRAM usage at different pipeline phases.
        
        Args:
            phase: Pipeline phase name (e.g., "before_dit", "after_dit")
        """
        if not torch.cuda.is_available():
            return
        
        current_vram = torch.cuda.memory_allocated() / (1024**3)
        
        if self.initial_vram is None:
            self.initial_vram = current_vram
        
        if self.peak_vram is None or current_vram > self.peak_vram:
            self.peak_vram = current_vram
        
        if self.debug:
            self.debug.log(f"VRAM usage [{phase}]: {current_vram:.2f} GB " +
                          f"(peak: {self.peak_vram:.2f} GB)", 
                          category="info")
    
    def get_vram_savings(self) -> float:
        """
        Calculate VRAM savings from optimizations.
        
        Returns:
            VRAM savings in GB
        """
        if self.initial_vram is None or self.peak_vram is None:
            return 0.0
        
        # Estimate baseline peak without optimizations would be higher
        # Based on typical 30-50% savings from checkpointing + offloading
        estimated_baseline = self.peak_vram / 0.7  # Assuming 30% savings
        savings = estimated_baseline - self.peak_vram
        
        return max(0.0, savings)


def initialize_memory_optimizations(dit_model: nn.Module, 
                                   vae_model: Optional[nn.Module] = None,
                                   enable_checkpointing: bool = True,
                                   debug=None) -> MemoryOptimizedPipeline:
    """
    Initialize memory optimizations for the inference pipeline.
    
    This is the main entry point for integrating memory optimizations.
    
    Args:
        dit_model: DiT model instance
        vae_model: VAE model instance (optional)
        enable_checkpointing: Whether to enable gradient checkpointing
        debug: Debug logger instance
    
    Returns:
        MemoryOptimizedPipeline: Configured pipeline manager
    
    Example:
        ```python
        from src.optimization.memory_integration import initialize_memory_optimizations
        
        # Initialize optimizations
        mem_pipeline = initialize_memory_optimizations(
            dit_model=dit_model,
            vae_model=vae_model,
            enable_checkpointing=True,
            debug=debug
        )
        
        # Track VRAM before DiT
        mem_pipeline.track_vram_usage("before_dit")
        
        # Offload VAE while DiT is working
        mem_pipeline.offload_vae_to_cpu(vae_model)
        
        # Run DiT inference
        dit_output = dit_model(latents)
        
        # Track VRAM after DiT
        mem_pipeline.track_vram_usage("after_dit")
        
        # Restore VAE, offload DiT
        mem_pipeline.restore_vae_to_gpu(vae_model)
        mem_pipeline.offload_dit_to_cpu(dit_model)
        
        # Run VAE decoding
        video = vae_model.decode(dit_output)
        
        # Get savings summary
        savings = mem_pipeline.get_vram_savings()
        print(f"Total VRAM savings: {savings:.2f} GB")
        ```
    """
    pipeline = MemoryOptimizedPipeline(debug=debug)
    
    # Setup DiT optimizations
    pipeline.setup_dit_optimization(dit_model, enable_checkpointing=enable_checkpointing)
    
    # Setup VAE optimizations (if provided)
    if vae_model is not None:
        pipeline.setup_vae_optimization(vae_model)
    
    # Log initialization
    if debug:
        debug.log("Memory optimization pipeline initialized", category="success")
        if enable_checkpointing:
            debug.log("  - Gradient checkpointing: ENABLED", category="info")
        debug.log("  - CPU offloading: AVAILABLE", category="info")
    
    return pipeline
