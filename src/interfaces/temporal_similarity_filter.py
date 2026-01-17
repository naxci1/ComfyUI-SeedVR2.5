"""
Temporal Similarity Filter Node for SeedVR2

Implements frame-to-frame latent comparison to bypass DiT upscaling
for highly similar consecutive frames, saving 100% compute for those frames.

This optimization is particularly effective for:
- Static scenes with minimal motion
- High-framerate video with temporal redundancy
- Sequences with slow camera movements
"""

import torch
from comfy_api.latest import io
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class TemporalSimilarityConfig:
    """
    Configuration for temporal similarity filtering.
    
    Attributes:
        enabled: Whether filtering is active
        similarity_threshold: Threshold for frame similarity (default 0.98 = 98%)
        reuse_previous: Whether to reuse previous upscaled frame when similar
    """
    enabled: bool = True
    similarity_threshold: float = 0.98
    reuse_previous: bool = True


class TemporalSimilarityCache:
    """
    Cache for storing previous latents and upscaled outputs for frame reuse.
    
    This cache enables the temporal similarity filter to:
    1. Compare current frame latent with previous frame latent
    2. Reuse previous upscaled output when frames are similar
    3. Track statistics for optimization analysis
    """
    
    def __init__(self):
        self.previous_latent: Optional[torch.Tensor] = None
        self.previous_upscaled: Optional[torch.Tensor] = None
        self.skip_count: int = 0
        self.total_count: int = 0
        self._last_similarity: float = 0.0
    
    def compute_similarity(self, current: torch.Tensor, previous: torch.Tensor) -> float:
        """
        Compute cosine similarity between current and previous latent.
        
        Uses cosine similarity which is more robust than L2 for comparing
        high-dimensional latent representations.
        
        Args:
            current: Current frame latent
            previous: Previous frame latent
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        if previous is None:
            return 0.0
        
        if current.shape != previous.shape:
            return 0.0
        
        # Flatten for similarity computation
        current_flat = current.flatten().float()
        previous_flat = previous.flatten().float()
        
        # Compute cosine similarity
        dot_product = torch.dot(current_flat, previous_flat)
        norm_current = torch.norm(current_flat)
        norm_previous = torch.norm(previous_flat)
        
        # Convert to scalars for comparison to avoid "Boolean value of Tensor is ambiguous" error
        norm_current_scalar = norm_current.item()
        norm_previous_scalar = norm_previous.item()
        if norm_current_scalar < 1e-8 or norm_previous_scalar < 1e-8:
            return 0.0
        
        similarity = (dot_product / (norm_current * norm_previous)).item()
        
        # Clamp to valid range
        return max(0.0, min(1.0, similarity))
    
    def should_reuse_previous(self, current_latent: torch.Tensor, 
                               threshold: float = 0.98) -> bool:
        """
        Determine if previous upscaled frame should be reused.
        
        Args:
            current_latent: Current frame latent
            threshold: Similarity threshold (default 0.98 = 98%)
            
        Returns:
            True if previous frame should be reused
        """
        if self.previous_latent is None or self.previous_upscaled is None:
            return False
        
        similarity = self.compute_similarity(current_latent, self.previous_latent)
        self._last_similarity = similarity  # Store for logging
        return similarity >= threshold
    
    def should_reuse_previous_with_similarity(self, current_latent: torch.Tensor, 
                                               threshold: float = 0.98) -> Tuple[bool, float]:
        """
        Determine if previous upscaled frame should be reused, returning similarity.
        
        Args:
            current_latent: Current frame latent
            threshold: Similarity threshold (default 0.98 = 98%)
            
        Returns:
            Tuple of (should_reuse, similarity_value)
        """
        if self.previous_latent is None or self.previous_upscaled is None:
            return False, 0.0
        
        similarity = self.compute_similarity(current_latent, self.previous_latent)
        self._last_similarity = similarity
        return similarity >= threshold, similarity
    
    def update(self, latent: torch.Tensor, upscaled: torch.Tensor):
        """
        Update cache with current frame data.
        
        Args:
            latent: Current frame latent
            upscaled: Upscaled output for current frame
        """
        self.previous_latent = latent.detach().clone()
        self.previous_upscaled = upscaled.detach().clone()
    
    def get_previous_upscaled(self) -> Optional[torch.Tensor]:
        """Get cached previous upscaled output."""
        return self.previous_upscaled
    
    def record_decision(self, reused: bool, similarity: float = 0.0):
        """
        Record whether frame was reused for statistics.
        
        Logs: "[Temporal Filter] Frame similarity: X%. Bypassing DiT: YES/NO"
        """
        self.total_count += 1
        if reused:
            self.skip_count += 1
        # Log with requested format
        bypass_status = "YES" if reused else "NO"
        print(f"[Temporal Filter] Frame similarity: {similarity*100:.1f}%. Bypassing DiT: {bypass_status}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get filtering statistics."""
        reuse_rate = self.skip_count / self.total_count if self.total_count > 0 else 0.0
        compute_saved = self.skip_count  # Each reused frame saves 100% compute
        return {
            'frames_reused': self.skip_count,
            'frames_processed': self.total_count - self.skip_count,
            'total_frames': self.total_count,
            'reuse_rate': reuse_rate,
            'compute_saved_percent': reuse_rate * 100
        }
    
    def reset(self):
        """Reset cache state."""
        self.previous_latent = None
        self.previous_upscaled = None
        self.skip_count = 0
        self.total_count = 0


class SeedVR2TemporalSimilarityFilter(io.ComfyNode):
    """
    Temporal Similarity Filter for SeedVR2 Video Upscaler
    
    Compares consecutive frame latents and bypasses DiT upscaling when
    frames are highly similar (>98% by default), reusing the previous
    upscaled frame to save 100% of compute for that frame.
    
    Particularly effective for:
    - Static or slow-motion scenes
    - High framerate video (60fps+)
    - Sequences with minimal camera movement
    """
    
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SeedVR2TemporalSimilarityFilter",
            display_name="SeedVR2 Temporal Similarity Filter",
            category="SEEDVR2/Optimization",
            description=(
                "Optimizes video processing by detecting similar consecutive frames. "
                "When frames are >98% similar, bypasses DiT upscaling and reuses "
                "the previous upscaled frame, saving 100% compute for that frame.\n\n"
                "Best for: static scenes, high framerate video, slow camera movements."
            ),
            inputs=[
                io.Image.Input("latents",
                    tooltip=(
                        "Input latent batch to analyze for temporal similarity.\n"
                        "Typically connected to VAE encoded latents."
                    )
                ),
                io.Float.Input("similarity_threshold",
                    default=0.98,
                    min=0.5,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "Similarity threshold for frame reuse (default: 0.98 = 98%).\n"
                        "Higher values = fewer frames reused (safer).\n"
                        "Lower values = more frames reused (faster but may cause artifacts)."
                    )
                ),
                io.Boolean.Input("enabled",
                    default=True,
                    tooltip="Enable/disable temporal similarity filtering."
                ),
                io.Boolean.Input("show_stats",
                    default=True,
                    optional=True,
                    tooltip="Print filtering statistics to console."
                ),
            ],
            outputs=[
                io.Custom("TEMPORAL_FILTER_CONFIG").Output(
                    tooltip="Temporal filter configuration to pass to video upscaler."
                )
            ]
        )
    
    @classmethod
    def execute(cls, latents: torch.Tensor, similarity_threshold: float = 0.98,
                enabled: bool = True, show_stats: bool = True) -> io.NodeOutput:
        """
        Configure temporal similarity filtering.
        
        Args:
            latents: Input latent batch for analysis
            similarity_threshold: Threshold for frame similarity (0.5-1.0)
            enabled: Whether filtering is active
            show_stats: Whether to print statistics
            
        Returns:
            NodeOutput containing filter configuration
        """
        config = TemporalSimilarityConfig(
            enabled=enabled,
            similarity_threshold=similarity_threshold,
            reuse_previous=True
        )
        
        # Create cache instance
        cache = TemporalSimilarityCache()
        
        # Analyze latent batch for potential savings
        if enabled and latents is not None and len(latents.shape) >= 4:
            num_frames = latents.shape[0]
            similar_pairs = 0
            
            for i in range(1, num_frames):
                similarity = cache.compute_similarity(latents[i], latents[i-1])
                if similarity >= similarity_threshold:
                    similar_pairs += 1
            
            potential_savings = (similar_pairs / max(1, num_frames - 1)) * 100
            
            if show_stats:
                print(f"[Blackwell-Optimized] Temporal Similarity Analysis:")
                print(f"  Total frames: {num_frames}")
                print(f"  Similar consecutive pairs: {similar_pairs}/{num_frames-1}")
                print(f"  Potential compute savings: {potential_savings:.1f}%")
                print(f"  Similarity threshold: {similarity_threshold:.2f}")
        
        # Return configuration for use by video upscaler
        output_config = {
            'config': config,
            'cache': cache,
            'enabled': enabled,
            'similarity_threshold': similarity_threshold
        }
        
        return io.NodeOutput(output_config)


def filter_frames_with_temporal_similarity(
    latents: List[torch.Tensor],
    upscale_fn,
    config: TemporalSimilarityConfig,
    cache: Optional[TemporalSimilarityCache] = None,
    debug: Optional[Any] = None
) -> List[torch.Tensor]:
    """
    Apply temporal similarity filtering to a batch of latents during upscaling.
    
    This function wraps the upscaling process and applies frame reuse logic:
    1. For each frame, check if it's similar to the previous frame
    2. If similar (>threshold), reuse previous upscaled output
    3. If not similar, perform full upscaling and cache result
    
    Args:
        latents: List of latent tensors for each frame
        upscale_fn: Function to upscale a single latent
        config: Temporal similarity configuration
        cache: Cache instance (created if None)
        debug: Optional debug instance for logging
        
    Returns:
        List of upscaled tensors (some may be reused from previous frames)
    """
    if cache is None:
        cache = TemporalSimilarityCache()
    
    results = []
    
    for i, latent in enumerate(latents):
        # Check if we should reuse previous frame - get similarity value too
        should_reuse = False
        similarity = 0.0
        
        if config.enabled and config.reuse_previous:
            should_reuse, similarity = cache.should_reuse_previous_with_similarity(
                latent, config.similarity_threshold
            )
        
        if should_reuse:
            # Reuse previous upscaled output - saves 100% compute
            upscaled = cache.get_previous_upscaled()
            if upscaled is not None:
                results.append(upscaled.clone())
                cache.record_decision(reused=True, similarity=similarity)
                
                if debug:
                    debug.log(f"Frame {i}: Reused (100% compute saved)", 
                             category="temporal", indent_level=1)
                continue
        
        # Perform full upscaling
        upscaled = upscale_fn(latent)
        results.append(upscaled)
        
        # Update cache for next frame comparison
        cache.update(latent, upscaled)
        cache.record_decision(reused=False, similarity=similarity)
        
        if debug:
            debug.log(f"Frame {i}: Upscaled", category="temporal", indent_level=1)
    
    # Log final statistics
    if debug:
        stats = cache.get_stats()
        debug.log(
            f"Temporal similarity: {stats['frames_reused']}/{stats['total_frames']} "
            f"frames reused ({stats['compute_saved_percent']:.1f}% compute saved)",
            category="temporal", force=True
        )
    
    return results
