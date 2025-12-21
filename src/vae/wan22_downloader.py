"""
Wan2.2 VAE Auto-Download Module for ComfyUI-SeedVR2.5

This module provides automatic download functionality for Wan2.2 VAE models from HuggingFace.
It integrates with the SeedVR2 download infrastructure and ensures models are available
in the ComfyUI VAE dropdown menu.

Features:
- Auto-detection of missing Wan2.2 VAE models
- Resumable downloads with progress bar
- File integrity verification via SHA256 hash
- Support for FP16 and BF16 precision variants
"""

import os
import hashlib
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class Wan22VAEVariant(Enum):
    """Wan2.2 VAE model variants"""
    FP16 = "fp16"
    BF16 = "bf16"
    GGUF_Q4 = "gguf_q4"
    GGUF_Q8 = "gguf_q8"


@dataclass
class Wan22VAEModel:
    """Wan2.2 VAE model definition"""
    filename: str
    repo: str
    precision: str
    sha256: Optional[str] = None
    description: str = ""
    file_size_gb: float = 0.0


# Wan2.2 VAE model registry
# Note: These are placeholder entries. Update repo/sha256 when official models are released.
WAN22_VAE_MODELS: Dict[str, Wan22VAEModel] = {
    "wan2.2_vae_fp16.safetensors": Wan22VAEModel(
        filename="wan2.2_vae_fp16.safetensors",
        repo="Wan-AI/Wan2.2-VAE",  # Placeholder - update when official repo is available
        precision="fp16",
        sha256=None,  # Will be populated once official model is available
        description="Wan2.2 3D Causal VAE - FP16 precision",
        file_size_gb=0.5,
    ),
    "wan2.2_vae_bf16.safetensors": Wan22VAEModel(
        filename="wan2.2_vae_bf16.safetensors",
        repo="Wan-AI/Wan2.2-VAE",  # Placeholder - update when official repo is available
        precision="bf16",
        sha256=None,
        description="Wan2.2 3D Causal VAE - BF16 precision",
        file_size_gb=0.5,
    ),
}

# Default Wan2.2 VAE model
DEFAULT_WAN22_VAE = "wan2.2_vae_fp16.safetensors"


def get_wan22_vae_models() -> List[str]:
    """Get list of available Wan2.2 VAE model names"""
    return list(WAN22_VAE_MODELS.keys())


def get_wan22_vae_info(model_name: str) -> Optional[Wan22VAEModel]:
    """Get info for a specific Wan2.2 VAE model"""
    return WAN22_VAE_MODELS.get(model_name)


def ensure_wan22_vae_exists(
    model_name: str = DEFAULT_WAN22_VAE,
    cache_dir: Optional[str] = None,
    force_download: bool = False,
    debug: Optional[object] = None,
) -> Tuple[bool, str]:
    """
    Ensure Wan2.2 VAE model exists, downloading if necessary.
    
    This function checks if the specified Wan2.2 VAE model exists in the local
    SeedVR2 VAE directory. If not found, it triggers an automatic download from
    HuggingFace with a progress bar.
    
    Args:
        model_name: Name of the model file to check/download
        cache_dir: Optional directory for model storage. Uses default if None.
        force_download: If True, re-download even if file exists
        debug: Optional debug logger instance
        
    Returns:
        Tuple of (success: bool, filepath: str)
        
    Raises:
        ValueError: If model_name is not in WAN22_VAE_MODELS
    """
    # Import here to avoid circular imports
    from ..utils.constants import (
        get_base_cache_dir, 
        find_model_file, 
        HUGGINGFACE_BASE_URL,
    )
    from ..utils.downloads import (
        download_with_resume, 
        validate_file, 
        is_file_validated_cached,
        DOWNLOAD_MAX_RETRIES,
        DOWNLOAD_RETRY_DELAY,
    )
    import time
    
    # Get model info
    model_info = WAN22_VAE_MODELS.get(model_name)
    if model_info is None:
        available = ", ".join(WAN22_VAE_MODELS.keys())
        raise ValueError(f"Unknown Wan2.2 VAE model: {model_name}. Available: {available}")
    
    # Determine cache directory
    if cache_dir is None:
        cache_dir = get_base_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    
    # Check if file exists in any registered path
    existing_filepath = find_model_file(model_name, fallback_dir=cache_dir)
    
    if os.path.exists(existing_filepath) and not force_download:
        # Check validation cache first (fast)
        if is_file_validated_cached(existing_filepath, cache_dir):
            if debug:
                debug.log(f"Wan2.2 VAE found (cached): {existing_filepath}", category="setup")
            return True, existing_filepath
        
        # Validate file integrity
        if validate_file(existing_filepath, model_info.sha256, cache_dir):
            if debug:
                debug.log(f"Wan2.2 VAE validated: {existing_filepath}", category="setup")
            return True, existing_filepath
        else:
            # File corrupted, need to re-download
            if debug:
                debug.log(f"Wan2.2 VAE corrupted, re-downloading: {model_name}", 
                         level="WARNING", category="download", force=True)
            os.remove(existing_filepath)
    
    # File doesn't exist or force_download - need to download
    filepath = os.path.join(cache_dir, model_name)
    url = HUGGINGFACE_BASE_URL.format(repo=model_info.repo, filename=model_name)
    
    if debug:
        debug.log(f"Downloading Wan2.2 VAE: {model_name}", category="download", force=True)
        debug.log(f"Source: {url}", category="download", indent_level=1)
        debug.log(f"Target: {filepath}", category="download", indent_level=1)
        if model_info.file_size_gb > 0:
            debug.log(f"Size: ~{model_info.file_size_gb:.1f} GB", category="download", indent_level=1)
    else:
        logger.info(f"Downloading Wan2.2 VAE: {model_name} from {url}")
    
    # Download with retries
    success = False
    for attempt in range(DOWNLOAD_MAX_RETRIES):
        if attempt > 0:
            time.sleep(DOWNLOAD_RETRY_DELAY * attempt)
            if debug:
                debug.log(f"Retry {attempt}/{DOWNLOAD_MAX_RETRIES}", 
                         category="download", force=True)
            else:
                logger.info(f"Retry {attempt}/{DOWNLOAD_MAX_RETRIES}")
        
        if download_with_resume(url, filepath, debug):
            # Validate downloaded file
            if validate_file(filepath, model_info.sha256, cache_dir):
                if debug:
                    debug.log(f"Wan2.2 VAE downloaded and validated: {model_name}", 
                             category="success", force=True)
                else:
                    logger.info(f"Wan2.2 VAE downloaded and validated: {model_name}")
                success = True
                break
            else:
                # Remove corrupted download
                temp_file = f"{filepath}.download"
                for f in [filepath, temp_file]:
                    if os.path.exists(f):
                        os.remove(f)
                if debug:
                    debug.log(f"Downloaded file failed validation, retrying...", 
                             level="WARNING", category="download", force=True)
                else:
                    logger.warning("Downloaded file failed validation, retrying...")
    
    if not success:
        error_msg = f"Failed to download Wan2.2 VAE: {model_name} after {DOWNLOAD_MAX_RETRIES} attempts"
        if debug:
            debug.log(error_msg, level="ERROR", category="download", force=True)
            debug.log(f"Manual download: https://huggingface.co/{model_info.repo}/blob/main/{model_name}", 
                     category="info", force=True)
            debug.log(f"Save to: {filepath}", category="info", force=True)
        else:
            logger.error(error_msg)
        return False, filepath
    
    return True, filepath


def check_wan22_vae_status(cache_dir: Optional[str] = None) -> Dict[str, Dict]:
    """
    Check the status of all Wan2.2 VAE models.
    
    Returns a dictionary with status info for each model:
    - exists: bool - whether the file exists
    - validated: bool - whether the file passed validation
    - filepath: str - full path to the file
    
    Args:
        cache_dir: Optional directory to check. Uses default if None.
        
    Returns:
        Dictionary mapping model names to status dictionaries
    """
    from ..utils.constants import get_base_cache_dir, find_model_file
    from ..utils.downloads import is_file_validated_cached
    
    if cache_dir is None:
        cache_dir = get_base_cache_dir()
    
    status = {}
    for model_name in WAN22_VAE_MODELS:
        filepath = find_model_file(model_name, fallback_dir=cache_dir)
        exists = os.path.exists(filepath)
        validated = is_file_validated_cached(filepath, cache_dir) if exists else False
        
        status[model_name] = {
            "exists": exists,
            "validated": validated,
            "filepath": filepath,
            "info": WAN22_VAE_MODELS[model_name],
        }
    
    return status


def download_all_wan22_vae_models(
    cache_dir: Optional[str] = None,
    debug: Optional[object] = None,
) -> Dict[str, bool]:
    """
    Download all available Wan2.2 VAE models.
    
    Args:
        cache_dir: Optional directory for model storage
        debug: Optional debug logger instance
        
    Returns:
        Dictionary mapping model names to download success status
    """
    results = {}
    for model_name in WAN22_VAE_MODELS:
        success, _ = ensure_wan22_vae_exists(model_name, cache_dir, debug=debug)
        results[model_name] = success
    return results


def get_wan22_download_url(model_name: str) -> Optional[str]:
    """
    Get the download URL for a Wan2.2 VAE model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Download URL or None if model not found
    """
    from ..utils.constants import HUGGINGFACE_BASE_URL
    
    model_info = WAN22_VAE_MODELS.get(model_name)
    if model_info is None:
        return None
    
    return HUGGINGFACE_BASE_URL.format(repo=model_info.repo, filename=model_name)


# Convenience function for node integration
def get_or_download_wan22_vae(
    model_name: str = DEFAULT_WAN22_VAE,
    debug: Optional[object] = None,
) -> Optional[str]:
    """
    Get path to Wan2.2 VAE model, downloading if necessary.
    
    This is a convenience wrapper for use in ComfyUI nodes.
    Returns None if download fails.
    
    Args:
        model_name: Name of the model to get
        debug: Optional debug logger instance
        
    Returns:
        Full path to the model file, or None if unavailable
    """
    try:
        success, filepath = ensure_wan22_vae_exists(model_name, debug=debug)
        return filepath if success else None
    except Exception as e:
        if debug:
            debug.log(f"Error getting Wan2.2 VAE: {e}", level="ERROR", category="download", force=True)
        else:
            logger.error(f"Error getting Wan2.2 VAE: {e}")
        return None
