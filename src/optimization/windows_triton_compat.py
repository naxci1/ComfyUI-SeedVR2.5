"""
Windows Triton Compatibility Module for SeedVR2

Provides Windows-specific support for torch.compile with Triton/Inductor backend.
Handles MSVC compiler detection, PATH configuration, and graceful fallbacks.

Technical Background:
- Standard torch.compile (Inductor backend) requires Triton for kernel generation
- Triton requires native C++ compiler for JIT compilation
- On Windows, this means MSVC (Visual Studio Build Tools) with cl.exe

Requirements for torch.compile on Windows 10 with RTX 50xx (Blackwell):
- CUDA 12.8+ (required for Blackwell/SM100 support)
- Triton 3.3+ (Windows support via triton-windows wheel)
- MSVC Build Tools with C++ compiler (cl.exe)
- PyTorch 2.6+ with CUDA support

Usage:
    from src.optimization.windows_triton_compat import (
        ensure_windows_triton_compat,
        setup_msvc_environment,
        get_torch_compile_backend,
        safe_compile_model
    )

Author: SeedVR2 Team
"""

import os
import sys
import platform
import subprocess
import warnings
from typing import Optional, Tuple, Dict, Any, Callable, Protocol, runtime_checkable
from pathlib import Path
from functools import lru_cache

import torch
import torch.nn as nn


@runtime_checkable
class DebugLogger(Protocol):
    """Protocol defining the debug logger interface"""
    def log(self, message: str, level: str = "INFO", category: str = "general", 
            force: bool = False, indent_level: int = 0) -> None: ...
    def start_timer(self, name: str) -> None: ...
    def end_timer(self, name: str, label: str = "") -> None: ...


# Module state
_MSVC_SETUP_DONE = False
_TRITON_AVAILABLE = None
_COMPILE_BACKEND_CACHE = {}


def is_windows() -> bool:
    """Check if running on Windows"""
    return platform.system() == "Windows"


@lru_cache(maxsize=1)
def detect_cuda_version() -> Tuple[int, int]:
    """
    Detect installed CUDA version.
    
    Returns:
        Tuple of (major, minor) CUDA version, e.g., (12, 8)
        Returns (0, 0) if CUDA is not available
    """
    if not torch.cuda.is_available():
        return (0, 0)
    
    try:
        cuda_version = torch.version.cuda
        if cuda_version is None:
            return (0, 0)
        
        parts = cuda_version.split('.')
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
        return (major, minor)
    except Exception:
        return (0, 0)


def is_cuda_128_plus() -> bool:
    """Check if CUDA version is 12.8 or higher (required for Blackwell)"""
    major, minor = detect_cuda_version()
    return (major > 12) or (major == 12 and minor >= 8)


@lru_cache(maxsize=1)
def find_msvc_compiler() -> Optional[str]:
    """
    Find MSVC cl.exe compiler path.
    
    Searches common Visual Studio installation paths and environment variables.
    
    Returns:
        Path to cl.exe if found, None otherwise
    """
    if not is_windows():
        return None
    
    # Check if cl.exe is already in PATH
    try:
        result = subprocess.run(
            ["where", "cl.exe"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().split('\n')[0]
    except Exception:
        pass
    
    # Common VS installation paths
    program_files = os.environ.get("ProgramFiles", "C:\\Program Files")
    program_files_x86 = os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)")
    
    vs_base_paths = [
        Path(program_files) / "Microsoft Visual Studio",
        Path(program_files_x86) / "Microsoft Visual Studio",
    ]
    
    # VS versions to check (newest first)
    vs_versions = ["2022", "2019", "2017"]
    vs_editions = ["Enterprise", "Professional", "Community", "BuildTools"]
    
    for base_path in vs_base_paths:
        for version in vs_versions:
            for edition in vs_editions:
                # Check x64 compiler
                cl_path = (
                    base_path / version / edition /
                    "VC" / "Tools" / "MSVC"
                )
                if cl_path.exists():
                    # Find newest MSVC version
                    try:
                        msvc_versions = sorted(
                            [d for d in cl_path.iterdir() if d.is_dir()],
                            reverse=True
                        )
                        for msvc_ver in msvc_versions:
                            cl_exe = msvc_ver / "bin" / "Hostx64" / "x64" / "cl.exe"
                            if cl_exe.exists():
                                return str(cl_exe)
                    except Exception:
                        continue
    
    return None


@lru_cache(maxsize=1)
def check_openmp_availability() -> bool:
    """
    Check if OpenMP headers are available in the MSVC installation.
    
    The Triton/Inductor backend compiles C++ code with OpenMP flags (/openmp).
    If omp.h is not available (common with minimal VS Build Tools installs),
    compilation will fail with 'Cannot open include file: omp.h'.
    
    Returns:
        True if OpenMP appears to be available, False otherwise
    """
    if not is_windows():
        return True  # Assume available on Linux/Mac
    
    # Find cl.exe path to locate include directories
    cl_path = find_msvc_compiler()
    if cl_path is None:
        return False
    
    # From cl.exe path, navigate to include directory
    # cl.exe is at: .../MSVC/<version>/bin/Hostx64/x64/cl.exe
    # omp.h is at: .../MSVC/<version>/include/omp.h
    try:
        cl_exe_path = Path(cl_path)
        # Go up from bin/Hostx64/x64/cl.exe to MSVC/<version>
        msvc_root = cl_exe_path.parent.parent.parent.parent
        omp_header = msvc_root / "include" / "omp.h"
        
        if omp_header.exists():
            return True
        
        # Also check in the parallel patterns library location
        # Some VS installations put it elsewhere
        alternate_locations = [
            msvc_root / "include" / "openmp" / "omp.h",
            msvc_root.parent.parent / "Auxiliary" / "VS" / "include" / "omp.h",
        ]
        
        for alt_path in alternate_locations:
            if alt_path.exists():
                return True
        
        return False
        
    except Exception:
        return False


def configure_inductor_for_windows(debug: Optional[DebugLogger] = None) -> None:
    """
    Configure PyTorch Inductor environment variables for Windows compatibility.
    
    Sets environment variables to work around common Windows issues:
    - Disables OpenMP if omp.h is not available
    - Configures temp directories
    - Sets reasonable defaults for Windows
    
    Args:
        debug: Optional debug instance for logging
    """
    if not is_windows():
        return
    
    # Check OpenMP availability
    openmp_available = check_openmp_availability()
    
    if not openmp_available:
        # Disable OpenMP in Inductor C++ compilation
        # This prevents the 'omp.h not found' error
        os.environ["TORCHINDUCTOR_CPP_OPENMP"] = "0"
        
        if debug:
            debug.log(
                "OpenMP headers (omp.h) not found in MSVC installation. "
                "Disabled OpenMP for torch.compile C++ compilation. "
                "For better performance, install VS Build Tools with 'Desktop development with C++' workload.",
                level="WARNING", category="setup", force=True
            )
    
    # Set reasonable defaults for Windows
    # Disable freezing which can cause issues
    os.environ.setdefault("TORCHINDUCTOR_FREEZING", "0")
    
    # Use a safe temp directory
    if "TORCHINDUCTOR_OUTPUT_CODE" not in os.environ:
        # Don't change if user has set it
        pass


def get_vcvars_path() -> Optional[str]:
    """
    Find vcvars64.bat for VS environment setup.
    
    Returns:
        Path to vcvars64.bat if found, None otherwise
    """
    if not is_windows():
        return None
    
    program_files = os.environ.get("ProgramFiles", "C:\\Program Files")
    program_files_x86 = os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)")
    
    vs_base_paths = [
        Path(program_files) / "Microsoft Visual Studio",
        Path(program_files_x86) / "Microsoft Visual Studio",
    ]
    
    vs_versions = ["2022", "2019", "2017"]
    vs_editions = ["Enterprise", "Professional", "Community", "BuildTools"]
    
    for base_path in vs_base_paths:
        for version in vs_versions:
            for edition in vs_editions:
                vcvars = (
                    base_path / version / edition /
                    "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
                )
                if vcvars.exists():
                    return str(vcvars)
    
    return None


def setup_msvc_environment(debug: Optional[DebugLogger] = None) -> bool:
    """
    Setup MSVC environment for Triton JIT compilation.
    
    Finds and adds MSVC compiler (cl.exe) to the system PATH.
    This is required for Triton to compile CUDA kernels on Windows.
    
    Args:
        debug: Optional debug instance for logging
        
    Returns:
        True if MSVC is available and configured, False otherwise
    """
    global _MSVC_SETUP_DONE
    
    if _MSVC_SETUP_DONE:
        return True
    
    if not is_windows():
        _MSVC_SETUP_DONE = True
        return True  # Not needed on non-Windows
    
    # Check if cl.exe already in PATH
    cl_path = find_msvc_compiler()
    
    if cl_path is None:
        if debug:
            debug.log(
                "MSVC compiler (cl.exe) not found. torch.compile with inductor backend "
                "may fall back to eager mode.\n"
                "Install Visual Studio Build Tools: "
                "https://visualstudio.microsoft.com/visual-cpp-build-tools/",
                level="WARNING", category="setup", force=True
            )
        return False
    
    # Add compiler directory to PATH if not already there
    cl_dir = str(Path(cl_path).parent)
    current_path = os.environ.get("PATH", "")
    
    if cl_dir.lower() not in current_path.lower():
        os.environ["PATH"] = f"{cl_dir};{current_path}"
        if debug:
            debug.log(f"Added MSVC compiler to PATH: {cl_dir}", category="setup")
    
    _MSVC_SETUP_DONE = True
    return True


@lru_cache(maxsize=1)
def check_triton_availability() -> Tuple[bool, str]:
    """
    Check if Triton is available and functional.
    
    For Windows, checks for triton-windows package.
    For Linux/Mac, checks for standard triton package.
    
    Returns:
        Tuple of (is_available, status_message)
    """
    global _TRITON_AVAILABLE
    
    try:
        import triton
        triton_version = getattr(triton, "__version__", "unknown")
        
        # Check minimum version for Blackwell support (3.3+)
        if triton_version != "unknown":
            try:
                parts = triton_version.split('.')
                major = int(parts[0])
                minor = int(parts[1]) if len(parts) > 1 else 0
                
                if (major, minor) < (3, 3):
                    return (True, f"Triton {triton_version} available (upgrade to 3.3+ for Blackwell support)")
            except (ValueError, IndexError):
                pass
        
        _TRITON_AVAILABLE = True
        return (True, f"Triton {triton_version} available")
        
    except ImportError as e:
        _TRITON_AVAILABLE = False
        
        if is_windows():
            # Note: PyTorch wheel URLs may change. Check https://pytorch.org/get-started/locally/
            # for current installation instructions.
            return (False, 
                "Triton not installed. For Windows, install triton-windows:\n"
                "  pip install triton-windows\n"
                "Or for CUDA 12.8+ (check pytorch.org for current URLs):\n"
                "  pip install --pre triton --index-url https://download.pytorch.org/whl/nightly/cu128"
            )
        else:
            return (False, f"Triton not installed: {e}")


def ensure_windows_triton_compat(debug: Optional[DebugLogger] = None) -> bool:
    """
    Ensure Windows environment is properly configured for Triton/torch.compile.
    
    This function:
    1. Sets up MSVC compiler in PATH (if available)
    2. Checks Triton availability
    3. Configures Inductor environment variables (including OpenMP workaround)
    4. Sets torch.compile environment variables
    
    Should be called once at application startup before using torch.compile.
    
    Args:
        debug: Optional debug instance for logging
        
    Returns:
        True if environment is fully configured for torch.compile, False otherwise
    """
    if not is_windows():
        # Non-Windows: just check Triton
        triton_ok, msg = check_triton_availability()
        if debug and not triton_ok:
            debug.log(msg, level="WARNING", category="setup", force=True)
        return triton_ok
    
    # Windows-specific setup
    all_ok = True
    
    # 1. Setup MSVC
    if not setup_msvc_environment(debug):
        all_ok = False
    
    # 2. Check Triton
    triton_ok, triton_msg = check_triton_availability()
    if not triton_ok:
        if debug:
            debug.log(triton_msg, level="WARNING", category="setup", force=True)
        all_ok = False
    elif debug:
        debug.log(triton_msg, category="setup")
    
    # 3. Configure Inductor for Windows (including OpenMP workaround)
    # This must be called even if Triton is not available, as it sets
    # environment variables that prevent crashes when torch.compile is attempted
    configure_inductor_for_windows(debug)
    
    # 4. Set additional environment variables for Windows torch.compile
    if all_ok:
        # Disable some features that cause issues on Windows
        os.environ.setdefault("TORCH_COMPILE_DEBUG", "0")
        
        # Enable TF32 for better Tensor Core utilization
        os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "1")
        
        if debug:
            debug.log("Windows torch.compile environment configured", category="setup")
    
    return all_ok


def get_torch_compile_backend(
    preferred_backend: str = "inductor",
    fallback_to_eager: bool = True,
    debug: Optional[DebugLogger] = None
) -> Optional[str]:
    """
    Get the best available torch.compile backend.
    
    Attempts to use the preferred backend, falling back to alternatives
    if not available. Returns None if no compilation is possible.
    
    Args:
        preferred_backend: Preferred backend ("inductor", "cudagraphs", "eager")
        fallback_to_eager: If True, returns None (eager mode) as final fallback
        debug: Optional debug instance for logging
        
    Returns:
        Backend name to use, or None for eager mode
    """
    # Check cache
    cache_key = (preferred_backend, fallback_to_eager)
    if cache_key in _COMPILE_BACKEND_CACHE:
        return _COMPILE_BACKEND_CACHE[cache_key]
    
    # Check if CUDA is available (required for most backends)
    if not torch.cuda.is_available():
        if debug:
            debug.log("CUDA not available, using eager mode", level="WARNING", category="setup")
        _COMPILE_BACKEND_CACHE[cache_key] = None
        return None
    
    # Try preferred backend
    if preferred_backend == "inductor":
        triton_ok, _ = check_triton_availability()
        if triton_ok:
            _COMPILE_BACKEND_CACHE[cache_key] = "inductor"
            return "inductor"
        
        # Fall back to cudagraphs
        if debug:
            debug.log(
                "Inductor backend requires Triton. Falling back to cudagraphs backend.",
                level="WARNING", category="setup"
            )
        preferred_backend = "cudagraphs"
    
    if preferred_backend == "cudagraphs":
        # cudagraphs works without Triton
        _COMPILE_BACKEND_CACHE[cache_key] = "cudagraphs"
        return "cudagraphs"
    
    if fallback_to_eager:
        if debug:
            debug.log("Using eager mode (no compilation)", category="setup")
        _COMPILE_BACKEND_CACHE[cache_key] = None
        return None
    
    return preferred_backend


def safe_compile_model(
    model: nn.Module,
    mode: str = "reduce-overhead",
    backend: str = "inductor",
    fullgraph: bool = False,
    dynamic: bool = False,
    debug: Optional[DebugLogger] = None,
    model_name: str = "model"
) -> nn.Module:
    """
    Safely compile a model with torch.compile, handling Windows-specific issues.
    
    This function:
    1. Ensures Windows Triton compatibility
    2. Selects best available backend with fallbacks
    3. Handles compilation errors gracefully
    4. Returns original model if compilation fails
    
    Args:
        model: PyTorch model to compile
        mode: Compilation mode ("default", "reduce-overhead", "max-autotune")
        backend: Preferred backend ("inductor", "cudagraphs")
        fullgraph: Whether to compile as single graph
        dynamic: Whether to use dynamic shapes
        debug: Optional debug instance for logging
        model_name: Name for logging purposes
        
    Returns:
        Compiled model, or original model if compilation fails
    """
    # Check PyTorch version
    torch_version = tuple(int(x) for x in torch.__version__.split('.')[:2])
    if torch_version < (2, 0):
        if debug:
            debug.log(
                f"torch.compile requires PyTorch 2.0+. Found {torch.__version__}",
                level="WARNING", category="setup"
            )
        return model
    
    # Ensure Windows compatibility
    ensure_windows_triton_compat(debug)
    
    # Get available backend
    actual_backend = get_torch_compile_backend(backend, fallback_to_eager=True, debug=debug)
    
    if actual_backend is None:
        if debug:
            debug.log(
                f"No torch.compile backend available for {model_name}, using eager mode",
                level="WARNING", category="setup"
            )
        return model
    
    # Attempt compilation
    try:
        if debug:
            debug.log(
                f"Compiling {model_name} with backend={actual_backend}, mode={mode}",
                category="setup"
            )
            debug.start_timer(f"compile_{model_name}")
        
        compiled_model = torch.compile(
            model,
            backend=actual_backend,
            mode=mode,
            fullgraph=fullgraph,
            dynamic=dynamic
        )
        
        if debug:
            debug.end_timer(f"compile_{model_name}", f"{model_name} compilation")
            debug.log(f"{model_name} compiled successfully", category="success")
        
        return compiled_model
        
    except Exception as e:
        if debug:
            debug.log(
                f"torch.compile failed for {model_name}: {e}\n"
                f"Falling back to eager mode",
                level="WARNING", category="setup", force=True
            )
        return model


def get_compilation_status() -> Dict[str, Any]:
    """
    Get detailed status of torch.compile environment.
    
    Returns:
        Dictionary with compilation environment status
    """
    triton_ok, triton_msg = check_triton_availability()
    cuda_version = detect_cuda_version()
    
    status = {
        'platform': platform.system(),
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': f"{cuda_version[0]}.{cuda_version[1]}" if cuda_version[0] > 0 else "N/A",
        'cuda_128_plus': is_cuda_128_plus(),
        'triton_available': triton_ok,
        'triton_status': triton_msg,
        'inductor_available': triton_ok,
        'cudagraphs_available': torch.cuda.is_available(),
    }
    
    if is_windows():
        cl_path = find_msvc_compiler()
        openmp_available = check_openmp_availability()
        status['msvc_available'] = cl_path is not None
        status['msvc_path'] = cl_path
        status['msvc_setup_done'] = _MSVC_SETUP_DONE
        status['openmp_available'] = openmp_available
        status['openmp_disabled'] = os.environ.get("TORCHINDUCTOR_CPP_OPENMP") == "0"
    
    return status


# Module exports
__all__ = [
    'is_windows',
    'detect_cuda_version',
    'is_cuda_128_plus',
    'find_msvc_compiler',
    'get_vcvars_path',
    'setup_msvc_environment',
    'check_triton_availability',
    'check_openmp_availability',
    'configure_inductor_for_windows',
    'ensure_windows_triton_compat',
    'get_torch_compile_backend',
    'safe_compile_model',
    'get_compilation_status',
]
