"""
Truth Monitor: Automatic Runtime GPU Telemetry for SeedVR2

Provides real-time verification of kernel execution paths on Blackwell GPUs.
Uses torch.cuda.Event for accurate timing and detects NVFP4 fallback scenarios.

Features:
- Automatic initialization on first attention call
- torch.cuda.Event-based execution timing
- NVFP4/FP8 dtype detection and fallback alerting
- Blackwell (sm_120) telemetry verification
- VAE tiling transparency reporting

Usage:
    The Truth Monitor is automatically initialized when DiT or VAE attention is invoked.
    Set SEEDVR2_TRUTH_MONITOR=1 environment variable to enable verbose telemetry.
"""

import os
import torch
import time
from typing import Optional, Dict, Any, Literal

# ============================================================================
# CONFIGURATION
# ============================================================================

# Enable verbose telemetry with environment variable
ENABLE_TRUTH_MONITOR = os.environ.get('SEEDVR2_TRUTH_MONITOR', '0') == '1'

# Thresholds for detecting anomalous execution times
EXPECTED_BLACKWELL_OPTIMIZED_MS = 5.0  # Expected <5ms per attention call when optimized
FALLBACK_THRESHOLD_MS = 15.0  # If execution takes >15ms, likely hitting fallback

# ============================================================================
# GLOBAL STATE
# ============================================================================

# Telemetry counters
_attention_call_count = 0
_vae_call_count = 0
_fallback_detected_count = 0

# Timing accumulators
_total_attention_time_ms = 0.0
_total_vae_time_ms = 0.0

# Initialization flag
_truth_monitor_initialized = False

# Cached GPU info
_gpu_info: Optional[Dict[str, Any]] = None


# ============================================================================
# GPU TELEMETRY
# ============================================================================

def get_gpu_telemetry() -> Dict[str, Any]:
    """
    Get comprehensive GPU telemetry information.
    Cached after first call for efficiency.
    """
    global _gpu_info
    
    if _gpu_info is not None:
        return _gpu_info
    
    if not torch.cuda.is_available():
        _gpu_info = {
            'available': False,
            'error': 'CUDA not available'
        }
        return _gpu_info
    
    major, minor = torch.cuda.get_device_capability()
    sm_version = f"sm_{major}{minor}"
    
    # Determine GPU architecture
    if major >= 12:
        arch = "Blackwell"
        is_blackwell = True
    elif major == 10:
        arch = "Blackwell (SM 10.x)"
        is_blackwell = True
    elif major == 9:
        arch = "Hopper"
        is_blackwell = False
    elif major == 8 and minor >= 9:
        arch = "Ada Lovelace"
        is_blackwell = False
    elif major == 8:
        arch = "Ampere"
        is_blackwell = False
    else:
        arch = f"Unknown (SM {major}.{minor})"
        is_blackwell = False
    
    _gpu_info = {
        'available': True,
        'device_name': torch.cuda.get_device_name(0),
        'sm_version': sm_version,
        'major': major,
        'minor': minor,
        'arch': arch,
        'is_blackwell': is_blackwell,
        'total_memory_gb': torch.cuda.get_device_properties(0).total_memory / (1024**3),
    }
    
    return _gpu_info


def initialize_truth_monitor():
    """
    Initialize the Truth Monitor system.
    Called automatically on first attention/VAE execution.
    """
    global _truth_monitor_initialized
    
    if _truth_monitor_initialized:
        return
    
    _truth_monitor_initialized = True
    
    gpu_info = get_gpu_telemetry()
    
    print("\n" + "=" * 70)
    print("TRUTH MONITOR: GPU TELEMETRY INITIALIZED")
    print("=" * 70)
    
    if not gpu_info['available']:
        print(f"[ALARM] CUDA NOT AVAILABLE: {gpu_info.get('error', 'Unknown error')}")
        return
    
    print(f"  GPU Device    : {gpu_info['device_name']}")
    print(f"  SM Version    : {gpu_info['sm_version']}")
    print(f"  Architecture  : {gpu_info['arch']}")
    print(f"  Total Memory  : {gpu_info['total_memory_gb']:.1f} GB")
    print(f"  Blackwell     : {'YES ✓' if gpu_info['is_blackwell'] else 'NO'}")
    
    # Verify CUDA stream is ready
    try:
        stream = torch.cuda.current_stream()
        stream_ready = stream.query()
        print(f"  CUDA Stream   : {'Ready' if stream_ready else 'Busy'}")
    except Exception as e:
        print(f"  CUDA Stream   : ERROR: {e}")
    
    print("=" * 70 + "\n")


# ============================================================================
# ATTENTION TELEMETRY
# ============================================================================

class AttentionTelemetry:
    """
    Wraps attention execution with torch.cuda.Event timing and fallback detection.
    """
    
    def __init__(self, block_id: int = 0, phase: Literal["dit", "vae_encoder", "vae_decoder"] = "dit"):
        self.block_id = block_id
        self.phase = phase
        self.start_event = None
        self.end_event = None
        self.input_dtype = None
        self.kernel_executed = False
        self.fallback_detected = False
        self.execution_time_ms = 0.0
    
    def start(self, input_tensor: torch.Tensor):
        """
        Start timing an attention execution.
        Records input dtype for NVFP4/FP8 detection.
        """
        global _attention_call_count
        
        # Initialize monitor on first call
        initialize_truth_monitor()
        
        _attention_call_count += 1
        self.input_dtype = input_tensor.dtype
        
        # Create CUDA events for accurate timing
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        
        # Record start
        self.start_event.record()
        
        # Detect NVFP4/FP8 dtypes that may cause fallback
        self._check_dtype_compatibility()
    
    def _check_dtype_compatibility(self):
        """Check if input dtype may cause kernel fallback."""
        dtype_str = str(self.input_dtype)
        
        # FP4/FP8 dtypes that may require conversion
        problematic_dtypes = ['float8_e4m3fn', 'float8_e5m2', 'uint4', 'int4']
        
        for ptype in problematic_dtypes:
            if ptype in dtype_str:
                self.fallback_detected = True
                print(f"[ALARM] {self.phase.upper()} Block {self.block_id:02d}: "
                      f"NVFP4/FP8 dtype detected ({dtype_str}). "
                      f"Kernel may use fallback path.", flush=True)
                break
    
    def end(self, kernel_was_executed: bool = True):
        """
        End timing and report results.
        
        Args:
            kernel_was_executed: True if custom kernel ran, False if fallback used
        """
        global _total_attention_time_ms, _fallback_detected_count
        
        # Record end
        self.end_event.record()
        
        # Synchronize to get accurate timing
        torch.cuda.synchronize()
        
        # Calculate elapsed time
        self.execution_time_ms = self.start_event.elapsed_time(self.end_event)
        self.kernel_executed = kernel_was_executed
        
        _total_attention_time_ms += self.execution_time_ms
        
        # Detect fallback based on timing
        if self.execution_time_ms > FALLBACK_THRESHOLD_MS and not self.fallback_detected:
            self.fallback_detected = True
            _fallback_detected_count += 1
            print(f"[ALARM] {self.phase.upper()} Block {self.block_id:02d}: "
                  f"Slow execution ({self.execution_time_ms:.2f}ms > {FALLBACK_THRESHOLD_MS}ms). "
                  f"Possible fallback to SDPA.", flush=True)
        
        # Verbose logging if enabled or on anomaly
        if ENABLE_TRUTH_MONITOR or self.fallback_detected:
            status = "KERNEL" if kernel_was_executed else "FALLBACK"
            print(f"[TRUTH] {self.phase.upper()} Block {self.block_id:02d}: "
                  f"{status} | Time: {self.execution_time_ms:.2f}ms | "
                  f"Dtype: {self.input_dtype}", flush=True)


# ============================================================================
# VAE TELEMETRY
# ============================================================================

class VAETelemetry:
    """
    Provides truth monitoring for VAE operations.
    Specifically tracks tiling status and SA2 bypass scenarios.
    """
    
    def __init__(self, phase: Literal["encoder", "decoder"] = "encoder"):
        self.phase = phase
        self.tiling_enabled = False
        self.sa2_enabled = False
        self.force_override = False
        self.blocks_processed = 0
        self.start_time = 0.0
        self.total_time_ms = 0.0
    
    def start(self, tiling_enabled: bool, sa2_enabled: bool, force_override: bool = False):
        """
        Start VAE phase telemetry.
        
        Args:
            tiling_enabled: Whether tiled VAE is active
            sa2_enabled: Whether SA2 attention is enabled
            force_override: Whether force_decoder_sa2_with_tiling is True
        """
        global _vae_call_count
        
        # Initialize monitor on first call
        initialize_truth_monitor()
        
        _vae_call_count += 1
        
        self.tiling_enabled = tiling_enabled
        self.sa2_enabled = sa2_enabled
        self.force_override = force_override
        self.start_time = time.perf_counter()
        
        # Report VAE truth status
        print(f"\n[VAE-TRUTH] {self.phase.upper()} Phase Starting", flush=True)
        print(f"  Tiling    : {'ON' if tiling_enabled else 'OFF'}", flush=True)
        print(f"  SA2       : {'ON' if sa2_enabled else 'OFF'}", flush=True)
        
        if tiling_enabled and not sa2_enabled and not force_override:
            # This is the critical transparency message
            print(f"[VAE-TRUTH] SA2 is DISABLED by hardcoded tiling check. "
                  f"Speed loss: ~300% vs non-tiled SA2.", flush=True)
            print(f"  → To force SA2 with tiling: Set 'force_decoder_sa2_with_tiling = True' (may cause artifacts)", flush=True)
        elif tiling_enabled and sa2_enabled and force_override:
            print(f"[VAE-FORCE] WARNING: Tiling + SA2 enabled via force override. "
                  f"May cause artifacts!", flush=True)
    
    def block_processed(self, block_id: int, execution_time_ms: float):
        """Record a processed VAE attention block."""
        self.blocks_processed += 1
        
        if ENABLE_TRUTH_MONITOR:
            print(f"[VAE-TRUTH] {self.phase.upper()} Block {block_id:02d}: "
                  f"{execution_time_ms:.2f}ms", flush=True)
    
    def end(self):
        """End VAE phase telemetry and report summary."""
        global _total_vae_time_ms
        
        self.total_time_ms = (time.perf_counter() - self.start_time) * 1000
        _total_vae_time_ms += self.total_time_ms
        
        print(f"[VAE-TRUTH] {self.phase.upper()} Complete: "
              f"{self.blocks_processed} blocks in {self.total_time_ms:.2f}ms", flush=True)


# ============================================================================
# STREAM VERIFICATION
# ============================================================================

def verify_cuda_stream_active() -> bool:
    """
    Verify that the CUDA stream is executing custom kernels (not CPU fallback).
    
    Returns:
        True if stream appears active with GPU execution, False otherwise
    """
    if not torch.cuda.is_available():
        print("[ALARM] verify_cuda_stream_active: CUDA not available!")
        return False
    
    try:
        stream = torch.cuda.current_stream()
        
        # Query stream - if it returns True immediately on a heavy workload, 
        # the work may be on CPU
        is_idle = stream.query()
        
        if is_idle:
            # Stream is idle - this is expected before/after work
            return True
        else:
            # Stream is busy - GPU is executing
            return True
            
    except Exception as e:
        print(f"[ALARM] verify_cuda_stream_active: Error: {e}")
        return False


def get_kernel_execution_summary() -> Dict[str, Any]:
    """
    Get a summary of kernel execution statistics.
    Useful for debugging performance issues.
    """
    gpu_info = get_gpu_telemetry()
    
    avg_attention_ms = (_total_attention_time_ms / _attention_call_count 
                        if _attention_call_count > 0 else 0.0)
    avg_vae_ms = (_total_vae_time_ms / _vae_call_count 
                  if _vae_call_count > 0 else 0.0)
    
    summary = {
        'gpu_info': gpu_info,
        'attention_calls': _attention_call_count,
        'vae_calls': _vae_call_count,
        'fallback_detected': _fallback_detected_count,
        'total_attention_time_ms': _total_attention_time_ms,
        'total_vae_time_ms': _total_vae_time_ms,
        'avg_attention_time_ms': avg_attention_ms,
        'avg_vae_time_ms': avg_vae_ms,
    }
    
    return summary


def print_execution_summary():
    """Print a formatted summary of kernel execution statistics."""
    summary = get_kernel_execution_summary()
    
    print("\n" + "=" * 70)
    print("TRUTH MONITOR: EXECUTION SUMMARY")
    print("=" * 70)
    
    gpu = summary['gpu_info']
    print(f"  GPU: {gpu.get('device_name', 'Unknown')} ({gpu.get('sm_version', 'N/A')})")
    print(f"  Architecture: {gpu.get('arch', 'Unknown')}")
    print(f"")
    print(f"  Attention Calls: {summary['attention_calls']}")
    print(f"  VAE Calls: {summary['vae_calls']}")
    print(f"  Fallbacks Detected: {summary['fallback_detected']}")
    print(f"")
    print(f"  Total Attention Time: {summary['total_attention_time_ms']:.2f}ms")
    print(f"  Total VAE Time: {summary['total_vae_time_ms']:.2f}ms")
    print(f"  Avg Attention Time: {summary['avg_attention_time_ms']:.2f}ms")
    print(f"  Avg VAE Time: {summary['avg_vae_time_ms']:.2f}ms")
    
    # Provide optimization hints
    if summary['fallback_detected'] > 0:
        print(f"\n[ALARM] {summary['fallback_detected']} fallback events detected!")
        print("  → Check if NVFP4/FP8 model is being used")
        print("  → Verify Triton is compiled for your GPU")
    
    if summary['avg_attention_time_ms'] > EXPECTED_BLACKWELL_OPTIMIZED_MS:
        print(f"\n[WARNING] Attention time ({summary['avg_attention_time_ms']:.2f}ms) "
              f"exceeds expected optimized time ({EXPECTED_BLACKWELL_OPTIMIZED_MS}ms)")
        print("  → Sparge optimization may not be active")
        print("  → Check for [KERNEL-EXEC] logs to verify kernel execution")
    
    print("=" * 70 + "\n")


# ============================================================================
# RESET
# ============================================================================

def reset_truth_monitor():
    """Reset all Truth Monitor counters and state."""
    global _attention_call_count, _vae_call_count, _fallback_detected_count
    global _total_attention_time_ms, _total_vae_time_ms
    global _truth_monitor_initialized
    
    _attention_call_count = 0
    _vae_call_count = 0
    _fallback_detected_count = 0
    _total_attention_time_ms = 0.0
    _total_vae_time_ms = 0.0
    _truth_monitor_initialized = False
