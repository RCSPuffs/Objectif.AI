"""
Hardware detection for Objectif.AI.
Detects available compute backends: NVIDIA CUDA, Intel GPU/OpenVINO, AMD ROCm, CPU.
"""

import logging
import platform
import subprocess
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class HardwareInfo:
    cpu_name: str = "Unknown CPU"
    cpu_cores: int = 0
    cpu_threads: int = 0
    ram_gb: float = 0.0

    has_nvidia: bool = False
    nvidia_name: str = ""
    nvidia_vram_gb: float = 0.0
    nvidia_driver: str = ""
    cuda_version: str = ""
    nvidia_compute_capability: str = ""   # e.g. "8.6"
    nvidia_legacy: bool = False           # True if CC < 7.5 (unsupported by modern PyTorch)

    has_intel_gpu: bool = False
    intel_gpu_name: str = ""
    has_openvino: bool = False
    openvino_version: str = ""

    has_amd_gpu: bool = False
    amd_gpu_name: str = ""
    has_rocm: bool = False
    rocm_version: str = ""

    has_directml: bool = False            # onnxruntime-directml provider present

    available_backends: list = field(default_factory=list)
    recommended_backend: str = "cpu"
    warnings: list = field(default_factory=list)


def detect_hardware() -> HardwareInfo:
    """Run full hardware detection and return HardwareInfo."""
    info = HardwareInfo()
    _detect_cpu(info)
    _detect_nvidia(info)
    _detect_intel(info)
    _detect_amd(info)
    _detect_directml(info)
    _build_backend_list(info)
    return info


def _detect_cpu(info: HardwareInfo):
    try:
        import psutil
        info.ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 1)
        info.cpu_cores = psutil.cpu_count(logical=False) or 0
        info.cpu_threads = psutil.cpu_count(logical=True) or 0
    except ImportError:
        logger.warning("psutil not available, CPU RAM info limited")

    try:
        import cpuinfo
        cpu = cpuinfo.get_cpu_info()
        info.cpu_name = cpu.get("brand_raw", "Unknown CPU")
    except Exception:
        # Fallback for Windows
        try:
            result = subprocess.run(
                ["wmic", "cpu", "get", "name"],
                capture_output=True, text=True, timeout=5
            )
            lines = [l.strip() for l in result.stdout.splitlines() if l.strip() and l.strip() != "Name"]
            if lines:
                info.cpu_name = lines[0]
        except Exception:
            info.cpu_name = platform.processor() or "Unknown CPU"


def _detect_nvidia(info: HardwareInfo):
    try:
        import torch
        if torch.cuda.is_available():
            info.has_nvidia = True
            info.nvidia_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            vram = props.total_memory
            info.nvidia_vram_gb = round(vram / (1024 ** 3), 1)
            info.cuda_version = torch.version.cuda or ""
            cc = f"{props.major}.{props.minor}"
            info.nvidia_compute_capability = cc
            # Modern PyTorch requires CC >= 7.5 (Turing+)
            if props.major < 7 or (props.major == 7 and props.minor < 5):
                info.nvidia_legacy = True
                info.warnings.append(
                    f"GPU compute capability {cc} is below the minimum (7.5) "
                    f"required by current PyTorch. CUDA acceleration will not be "
                    f"available. CPU will be used instead. "
                    f"Legacy GPU support (ONNX Runtime path) is planned for v0.7."
                )
            return
    except ImportError:
        pass

    # Try nvidia-smi as fallback
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.total,driver_version,compute_cap",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = [p.strip() for p in result.stdout.strip().split(",")]
            if len(parts) >= 3:
                info.has_nvidia = True
                info.nvidia_name = parts[0]
                try:
                    info.nvidia_vram_gb = round(float(parts[1]) / 1024, 1)
                except ValueError:
                    pass
                info.nvidia_driver = parts[2] if len(parts) > 2 else ""
                if len(parts) > 3 and parts[3]:
                    info.nvidia_compute_capability = parts[3]
                    try:
                        major = int(parts[3].split(".")[0])
                        minor = int(parts[3].split(".")[1]) if "." in parts[3] else 0
                        if major < 7 or (major == 7 and minor < 5):
                            info.nvidia_legacy = True
                            info.warnings.append(
                                f"GPU compute capability {parts[3]} is below the "
                                f"minimum (7.5) required by current PyTorch. "
                                f"CUDA acceleration will not be available — CPU "
                                f"will be used. Legacy GPU support planned for v0.7."
                            )
                    except Exception:
                        pass
    except Exception:
        pass


def _detect_intel(info: HardwareInfo):
    # Check for OpenVINO
    try:
        from openvino.runtime import Core
        core = Core()
        devices = core.available_devices
        info.has_openvino = True
        try:
            import openvino
            info.openvino_version = openvino.__version__
        except Exception:
            info.openvino_version = "unknown"

        for device in devices:
            if "GPU" in device:
                info.has_intel_gpu = True
                try:
                    info.intel_gpu_name = core.get_property(device, "FULL_DEVICE_NAME")
                except Exception:
                    info.intel_gpu_name = "Intel GPU"
                break
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"OpenVINO detection error: {e}")

    # Fallback: check for Intel GPU via WMI on Windows
    if not info.has_intel_gpu:
        try:
            result = subprocess.run(
                ["wmic", "path", "win32_VideoController", "get", "name"],
                capture_output=True, text=True, timeout=5
            )
            for line in result.stdout.splitlines():
                if "intel" in line.lower() and ("arc" in line.lower() or
                        "iris" in line.lower() or "uhd" in line.lower() or
                        "hd graphics" in line.lower()):
                    info.has_intel_gpu = True
                    info.intel_gpu_name = line.strip()
                    break
        except Exception:
            pass


def _detect_amd(info: HardwareInfo):
    # Check for ROCm via torch
    try:
        import torch
        if hasattr(torch, 'version') and hasattr(torch.version, 'hip') and torch.version.hip:
            info.has_rocm = True
            info.rocm_version = torch.version.hip
            if torch.cuda.is_available():  # ROCm exposes via CUDA API
                info.has_amd_gpu = True
                info.amd_gpu_name = torch.cuda.get_device_name(0)
            return
    except Exception:
        pass

    # Check for AMD GPU via WMI on Windows
    try:
        result = subprocess.run(
            ["wmic", "path", "win32_VideoController", "get", "name"],
            capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.splitlines():
            if "amd" in line.lower() or "radeon" in line.lower():
                info.has_amd_gpu = True
                info.amd_gpu_name = line.strip()
                break
    except Exception:
        pass

    if info.has_amd_gpu and not info.has_rocm:
        info.warnings.append(
            "AMD GPU detected but ROCm is not installed. "
            "ROCm on Windows is experimental — CPU will be used as fallback. "
            "See https://rocm.docs.amd.com for installation instructions."
        )


def _build_backend_list(info: HardwareInfo):
    backends = ["cpu"]

    if info.has_nvidia:
        backends.append("cuda")
        info.recommended_backend = "cuda"

    if info.has_intel_gpu and info.has_openvino:
        backends.append("openvino")
        if info.recommended_backend == "cpu":
            info.recommended_backend = "openvino"

    if info.has_directml:
        backends.append("directml")
        # DirectML is a good default for AMD/Intel GPUs when nothing better is set.
        if info.recommended_backend == "cpu" and (info.has_amd_gpu or info.has_intel_gpu):
            info.recommended_backend = "directml"

    if info.has_amd_gpu and info.has_rocm:
        backends.append("rocm")
        if info.recommended_backend == "cpu":
            info.recommended_backend = "rocm"
    elif info.has_amd_gpu and not info.has_directml:
        backends.append("amd_cpu_fallback")

    info.available_backends = backends


def _detect_directml(info: HardwareInfo):
    """Detect whether the onnxruntime-directml execution provider is available."""
    try:
        import onnxruntime as ort
        info.has_directml = "DmlExecutionProvider" in ort.get_available_providers()
    except Exception:
        info.has_directml = False


def hardware_info_to_dict(info: HardwareInfo) -> dict:
    """Serialize HardwareInfo to a JSON-serializable dict."""
    return {
        "cpu": {
            "name": info.cpu_name,
            "cores": info.cpu_cores,
            "threads": info.cpu_threads,
            "ram_gb": info.ram_gb,
        },
        "nvidia": {
            "available": info.has_nvidia,
            "name": info.nvidia_name,
            "vram_gb": info.nvidia_vram_gb,
            "driver": info.nvidia_driver,
            "cuda_version": info.cuda_version,
            "compute_capability": info.nvidia_compute_capability,
            "legacy": info.nvidia_legacy,
        },
        "intel": {
            "gpu_available": info.has_intel_gpu,
            "gpu_name": info.intel_gpu_name,
            "openvino_available": info.has_openvino,
            "openvino_version": info.openvino_version,
        },
        "amd": {
            "gpu_available": info.has_amd_gpu,
            "gpu_name": info.amd_gpu_name,
            "rocm_available": info.has_rocm,
            "rocm_version": info.rocm_version,
        },
        "directml": {
            "available": info.has_directml,
        },
        "backends": info.available_backends,
        "recommended_backend": info.recommended_backend,
        "warnings": info.warnings,
    }
