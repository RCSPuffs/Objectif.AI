"""
Dependency checker for Objectif.AI.
Checks all required and optional packages, returns structured status.
"""

import sys
import logging
import importlib
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Full dependency catalog
# ---------------------------------------------------------------------------
# Each entry:
#   id          -- unique key
#   name        -- display name
#   package     -- pip package name (for install)
#   import_name -- Python import name (may differ from package name)
#   category    -- grouping label
#   description -- what it does in plain English
#   required    -- True = app won't work without it
#   size_mb     -- approximate download size (0 = small/unknown)
#   install_cmd -- list of args to pass to pip (None = use [package])
#   manual_note -- shown instead of install button if set
#   min_version -- minimum acceptable version string (None = any)

DEPENDENCY_CATALOG = [
    # ── Core server ─────────────────────────────────────────────────────────
    {
        "id": "fastapi",
        "name": "FastAPI",
        "package": "fastapi",
        "import_name": "fastapi",
        "category": "Core Server",
        "description": "Web framework that serves the dashboard and BlueIris API endpoint.",
        "required": True,
        "size_mb": 0,
        "install_cmd": None,
        "manual_note": None,
        "min_version": "0.110.0",
    },
    {
        "id": "uvicorn",
        "name": "Uvicorn",
        "package": "uvicorn",
        "import_name": "uvicorn",
        "category": "Core Server",
        "description": "ASGI server that runs the FastAPI application.",
        "required": True,
        "size_mb": 0,
        "install_cmd": None,
        "manual_note": None,
        "min_version": None,
    },
    {
        "id": "pyyaml",
        "name": "PyYAML",
        "package": "pyyaml",
        "import_name": "yaml",
        "category": "Core Server",
        "description": "Reads and writes config.yaml to persist your settings.",
        "required": True,
        "size_mb": 0,
        "install_cmd": None,
        "manual_note": None,
        "min_version": None,
    },
    {
        "id": "python_multipart",
        "name": "python-multipart",
        "package": "python-multipart",
        "import_name": "multipart",
        "category": "Core Server",
        "description": "Parses image uploads from BlueIris (multipart/form-data).",
        "required": True,
        "size_mb": 0,
        "install_cmd": None,
        "manual_note": None,
        "min_version": None,
    },
    # ── Image processing ────────────────────────────────────────────────────
    {
        "id": "opencv",
        "name": "OpenCV",
        "package": "opencv-python",
        "import_name": "cv2",
        "category": "Image Processing",
        "description": "Decodes images from BlueIris and prepares them for the detection model.",
        "required": True,
        "size_mb": 30,
        "install_cmd": None,
        "manual_note": None,
        "min_version": None,
    },
    {
        "id": "numpy",
        "name": "NumPy",
        "package": "numpy",
        "import_name": "numpy",
        "category": "Image Processing",
        "description": "Array operations for image preprocessing and model output parsing.",
        "required": True,
        "size_mb": 15,
        "install_cmd": None,
        "manual_note": None,
        "min_version": None,
    },
    {
        "id": "pillow",
        "name": "Pillow",
        "package": "Pillow",
        "import_name": "PIL",
        "category": "Image Processing",
        "description": "Image handling library for processing camera frames.",
        "required": True,
        "size_mb": 4,
        "install_cmd": None,
        "manual_note": None,
        "min_version": None,
    },
    # ── Detection engines ───────────────────────────────────────────────────
    {
        "id": "ultralytics",
        "name": "Ultralytics",
        "package": "ultralytics",
        "import_name": "ultralytics",
        "category": "Detection Engine",
        "description": "Powers all YOLO v8/v9/v10/v11/v26 and RT-DETR models.",
        "required": True,
        "size_mb": 20,
        "install_cmd": None,
        "manual_note": None,
        "min_version": "8.3.0",
    },
    {
        "id": "torch_cpu",
        "name": "PyTorch (CPU)",
        "package": "torch",
        "import_name": "torch",
        "category": "Detection Engine",
        "description": "Deep learning framework. CPU build is the baseline — GPU builds replace this.",
        "required": True,
        "size_mb": 200,
        "install_cmd": None,
        "manual_note": None,
        "min_version": None,
    },
    {
        "id": "torchvision",
        "name": "Torchvision",
        "package": "torchvision",
        "import_name": "torchvision",
        "category": "Detection Engine",
        "description": "Provides Faster R-CNN, SSD, and RetinaNet detection models.",
        "required": False,
        "size_mb": 10,
        "install_cmd": None,
        "manual_note": None,
        "min_version": None,
    },
    {
        "id": "onnxruntime",
        "name": "ONNX Runtime (CPU)",
        "package": "onnxruntime",
        "import_name": "onnxruntime",
        "category": "Detection Engine",
        "description": "Runs ONNX format models on CPU. Also used as fallback by onnxruntime-gpu.",
        "required": False,
        "size_mb": 10,
        "install_cmd": ["onnxruntime"],
        "manual_note": None,
        "min_version": None,
    },
    {
        "id": "fast_alpr",
        "name": "fast-alpr (ALPR)",
        "package": "fast-alpr",
        "import_name": "fast_alpr",
        "category": "Detection Engine",
        "description": "License plate detection + OCR pipeline. Downloads its ONNX model weights on first use. Needed only if you use the ALPR endpoint.",
        "required": False,
        "size_mb": 5,
        "install_cmd": ["fast-alpr"],
        "manual_note": None,
        "min_version": None,
    },
    # ── YOLOv5 extras ───────────────────────────────────────────────────────
    {
        "id": "seaborn",
        "name": "Seaborn",
        "package": "seaborn",
        "import_name": "seaborn",
        "category": "YOLOv5 Support",
        "description": "Required by the YOLOv5 repository code (used internally for plotting). Not needed unless you load a YOLOv5 model.",
        "required": False,
        "size_mb": 2,
        "install_cmd": ["seaborn", "pandas", "matplotlib"],
        "manual_note": None,
        "min_version": None,
    },
    {
        "id": "pandas",
        "name": "Pandas",
        "package": "pandas",
        "import_name": "pandas",
        "category": "YOLOv5 Support",
        "description": "Required by the YOLOv5 repository code. Not needed unless you load a YOLOv5 model.",
        "required": False,
        "size_mb": 15,
        "install_cmd": None,  # installed with seaborn
        "manual_note": "Installed together with Seaborn.",
        "min_version": None,
    },
    {
        "id": "matplotlib",
        "name": "Matplotlib",
        "package": "matplotlib",
        "import_name": "matplotlib",
        "category": "YOLOv5 Support",
        "description": "Required by the YOLOv5 repository code. Not needed unless you load a YOLOv5 model.",
        "required": False,
        "size_mb": 20,
        "install_cmd": None,  # installed with seaborn
        "manual_note": "Installed together with Seaborn.",
        "min_version": None,
    },
    # ── NVIDIA CUDA ─────────────────────────────────────────────────────────
    {
        "id": "torch_cuda",
        "name": "PyTorch CUDA",
        "package": "torch",
        "import_name": "torch",
        "category": "NVIDIA GPU (CUDA)",
        "description": "GPU-accelerated PyTorch for NVIDIA cards (CC 7.5+). Replaces the CPU build. ~1.8 GB download.",
        "required": False,
        "size_mb": 1800,
        "install_cmd": None,
        "manual_note": "Use the Install button on the Hardware tab — it picks the correct CUDA version automatically.",
        "min_version": None,
    },
    {
        "id": "onnxruntime_gpu",
        "name": "ONNX Runtime GPU",
        "package": "onnxruntime-gpu",
        "import_name": "onnxruntime",
        "category": "NVIDIA GPU (CUDA)",
        "description": "GPU-accelerated ONNX Runtime for NVIDIA cards. Required for legacy GPU mode (CC < 7.5) and ONNX model inference on GPU.",
        "required": False,
        "size_mb": 120,
        "install_cmd": ["onnxruntime-gpu"],
        "manual_note": None,
        "min_version": None,
    },
    # ── Intel OpenVINO ──────────────────────────────────────────────────────
    {
        "id": "openvino",
        "name": "OpenVINO Runtime",
        "package": "openvino",
        "import_name": "openvino",
        "category": "Intel GPU (OpenVINO)",
        "description": "Intel's inference runtime for CPU and Intel GPU/iGPU acceleration.",
        "required": False,
        "size_mb": 150,
        "install_cmd": ["openvino", "onnxruntime-openvino"],
        "manual_note": None,
        "min_version": None,
    },
    {
        "id": "onnxruntime_openvino",
        "name": "ONNX Runtime OpenVINO",
        "package": "onnxruntime-openvino",
        "import_name": "onnxruntime",
        "category": "Intel GPU (OpenVINO)",
        "description": "ONNX Runtime execution provider for Intel OpenVINO backend.",
        "required": False,
        "size_mb": 10,
        "install_cmd": None,
        "manual_note": "Installed together with OpenVINO Runtime.",
        "min_version": None,
    },
    # ── DirectML (any DX12 GPU on Windows) ──────────────────────────────────
    {
        "id": "onnxruntime_directml",
        "name": "ONNX Runtime DirectML",
        "package": "onnxruntime-directml",
        "import_name": "onnxruntime",
        "category": "DirectML (Any GPU)",
        "description": "GPU-accelerated ONNX Runtime for any DirectX 12 GPU (NVIDIA, AMD, or Intel) on Windows. Mutually exclusive with the CPU and CUDA ONNX Runtime builds.",
        "required": False,
        "size_mb": 20,
        "install_cmd": None,
        "manual_note": "Use the Install button on the Hardware tab — it removes the conflicting ONNX Runtime builds first.",
        "min_version": None,
    },
    # ── AMD ROCm ────────────────────────────────────────────────────────────
    {
        "id": "torch_rocm",
        "name": "PyTorch ROCm",
        "package": "torch",
        "import_name": "torch",
        "category": "AMD GPU (ROCm)",
        "description": "GPU-accelerated PyTorch for AMD cards. ROCm on Windows is experimental and not officially tested.",
        "required": False,
        "size_mb": 1800,
        "install_cmd": None,
        "manual_note": "Manual install — see rocm.docs.amd.com. Not tested by Objectif.AI team.",
        "min_version": None,
    },
    # ── Hardware detection ──────────────────────────────────────────────────
    {
        "id": "psutil",
        "name": "psutil",
        "package": "psutil",
        "import_name": "psutil",
        "category": "Hardware Detection",
        "description": "Reads CPU core count and system RAM for the Hardware tab.",
        "required": False,
        "size_mb": 0,
        "install_cmd": None,
        "manual_note": None,
        "min_version": None,
    },
    {
        "id": "cpuinfo",
        "name": "py-cpuinfo",
        "package": "py-cpuinfo",
        "import_name": "cpuinfo",
        "category": "Hardware Detection",
        "description": "Reads the CPU model name for the Hardware tab.",
        "required": False,
        "size_mb": 0,
        "install_cmd": ["py-cpuinfo"],
        "manual_note": None,
        "min_version": None,
    },
]


# ---------------------------------------------------------------------------
# Check functions
# ---------------------------------------------------------------------------

def _get_version(import_name: str) -> Optional[str]:
    """Try to get the installed version of a package."""
    try:
        mod = importlib.import_module(import_name)
        for attr in ("__version__", "version", "VERSION"):
            v = getattr(mod, attr, None)
            if v and isinstance(v, str):
                return v
        # Try importlib.metadata
        try:
            from importlib.metadata import version as meta_version, PackageNotFoundError
            # Map import name to package name for metadata lookup
            pkg_map = {
                "cv2": "opencv-python",
                "PIL": "Pillow",
                "yaml": "PyYAML",
                "sklearn": "scikit-learn",
                "multipart": "python-multipart",
                "cpuinfo": "py-cpuinfo",
            }
            pkg_name = pkg_map.get(import_name, import_name)
            return meta_version(pkg_name)
        except Exception:
            return "installed"
    except ImportError:
        return None


def _check_torch_cuda() -> bool:
    """Returns True if torch is installed AND has CUDA support."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _check_torch_rocm() -> bool:
    """Returns True if torch has ROCm support."""
    try:
        import torch
        return bool(getattr(torch.version, "hip", None))
    except ImportError:
        return False


def _check_openvino_ort() -> bool:
    """Returns True if onnxruntime-openvino execution provider is available."""
    try:
        import onnxruntime as ort
        return "OpenVINOExecutionProvider" in ort.get_available_providers()
    except ImportError:
        return False


def _check_onnxruntime_gpu() -> bool:
    """Returns True if onnxruntime-gpu CUDA provider is available."""
    try:
        import onnxruntime as ort
        return "CUDAExecutionProvider" in ort.get_available_providers()
    except ImportError:
        return False


def _check_onnxruntime_directml() -> bool:
    """Returns True if the onnxruntime-directml provider is available."""
    try:
        import onnxruntime as ort
        return "DmlExecutionProvider" in ort.get_available_providers()
    except ImportError:
        return False


def check_all_dependencies() -> list:
    """
    Check all dependencies and return a list of status dicts.
    Each dict has all catalog fields plus:
      installed: bool
      version: str or None
      status: "ok" | "missing" | "no_cuda" | "manual"
      status_label: human readable
    """
    results = []

    for dep in DEPENDENCY_CATALOG:
        entry = dict(dep)
        dep_id = dep["id"]

        # Special checks for GPU-specific variants
        if dep_id == "torch_cuda":
            has_cuda = _check_torch_cuda()
            entry["installed"] = has_cuda
            entry["version"] = _get_version("torch") if has_cuda else None
            entry["status"] = "ok" if has_cuda else "missing"
            entry["status_label"] = "CUDA ready" if has_cuda else "Not installed / No CUDA"

        elif dep_id == "torch_rocm":
            has_rocm = _check_torch_rocm()
            entry["installed"] = has_rocm
            entry["version"] = _get_version("torch") if has_rocm else None
            entry["status"] = "ok" if has_rocm else "manual"
            entry["status_label"] = "ROCm ready" if has_rocm else "Manual install required"

        elif dep_id == "onnxruntime_gpu":
            ok = _check_onnxruntime_gpu()
            entry["installed"] = ok
            entry["version"] = _get_version("onnxruntime") if ok else None
            entry["status"] = "ok" if ok else "missing"
            entry["status_label"] = "GPU provider ready" if ok else "Not installed"

        elif dep_id == "onnxruntime_openvino":
            ok = _check_openvino_ort()
            entry["installed"] = ok
            entry["version"] = _get_version("onnxruntime") if ok else None
            entry["status"] = "ok" if ok else "missing"
            entry["status_label"] = "OpenVINO provider ready" if ok else "Not installed"

        elif dep_id == "onnxruntime_directml":
            ok = _check_onnxruntime_directml()
            entry["installed"] = ok
            entry["version"] = _get_version("onnxruntime") if ok else None
            entry["status"] = "ok" if ok else "manual"
            entry["status_label"] = "DirectML provider ready" if ok else "Not installed"

        elif dep_id == "torch_cpu":
            # torch_cpu shows as ok if torch is installed, regardless of CUDA
            v = _get_version("torch")
            entry["installed"] = v is not None
            entry["version"] = v
            # If CUDA is available, note it's actually a CUDA build
            if v and _check_torch_cuda():
                entry["status"] = "ok"
                entry["status_label"] = f"Installed (CUDA build — {v})"
            elif v:
                entry["status"] = "ok"
                entry["status_label"] = f"Installed (CPU build — {v})"
            else:
                entry["status"] = "missing"
                entry["status_label"] = "Not installed"

        else:
            v = _get_version(dep["import_name"])
            entry["installed"] = v is not None
            entry["version"] = v
            if dep["manual_note"] and v is None:
                entry["status"] = "manual"
                entry["status_label"] = "Not installed"
            elif v is not None:
                entry["status"] = "ok"
                entry["status_label"] = f"Installed{f' ({v})' if v != 'installed' else ''}"
            else:
                entry["status"] = "missing"
                entry["status_label"] = "Not installed"

        results.append(entry)

    return results


def get_python_info() -> dict:
    """Return Python version and executable path (basename only for privacy)."""
    import os
    return {
        "version": sys.version.split()[0],
        "executable_name": os.path.basename(sys.executable),
        "ok": sys.version_info >= (3, 10),
    }
