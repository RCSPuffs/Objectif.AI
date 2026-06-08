"""
Model registry for Objectif.AI.
Defines all supported models, their metadata, download sources, and hardware recommendations.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelInfo:
    id: str                          # Unique identifier, used as filename base
    name: str                        # Display name
    family: str                      # e.g. "YOLOv11", "RT-DETR", "ONNX"
    variant: str                     # e.g. "nano", "small", "medium", "large", "xlarge"
    description: str                 # Human-readable description
    best_hardware: list              # e.g. ["cpu"], ["cuda"], ["cuda", "openvino"]
    speed_rating: int                # 1-5, 5 = fastest
    accuracy_rating: int             # 1-5, 5 = most accurate
    size_mb: float                   # Approximate download size in MB
    download_url: str                # Direct download URL
    engine: str                      # "ultralytics", "onnx", "torchvision", "torchhub"
    filename: str                    # Local filename in models/
    input_size: int                  # Default input resolution (square)
    notes: str = ""                  # Extra notes (e.g. "not tested on AMD")
    classes: int = 80                # Number of detection classes
    experimental: bool = False       # Mark as experimental/untested
    sha256: str = ""                 # Expected SHA-256 hex digest (empty = skip check)


# ---------------------------------------------------------------------------
# YOLO v11 family  (Ultralytics)
# ---------------------------------------------------------------------------
YOLO11_BASE = (
    "YOLOv11 is Ultralytics' latest generation model released in 2024. "
    "It improves on v8/v9 with better efficiency and accuracy tradeoffs."
)

YOLO11_MODELS = [
    ModelInfo(
        id="yolo11n", name="YOLO v11 Nano", family="YOLOv11", variant="nano",
        description=f"{YOLO11_BASE} Nano is the smallest and fastest variant — "
                    "ideal for older CPUs or low-power systems where speed matters more than accuracy.",
        best_hardware=["cpu", "cuda", "openvino"],
        speed_rating=5, accuracy_rating=2, size_mb=5.4,
        download_url="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
        engine="ultralytics", filename="yolo11n.pt", input_size=640,
    ),
    ModelInfo(
        id="yolo11s", name="YOLO v11 Small", family="YOLOv11", variant="small",
        description=f"{YOLO11_BASE} Small balances speed and accuracy — "
                    "a solid all-round choice for CPU or entry-level GPU.",
        best_hardware=["cpu", "cuda", "openvino"],
        speed_rating=4, accuracy_rating=3, size_mb=18.4,
        download_url="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt",
        engine="ultralytics", filename="yolo11s.pt", input_size=640,
    ),
    ModelInfo(
        id="yolo11m", name="YOLO v11 Medium", family="YOLOv11", variant="medium",
        description=f"{YOLO11_BASE} Medium offers strong accuracy with moderate GPU requirements. "
                    "Recommended for NVIDIA GTX 1060 / RTX class cards.",
        best_hardware=["cuda", "openvino"],
        speed_rating=3, accuracy_rating=4, size_mb=38.8,
        download_url="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt",
        engine="ultralytics", filename="yolo11m.pt", input_size=640,
    ),
    ModelInfo(
        id="yolo11l", name="YOLO v11 Large", family="YOLOv11", variant="large",
        description=f"{YOLO11_BASE} Large provides high accuracy — best on a dedicated NVIDIA GPU with 6GB+ VRAM.",
        best_hardware=["cuda"],
        speed_rating=2, accuracy_rating=5, size_mb=49.0,
        download_url="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt",
        engine="ultralytics", filename="yolo11l.pt", input_size=640,
    ),
    ModelInfo(
        id="yolo11x", name="YOLO v11 XLarge", family="YOLOv11", variant="xlarge",
        description=f"{YOLO11_BASE} XLarge is the most accurate YOLO v11 model. "
                    "Requires a powerful NVIDIA GPU (RTX 3070+) for real-time use.",
        best_hardware=["cuda"],
        speed_rating=1, accuracy_rating=5, size_mb=109.3,
        download_url="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt",
        engine="ultralytics", filename="yolo11x.pt", input_size=640,
    ),
]

# ---------------------------------------------------------------------------
# YOLO v8 family  (Ultralytics)
# ---------------------------------------------------------------------------
YOLO8_BASE = (
    "YOLOv8 is a mature, battle-tested Ultralytics model with excellent community support. "
    "Highly compatible across all hardware including Intel OpenVINO."
)

YOLO8_MODELS = [
    ModelInfo(
        id="yolov8n", name="YOLO v8 Nano", family="YOLOv8", variant="nano",
        description=f"{YOLO8_BASE} Nano is extremely fast — great for weak CPUs and Raspberry Pi class hardware.",
        best_hardware=["cpu", "cuda", "openvino"],
        speed_rating=5, accuracy_rating=2, size_mb=6.2,
        download_url="https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt",
        engine="ultralytics", filename="yolov8n.pt", input_size=640,
    ),
    ModelInfo(
        id="yolov8s", name="YOLO v8 Small", family="YOLOv8", variant="small",
        description=f"{YOLO8_BASE} Small is a good starting point for most systems.",
        best_hardware=["cpu", "cuda", "openvino"],
        speed_rating=4, accuracy_rating=3, size_mb=21.5,
        download_url="https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt",
        engine="ultralytics", filename="yolov8s.pt", input_size=640,
    ),
    ModelInfo(
        id="yolov8m", name="YOLO v8 Medium", family="YOLOv8", variant="medium",
        description=f"{YOLO8_BASE} Medium — best on a mid-range NVIDIA GPU.",
        best_hardware=["cuda", "openvino"],
        speed_rating=3, accuracy_rating=4, size_mb=49.7,
        download_url="https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt",
        engine="ultralytics", filename="yolov8m.pt", input_size=640,
    ),
    ModelInfo(
        id="yolov8l", name="YOLO v8 Large", family="YOLOv8", variant="large",
        description=f"{YOLO8_BASE} Large — needs 6GB+ VRAM for smooth real-time operation.",
        best_hardware=["cuda"],
        speed_rating=2, accuracy_rating=4, size_mb=83.7,
        download_url="https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l.pt",
        engine="ultralytics", filename="yolov8l.pt", input_size=640,
    ),
    ModelInfo(
        id="yolov8x", name="YOLO v8 XLarge", family="YOLOv8", variant="xlarge",
        description=f"{YOLO8_BASE} XLarge — maximum accuracy, requires RTX 3070+ or equivalent.",
        best_hardware=["cuda"],
        speed_rating=1, accuracy_rating=5, size_mb=130.5,
        download_url="https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x.pt",
        engine="ultralytics", filename="yolov8x.pt", input_size=640,
    ),
]

# ---------------------------------------------------------------------------
# YOLO v9 family  (Ultralytics)
# ---------------------------------------------------------------------------
YOLO9_BASE = (
    "YOLOv9 introduces Programmable Gradient Information (PGI) for improved learning efficiency. "
    "Strong accuracy gains over v8 with similar speed."
)

YOLO9_MODELS = [
    ModelInfo(
        id="yolov9c", name="YOLO v9 Compact", family="YOLOv9", variant="compact",
        description=f"{YOLO9_BASE} Compact variant — good balance of speed and accuracy on mid-range GPU.",
        best_hardware=["cuda", "openvino"],
        speed_rating=3, accuracy_rating=4, size_mb=51.0,
        download_url="https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov9c.pt",
        engine="ultralytics", filename="yolov9c.pt", input_size=640,
    ),
    ModelInfo(
        id="yolov9e", name="YOLO v9 Extended", family="YOLOv9", variant="extended",
        description=f"{YOLO9_BASE} Extended — highest accuracy in the v9 family, requires powerful NVIDIA GPU.",
        best_hardware=["cuda"],
        speed_rating=2, accuracy_rating=5, size_mb=117.0,
        download_url="https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov9e.pt",
        engine="ultralytics", filename="yolov9e.pt", input_size=640,
    ),
]

# ---------------------------------------------------------------------------
# YOLO v10 family  (Ultralytics)
# ---------------------------------------------------------------------------
YOLO10_BASE = (
    "YOLOv10 removes the NMS post-processing step for faster end-to-end detection. "
    "Excellent for latency-sensitive deployments."
)

YOLO10_MODELS = [
    ModelInfo(
        id="yolov10n", name="YOLO v10 Nano", family="YOLOv10", variant="nano",
        description=f"{YOLO10_BASE} Nano — fastest v10 variant, suitable for CPU and low-end GPU.",
        best_hardware=["cpu", "cuda", "openvino"],
        speed_rating=5, accuracy_rating=2, size_mb=5.7,
        download_url="https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10n.pt",
        engine="ultralytics", filename="yolov10n.pt", input_size=640,
    ),
    ModelInfo(
        id="yolov10s", name="YOLO v10 Small", family="YOLOv10", variant="small",
        description=f"{YOLO10_BASE} Small — good for CPU with better accuracy than nano.",
        best_hardware=["cpu", "cuda", "openvino"],
        speed_rating=4, accuracy_rating=3, size_mb=15.8,
        download_url="https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10s.pt",
        engine="ultralytics", filename="yolov10s.pt", input_size=640,
    ),
    ModelInfo(
        id="yolov10m", name="YOLO v10 Medium", family="YOLOv10", variant="medium",
        description=f"{YOLO10_BASE} Medium — recommended for NVIDIA GTX 1060+ class GPUs.",
        best_hardware=["cuda", "openvino"],
        speed_rating=3, accuracy_rating=4, size_mb=31.9,
        download_url="https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10m.pt",
        engine="ultralytics", filename="yolov10m.pt", input_size=640,
    ),
    ModelInfo(
        id="yolov10l", name="YOLO v10 Large", family="YOLOv10", variant="large",
        description=f"{YOLO10_BASE} Large — high accuracy, needs 6GB+ VRAM.",
        best_hardware=["cuda"],
        speed_rating=2, accuracy_rating=4, size_mb=49.0,
        download_url="https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10l.pt",
        engine="ultralytics", filename="yolov10l.pt", input_size=640,
    ),
    ModelInfo(
        id="yolov10x", name="YOLO v10 XLarge", family="YOLOv10", variant="xlarge",
        description=f"{YOLO10_BASE} XLarge — maximum v10 accuracy, RTX 3070+ recommended.",
        best_hardware=["cuda"],
        speed_rating=1, accuracy_rating=5, size_mb=56.9,
        download_url="https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10x.pt",
        engine="ultralytics", filename="yolov10x.pt", input_size=640,
    ),
]

# ---------------------------------------------------------------------------
# YOLO v26 family  (Ultralytics — latest generation)
# ---------------------------------------------------------------------------
YOLO26_BASE = (
    "YOLOv26 is the newest generation YOLO architecture with significant architecture improvements "
    "over previous versions. Delivers state-of-the-art accuracy and speed. "
    "Recommended for new deployments where maximum performance is desired."
)

YOLO26_MODELS = [
    ModelInfo(
        id="yolo26n", name="YOLO v26 Nano", family="YOLOv26", variant="nano",
        description=f"{YOLO26_BASE} Nano — fastest v26 variant, works well on modern CPUs and Intel Arc.",
        best_hardware=["cpu", "cuda", "openvino"],
        speed_rating=5, accuracy_rating=3, size_mb=8.0,
        download_url="https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt",
        engine="ultralytics", filename="yolo26n.pt", input_size=640,
        experimental=True, notes="Verify download URL at ultralytics.com/models",
    ),
    ModelInfo(
        id="yolo26s", name="YOLO v26 Small", family="YOLOv26", variant="small",
        description=f"{YOLO26_BASE} Small — excellent CPU/entry GPU performance with strong accuracy.",
        best_hardware=["cpu", "cuda", "openvino"],
        speed_rating=4, accuracy_rating=3, size_mb=22.0,
        download_url="https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt",
        engine="ultralytics", filename="yolo26s.pt", input_size=640,
        experimental=True, notes="Verify download URL at ultralytics.com/models",
    ),
    ModelInfo(
        id="yolo26m", name="YOLO v26 Medium", family="YOLOv26", variant="medium",
        description=f"{YOLO26_BASE} Medium — strong all-round model for NVIDIA GTX 1060+ GPUs.",
        best_hardware=["cuda", "openvino"],
        speed_rating=3, accuracy_rating=4, size_mb=52.0,
        download_url="https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt",
        engine="ultralytics", filename="yolo26m.pt", input_size=640,
        experimental=True, notes="Verify download URL at ultralytics.com/models",
    ),
    ModelInfo(
        id="yolo26l", name="YOLO v26 Large", family="YOLOv26", variant="large",
        description=f"{YOLO26_BASE} Large — high accuracy variant requiring 6GB+ VRAM.",
        best_hardware=["cuda"],
        speed_rating=2, accuracy_rating=5, size_mb=85.0,
        download_url="https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26l.pt",
        engine="ultralytics", filename="yolo26l.pt", input_size=640,
        experimental=True, notes="Verify download URL at ultralytics.com/models",
    ),
    ModelInfo(
        id="yolo26x", name="YOLO v26 XLarge", family="YOLOv26", variant="xlarge",
        description=f"{YOLO26_BASE} XLarge — maximum accuracy. Requires RTX 3070+ or equivalent. "
                    "Not suitable for real-time use on CPU.",
        best_hardware=["cuda"],
        speed_rating=1, accuracy_rating=5, size_mb=130.0,
        download_url="https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x.pt",
        engine="ultralytics", filename="yolo26x.pt", input_size=640,
        experimental=True, notes="Verify download URL at ultralytics.com/models",
    ),
]

# ---------------------------------------------------------------------------
# RT-DETR family  (Ultralytics — transformer-based)
# ---------------------------------------------------------------------------
RTDETR_BASE = (
    "RT-DETR (Real-Time Detection Transformer) is a transformer-based detector from Baidu. "
    "Eliminates NMS post-processing entirely. Particularly strong on complex scenes with many overlapping objects. "
    "Best on NVIDIA GPU — transformer architecture is slow on CPU."
)

RTDETR_MODELS = [
    ModelInfo(
        id="rtdetr-l", name="RT-DETR Large", family="RT-DETR", variant="large",
        description=f"{RTDETR_BASE} Large — good accuracy/speed balance for RTX class cards.",
        best_hardware=["cuda"],
        speed_rating=3, accuracy_rating=4, size_mb=67.0,
        download_url="https://github.com/ultralytics/assets/releases/download/v8.2.0/rtdetr-l.pt",
        engine="ultralytics", filename="rtdetr-l.pt", input_size=640,
    ),
    ModelInfo(
        id="rtdetr-x", name="RT-DETR XLarge", family="RT-DETR", variant="xlarge",
        description=f"{RTDETR_BASE} XLarge — highest RT-DETR accuracy. RTX 3080+ recommended.",
        best_hardware=["cuda"],
        speed_rating=2, accuracy_rating=5, size_mb=143.0,
        download_url="https://github.com/ultralytics/assets/releases/download/v8.2.0/rtdetr-x.pt",
        engine="ultralytics", filename="rtdetr-x.pt", input_size=640,
    ),
]

# ---------------------------------------------------------------------------
# ONNX models  (engine="onnx")
# ---------------------------------------------------------------------------
ONNX_MODELS = [
    ModelInfo(
        id="mobilenet-ssd", name="MobileNet SSD", family="MobileNet", variant="v2",
        description=(
            "MobileNet SSD v2 is an extremely lightweight detector designed for mobile and edge devices. "
            "Very fast on CPU — ideal for older hardware, NAS boxes, or systems without a GPU. "
            "Lower accuracy than YOLO but uses a fraction of the resources. "
            "Best hardware: CPU or any GPU via ONNX Runtime. "
            "Manual install: place mobilenet-ssd.onnx in the models/ folder."
        ),
        best_hardware=["cpu", "openvino"],
        speed_rating=5, accuracy_rating=2, size_mb=19.0,
        download_url="",  # No reliable public direct-download URL — manual install
        engine="onnx", filename="mobilenet-ssd.onnx", input_size=300, classes=21,
        notes="Manual install only — no reliable direct download URL exists. Place the .onnx file in the models/ folder.",
    ),
    ModelInfo(
        id="efficientdet-d0", name="EfficientDet D0", family="EfficientDet", variant="d0",
        description=(
            "EfficientDet D0 is Google's smallest EfficientDet model. "
            "Very efficient on CPU — uses compound scaling for better accuracy/speed tradeoff than MobileNet. "
            "Good choice for systems with a modern CPU but no discrete GPU. "
            "Manual install: place efficientdet-d0.onnx in the models/ folder."
        ),
        best_hardware=["cpu", "openvino", "cuda"],
        speed_rating=4, accuracy_rating=3, size_mb=15.0,
        download_url="",  # No reliable public direct-download URL — manual install
        engine="onnx", filename="efficientdet-d0.onnx", input_size=512,
        notes="Manual install only — no reliable direct download URL exists. Place the .onnx file in the models/ folder.",
    ),
    ModelInfo(
        id="efficientdet-d1", name="EfficientDet D1", family="EfficientDet", variant="d1",
        description=(
            "EfficientDet D1 is one step up from D0 — better accuracy with modest extra compute. "
            "Good on a mid-range CPU or entry-level GPU via ONNX Runtime. "
            "Manual install: place efficientdet-d1.onnx in the models/ folder."
        ),
        best_hardware=["cpu", "cuda", "openvino"],
        speed_rating=3, accuracy_rating=3, size_mb=21.0,
        download_url="",  # No reliable public direct-download URL — manual install
        engine="onnx", filename="efficientdet-d1.onnx", input_size=640,
        notes="Manual install only — no reliable direct download URL exists. Place the .onnx file in the models/ folder.",
    ),
]

# ---------------------------------------------------------------------------
# YOLO v5 family  (torch.hub)
# ---------------------------------------------------------------------------
YOLO5_BASE = (
    "YOLOv5 is the highly mature and battle-tested original from Ultralytics. "
    "First-run downloads automatically via torch.hub (~first load only). "
    "Excellent compatibility and community support. "
    "Runs via torch.hub — model cached to the models/ folder on first load."
)

YOLO5_MODELS = [
    ModelInfo(
        id="yolov5n", name="YOLO v5 Nano", family="YOLOv5", variant="nano",
        description=f"{YOLO5_BASE} Nano — the smallest and fastest v5 variant. Great for CPU and weak GPU.",
        best_hardware=["cpu", "cuda", "openvino"],
        speed_rating=5, accuracy_rating=2, size_mb=4.0,
        download_url="https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt",
        engine="ultralytics", filename="yolov5n.pt", input_size=640,
    ),
    ModelInfo(
        id="yolov5s", name="YOLO v5 Small", family="YOLOv5", variant="small",
        description=f"{YOLO5_BASE} Small — good all-rounder for CPU or entry GPU.",
        best_hardware=["cpu", "cuda", "openvino"],
        speed_rating=4, accuracy_rating=3, size_mb=14.0,
        download_url="https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt",
        engine="ultralytics", filename="yolov5s.pt", input_size=640,
    ),
    ModelInfo(
        id="yolov5m", name="YOLO v5 Medium", family="YOLOv5", variant="medium",
        description=f"{YOLO5_BASE} Medium — solid accuracy, best on a mid-range NVIDIA GPU.",
        best_hardware=["cuda", "openvino"],
        speed_rating=3, accuracy_rating=4, size_mb=42.0,
        download_url="https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m.pt",
        engine="ultralytics", filename="yolov5m.pt", input_size=640,
    ),
    ModelInfo(
        id="yolov5l", name="YOLO v5 Large", family="YOLOv5", variant="large",
        description=f"{YOLO5_BASE} Large — high accuracy, needs 6GB+ VRAM for real-time use.",
        best_hardware=["cuda"],
        speed_rating=2, accuracy_rating=4, size_mb=92.0,
        download_url="https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5l.pt",
        engine="ultralytics", filename="yolov5l.pt", input_size=640,
    ),
    ModelInfo(
        id="yolov5x", name="YOLO v5 XLarge", family="YOLOv5", variant="xlarge",
        description=f"{YOLO5_BASE} XLarge — maximum v5 accuracy. Requires RTX 3070+ for real-time.",
        best_hardware=["cuda"],
        speed_rating=1, accuracy_rating=5, size_mb=178.0,
        download_url="https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5x.pt",
        engine="ultralytics", filename="yolov5x.pt", input_size=640,
    ),
]

# ---------------------------------------------------------------------------
# Torchvision detection models
# ---------------------------------------------------------------------------
TV_BASE = (
    "Part of PyTorch\'s torchvision library — already installed with PyTorch, no separate download needed. "
    "Downloads model weights automatically on first load (~170 MB) and caches to the models/ folder. "
    "COCO-trained, 91 classes. Supports CPU and NVIDIA CUDA."
)

TORCHVISION_MODELS = [
    ModelInfo(
        id="fasterrcnn-resnet50", name="Faster R-CNN ResNet-50", family="Torchvision", variant="resnet50",
        description=(
            "Faster R-CNN is a classic two-stage detector with excellent accuracy on complex scenes. "
            "Slower than YOLO but often catches objects that single-stage detectors miss. "
            f"{TV_BASE} Best on NVIDIA GPU — slow on CPU."
        ),
        best_hardware=["cuda", "cpu"],
        speed_rating=2, accuracy_rating=5, size_mb=170.0,
        download_url="torchvision:fasterrcnn_resnet50_fpn",
        engine="torchvision", filename="fasterrcnn_resnet50_fpn_coco.pth", input_size=800,
        notes="Two-stage detector — higher latency than YOLO but strong accuracy.",
    ),
    ModelInfo(
        id="fasterrcnn-mobilenet", name="Faster R-CNN MobileNet", family="Torchvision", variant="mobilenet",
        description=(
            "A lighter version of Faster R-CNN using MobileNetV3 as the backbone instead of ResNet-50. "
            "Much faster than the full Faster R-CNN while keeping the two-stage accuracy advantage. "
            f"{TV_BASE} Good CPU performance."
        ),
        best_hardware=["cpu", "cuda"],
        speed_rating=3, accuracy_rating=4, size_mb=77.0,
        download_url="torchvision:fasterrcnn_mobilenet_v3_large_fpn",
        engine="torchvision", filename="fasterrcnn_mobilenet_v3_large_fpn_coco.pth", input_size=800,
    ),
    ModelInfo(
        id="ssd300-vgg16", name="SSD300 VGG16", family="Torchvision", variant="vgg16",
        description=(
            "Single Shot Detector with VGG16 backbone. Fast single-stage detection — a reliable CPU option. "
            "Good speed/accuracy balance for systems without a powerful GPU. "
            f"{TV_BASE}"
        ),
        best_hardware=["cpu", "cuda"],
        speed_rating=4, accuracy_rating=3, size_mb=135.0,
        download_url="torchvision:ssd300_vgg16",
        engine="torchvision", filename="ssd300_vgg16_coco.pth", input_size=300,
    ),
    ModelInfo(
        id="ssdlite-mobilenet", name="SSD Lite MobileNet", family="Torchvision", variant="mobilenet",
        description=(
            "Lightweight SSD with MobileNetV3 backbone. Very fast on CPU — "
            "the torchvision equivalent of MobileNet SSD but with reliable output format. "
            f"{TV_BASE} Best CPU-only option in this family."
        ),
        best_hardware=["cpu", "cuda"],
        speed_rating=5, accuracy_rating=2, size_mb=56.0,
        download_url="torchvision:ssdlite320_mobilenet_v3_large",
        engine="torchvision", filename="ssdlite320_mobilenet_v3_large_coco.pth", input_size=320,
    ),
    ModelInfo(
        id="retinanet-resnet50", name="RetinaNet ResNet-50", family="Torchvision", variant="resnet50",
        description=(
            "RetinaNet uses Focal Loss to handle the foreground/background class imbalance problem. "
            "Strong at detecting small objects — useful for distant cameras or crowded scenes. "
            f"{TV_BASE} Best on NVIDIA GPU."
        ),
        best_hardware=["cuda", "cpu"],
        speed_rating=2, accuracy_rating=4, size_mb=145.0,
        download_url="torchvision:retinanet_resnet50_fpn",
        engine="torchvision", filename="retinanet_resnet50_fpn_coco.pth", input_size=800,
        notes="Particularly good at detecting small or distant objects.",
    ),
]

# ---------------------------------------------------------------------------
# Master registry
# ---------------------------------------------------------------------------
ALL_MODELS: list[ModelInfo] = (
    YOLO26_MODELS +
    YOLO11_MODELS +
    YOLO10_MODELS +
    YOLO9_MODELS +
    YOLO8_MODELS +
    YOLO5_MODELS +
    RTDETR_MODELS +
    TORCHVISION_MODELS +
    ONNX_MODELS
)

MODEL_REGISTRY: dict[str, ModelInfo] = {m.id: m for m in ALL_MODELS}


def get_model(model_id: str) -> Optional[ModelInfo]:
    return MODEL_REGISTRY.get(model_id)


def get_all_models() -> list[dict]:
    """Return all models serialized as dicts for the API."""
    return [model_to_dict(m) for m in ALL_MODELS]


def model_to_dict(m: ModelInfo) -> dict:
    return {
        "id": m.id,
        "name": m.name,
        "family": m.family,
        "variant": m.variant,
        "description": m.description,
        "best_hardware": m.best_hardware,
        "speed_rating": m.speed_rating,
        "accuracy_rating": m.accuracy_rating,
        "size_mb": m.size_mb,
        "engine": m.engine,
        "filename": m.filename,
        "input_size": m.input_size,
        "classes": m.classes,
        "notes": m.notes,
        "experimental": m.experimental,
        "sha256": m.sha256,
        "download_url": m.download_url,
    }
