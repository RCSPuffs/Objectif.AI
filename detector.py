"""
Inference engine for Objectif.AI.
Handles model loading, backend selection, and running detections.
Supports Ultralytics (YOLO/RT-DETR), torch.hub (YOLOv5), torchvision, and ONNX Runtime engines.
"""

import logging
import time
import threading
from pathlib import Path
from typing import Optional

from model_registry import ModelInfo, get_model
from config import get_value

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


class DetectionResult:
    def __init__(self, detections: list, inference_ms: float, model_id: str):
        self.detections = detections        # list of dicts: label, confidence, x_min, y_min, x_max, y_max
        self.inference_ms = inference_ms
        self.model_id = model_id

    def to_cpai_response(self, min_confidence: float = 0.0, filter_enabled: bool = False,
                         allowed_classes: Optional[list] = None) -> dict:
        """
        Format as CodeProject.AI compatible response.
        BlueIris expects:
        {
            "success": true,
            "predictions": [
                {"label": "person", "confidence": 0.94, "x_min": 10, "y_min": 20, "x_max": 100, "y_max": 200},
                ...
            ],
            "count": 1,
            "inferenceMs": 43
        }
        """
        preds = []
        for d in self.detections:
            if d["confidence"] < min_confidence:
                continue
            if filter_enabled and allowed_classes and d["label"] not in allowed_classes:
                continue
            preds.append({
                "label": d["label"],
                "confidence": round(d["confidence"], 4),
                "x_min": int(d.get("x_min", 0)),
                "y_min": int(d.get("y_min", 0)),
                "x_max": int(d.get("x_max", 0)),
                "y_max": int(d.get("y_max", 0)),
            })

        return {
            "success": True,
            "predictions": preds,
            "count": len(preds),
            "inferenceMs": round(self.inference_ms, 1),
            "processMs": round(self.inference_ms, 1),
            "moduleId": "ObjectifAI",
            "moduleName": "Objectif.AI",
            "code": 200,
            "command": "detect",
            "requestId": "",
            "inferenceDevice": get_value("detection.backend", "cpu").upper(),
        }


class InferenceEngine:
    """
    Wraps model loading and inference.
    Thread-safe — uses a lock around inference calls.
    """

    def __init__(self):
        self._model = None
        self._model_info: Optional[ModelInfo] = None
        self._engine_type: Optional[str] = None   # "ultralytics", "onnx", "torchhub", "torchvision", "onnx_legacy"
        self._device: str = "cpu"
        self._lock = threading.Lock()
        self._loading = False
        self._load_error: Optional[str] = None
        self._tv_labels: Optional[list] = None  # torchvision label list
        self._legacy_mode: bool = False          # True when using ONNX Runtime CUDA for CC < 7.5 GPUs
        self.log_callback = None                 # set to console.system/info for progress messages

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def model_info(self) -> Optional[ModelInfo]:
        return self._model_info

    @property
    def is_loading(self) -> bool:
        return self._loading

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    @property
    def legacy_mode(self) -> bool:
        return self._legacy_mode

    def _log(self, msg: str):
        """Log via console callback if set, else just logger."""
        logger.info(msg)
        if self.log_callback:
            try:
                self.log_callback(msg)
            except Exception:
                pass

    def load_model(self, model_id: str, backend: str = "auto") -> bool:
        """
        Load a model by ID. backend can be: auto, cpu, cuda, openvino, rocm.
        Returns True on success.
        """
        info = get_model(model_id)
        if info is None:
            self._load_error = f"Unknown model ID: {model_id}"
            logger.error(self._load_error)
            return False

        model_path = MODELS_DIR / info.filename
        # torchhub and torchvision download on first use — don't require file to exist
        if info.engine not in ("torchhub", "torchvision") and not model_path.exists():
            self._load_error = f"Model file not found: {info.filename}"
            logger.error(f"Model file not found at path: {model_path}")
            return False

        self._loading = True
        self._load_error = None

        try:
            device = _resolve_device(backend, info)
            logger.info(f"Loading {info.name} on device={device}")

            # Check for legacy GPU mode (NVIDIA CC < 7.5, CUDA backend requested)
            # Torchvision and torchhub models can't use the ONNX legacy path
            use_legacy = (
                device in ("cuda", "0") and
                info.engine == "ultralytics" and
                _is_legacy_gpu()
            )

            if use_legacy:
                success = self._load_ultralytics_legacy(info, model_path)
            elif info.engine == "ultralytics":
                success = self._load_ultralytics(info, model_path, device)
            elif info.engine == "onnx":
                success = self._load_onnx(info, model_path, backend)
            elif info.engine == "torchhub":
                if _is_legacy_gpu() and device in ("cuda", "0"):
                    self._log("Legacy GPU detected — YOLOv5 (torch.hub) requires PyTorch CUDA. Falling back to CPU.")
                    device = "cpu"
                success = self._load_torchhub(info, model_path, device)
            elif info.engine == "torchvision":
                if _is_legacy_gpu() and device in ("cuda", "0"):
                    self._log("Legacy GPU detected — Torchvision models require PyTorch CUDA. Falling back to CPU.")
                    device = "cpu"
                success = self._load_torchvision(info, model_path, device)
            else:
                self._load_error = f"Unknown engine: {info.engine}"
                return False

            if success:
                self._model_info = info
                self._device = device
                logger.info(f"Model loaded: {info.name} on {device}")
            return success

        except Exception as e:
            self._load_error = str(e)
            logger.exception(f"Failed to load model {model_id}: {e}")
            return False
        finally:
            self._loading = False

    def _load_ultralytics(self, info: ModelInfo, model_path: Path, device: str) -> bool:
        try:
            from ultralytics import YOLO
            # Map our device names to ultralytics format
            ul_device = _device_to_ultralytics(device)
            self._model = YOLO(str(model_path))
            # Warm up with a tiny inference
            import numpy as np
            dummy = np.zeros((info.input_size, info.input_size, 3), dtype=np.uint8)
            self._model.predict(dummy, device=ul_device, verbose=False)
            self._engine_type = "ultralytics"
            self._device = ul_device
            return True
        except ImportError:
            self._load_error = "ultralytics package not installed. Run: pip install ultralytics"
            return False
        except Exception as e:
            self._load_error = str(e)
            raise

    def _load_ultralytics_legacy(self, info: ModelInfo, model_path: Path) -> bool:
        """
        Legacy GPU path for NVIDIA cards with compute capability < 7.5.
        Exports the Ultralytics model to ONNX then loads via ONNX Runtime
        with the CUDA execution provider, which supports older Pascal/Turing GPUs.
        The ONNX file is cached so export only happens once.
        """
        onnx_path = model_path.with_suffix('.legacy.onnx')

        if not onnx_path.exists():
            self._log(
                f"Legacy GPU mode — exporting {info.name} to ONNX for older GPU compatibility. "
                f"This may take up to 60 seconds on first load..."
            )
            try:
                from ultralytics import YOLO
                tmp_model = YOLO(str(model_path))
                exported = tmp_model.export(
                    format="onnx",
                    imgsz=info.input_size,
                    simplify=True,
                    opset=12,
                )
                # Ultralytics exports alongside the source file — move to our cache path
                import shutil
                exported_path = Path(str(model_path).replace('.pt', '.onnx'))
                if exported_path.exists():
                    shutil.move(str(exported_path), str(onnx_path))
                elif Path(str(exported)).exists():
                    shutil.move(str(exported), str(onnx_path))
                else:
                    self._load_error = "ONNX export completed but output file not found"
                    return False
                self._log(f"ONNX export complete — loading on legacy GPU...")
                del tmp_model
            except ImportError:
                self._load_error = "ultralytics not installed"
                return False
            except Exception as e:
                self._load_error = f"ONNX export failed: {e}"
                logger.exception(f"Legacy ONNX export error: {e}")
                return False
        else:
            self._log(f"Legacy GPU mode — loading cached ONNX export for {info.name}...")

        # Load via ONNX Runtime with CUDA execution provider
        try:
            import onnxruntime as ort
            available = ort.get_available_providers()
            if "CUDAExecutionProvider" not in available:
                self._load_error = (
                    "onnxruntime-gpu not installed. Run: python -m pip install onnxruntime-gpu"
                )
                return False

            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._model = ort.InferenceSession(
                str(onnx_path),
                sess_options=opts,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            self._engine_type = "onnx_legacy"
            self._legacy_mode = True
            self._device = "cuda_legacy"
            self._log(f"Legacy GPU mode active — {info.name} running via ONNX Runtime CUDA")
            return True
        except ImportError:
            self._load_error = "onnxruntime not installed. Run: pip install onnxruntime"
            return False
        except Exception as e:
            self._load_error = str(e)
            logger.exception(f"Legacy ONNX load error: {e}")
            return False

    def _load_onnx(self, info: ModelInfo, model_path: Path, backend: str) -> bool:
        try:
            import onnxruntime as ort
            providers = _onnx_providers(backend)
            logger.info(f"ONNX providers: {providers}")
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._model = ort.InferenceSession(
                str(model_path), sess_options=opts, providers=providers
            )
            self._engine_type = "onnx"
            return True
        except ImportError:
            self._load_error = "onnxruntime not installed. Run: pip install onnxruntime"
            return False
        except Exception as e:
            self._load_error = str(e)
            raise

    def _load_torchhub(self, info: ModelInfo, model_path: Path, device: str) -> bool:
        """Load YOLOv5 via torch.hub, caching weights to models/ folder."""
        # Verify torch is importable before attempting hub load
        try:
            import torch
        except ImportError:
            self._load_error = "torch not installed. Run: pip install torch"
            return False

        # YOLOv5 repo requires seaborn, pandas, matplotlib for its utility code
        # even though they are not needed for inference. Install silently if missing.
        try:
            import seaborn  # noqa
        except ImportError:
            logger.info("Installing seaborn (required by YOLOv5 repo)...")
            import subprocess as _sp
            _sp.run([__import__('sys').executable, "-m", "pip", "install",
                     "seaborn", "pandas", "matplotlib", "--quiet"], check=False)

        try:
            import os
            # Redirect torch hub cache to our models directory
            os.environ["TORCH_HOME"] = str(MODELS_DIR)
            torch.hub.set_dir(str(MODELS_DIR))

            # Parse "torchhub:repo:model" from download_url
            # e.g. "torchhub:ultralytics/yolov5:yolov5n"
            parts = info.download_url.replace("torchhub:", "").split(":")
            repo = parts[0]   # "ultralytics/yolov5"
            model_name = parts[1]  # "yolov5n"

            tv_device = "cuda" if device in ("cuda", "0") else "cpu"
            logger.info(f"Loading torch.hub model: {repo} / {model_name} on {tv_device}")
            # If already cached, load from disk (skip network download)
            self._model = torch.hub.load(repo, model_name, pretrained=True,
                                         verbose=False, trust_repo=True,
                                         force_reload=False)
            self._model.to(tv_device)
            self._model.eval()
            self._engine_type = "torchhub"
            self._device = tv_device
            return True
        except Exception as e:
            # Surface the real error — don't swallow it as "torch not installed"
            self._load_error = str(e)
            logger.exception(f"YOLOv5 hub load error: {e}")
            raise

    def _load_torchvision(self, info: ModelInfo, model_path: Path, device: str) -> bool:
        """Load a torchvision detection model, caching weights to models/ folder."""
        try:
            import torch
            import torchvision
            import os
            # Redirect torch hub/checkpoint cache to our models directory
            os.environ["TORCH_HOME"] = str(MODELS_DIR)

            model_name = info.download_url.replace("torchvision:", "")
            tv_device = "cuda" if device in ("cuda", "0") else "cpu"
            # Point TORCH_HOME at our models dir so torchvision finds its cached weights
            import os as _os
            _os.environ["TORCH_HOME"] = str(MODELS_DIR)
            logger.info(f"Loading torchvision model: {model_name} on {tv_device}")

            weights_map = {
                "fasterrcnn_resnet50_fpn":
                    torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1,
                "fasterrcnn_mobilenet_v3_large_fpn":
                    torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1,
                "ssd300_vgg16":
                    torchvision.models.detection.SSD300_VGG16_Weights.COCO_V1,
                "ssdlite320_mobilenet_v3_large":
                    torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.COCO_V1,
                "retinanet_resnet50_fpn":
                    torchvision.models.detection.RetinaNet_ResNet50_FPN_Weights.COCO_V1,
            }
            loader_map = {
                "fasterrcnn_resnet50_fpn":
                    torchvision.models.detection.fasterrcnn_resnet50_fpn,
                "fasterrcnn_mobilenet_v3_large_fpn":
                    torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn,
                "ssd300_vgg16":
                    torchvision.models.detection.ssd300_vgg16,
                "ssdlite320_mobilenet_v3_large":
                    torchvision.models.detection.ssdlite320_mobilenet_v3_large,
                "retinanet_resnet50_fpn":
                    torchvision.models.detection.retinanet_resnet50_fpn,
            }

            if model_name not in loader_map:
                self._load_error = f"Unknown torchvision model: {model_name}"
                return False

            weights = weights_map[model_name]
            self._model = loader_map[model_name](weights=weights)
            self._model.to(tv_device)
            self._model.eval()
            self._engine_type = "torchvision"
            self._device = tv_device

            # Store the label map from weights metadata
            self._tv_labels = weights.meta["categories"]
            return True
        except ImportError as e:
            if 'torchvision' not in str(e).lower() and 'torch' not in str(e).lower():
                self._load_error = f"Dependency error: {e}"
                logger.exception(f"Torchvision load ImportError: {e}")
                raise
            self._load_error = "torchvision not installed. Run: pip install torchvision"
            return False
        except Exception as e:
            self._load_error = str(e)
            raise

    def unload(self):
        with self._lock:
            self._model = None
            self._model_info = None
            self._engine_type = None
            self._load_error = None
            self._tv_labels = None
            self._legacy_mode = False
        logger.info("Model unloaded")

    def detect(self, image_bytes: bytes) -> Optional[DetectionResult]:
        """
        Run detection on raw image bytes.
        Returns DetectionResult or None if no model loaded.
        """
        if not self.is_loaded:
            return None

        with self._lock:
            if self._engine_type == "ultralytics":
                return self._detect_ultralytics(image_bytes)
            elif self._engine_type in ("onnx", "onnx_legacy"):
                # Legacy-exported YOLO and generic YOLO ONNX use the YOLO parser.
                # MobileNet SSD / EfficientDet have their own output layouts.
                info = self._model_info
                if self._engine_type == "onnx" and info is not None and \
                        info.family in ("MobileNet", "EfficientDet"):
                    return self._detect_onnx(image_bytes)
                return self._detect_onnx_yolo(image_bytes)
            elif self._engine_type == "torchhub":
                return self._detect_torchhub(image_bytes)
            elif self._engine_type == "torchvision":
                return self._detect_torchvision(image_bytes)
        return None

    def _detect_ultralytics(self, image_bytes: bytes) -> DetectionResult:
        import numpy as np
        import cv2

        # Decode image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image")

        t0 = time.perf_counter()
        results = self._model.predict(
            img,
            device=self._device,
            verbose=False,
            conf=0.01,  # We apply our own threshold later
        )
        inference_ms = (time.perf_counter() - t0) * 1000

        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            names = r.names
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                detections.append({
                    "label": names[cls_id],
                    "confidence": conf,
                    "x_min": xyxy[0],
                    "y_min": xyxy[1],
                    "x_max": xyxy[2],
                    "y_max": xyxy[3],
                })

        return DetectionResult(detections, inference_ms, self._model_info.id)

    def _detect_onnx(self, image_bytes: bytes) -> DetectionResult:
        """ONNX inference — handles MobileNet SSD and EfficientDet."""
        import numpy as np
        import cv2

        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image")

        info = self._model_info
        input_size = info.input_size

        # Preprocess
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (input_size, input_size))
        img_float = img_resized.astype(np.float32) / 255.0
        img_chw = np.transpose(img_float, (2, 0, 1))
        img_batch = np.expand_dims(img_chw, axis=0)

        input_name = self._model.get_inputs()[0].name
        t0 = time.perf_counter()
        outputs = self._model.run(None, {input_name: img_batch})
        inference_ms = (time.perf_counter() - t0) * 1000

        # Parse outputs — format varies by model family
        detections = _parse_onnx_outputs(outputs, info, img.shape)

        return DetectionResult(detections, inference_ms, info.id)


    def _detect_onnx_yolo(self, image_bytes: bytes) -> DetectionResult:
        """
        ONNX inference for Ultralytics-exported YOLO models (legacy GPU path).
        Handles the standard Ultralytics ONNX output format: [1, 84, N] or [1, N, 84].
        """
        import numpy as np
        import cv2

        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image")

        info = self._model_info
        input_size = info.input_size
        orig_h, orig_w = img.shape[:2]

        # Letterbox resize to input_size x input_size
        scale = min(input_size / orig_w, input_size / orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        pad_x = (input_size - new_w) // 2
        pad_y = (input_size - new_h) // 2
        img_resized = cv2.resize(img, (new_w, new_h))
        canvas = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
        canvas[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = img_resized

        # Preprocess: BGR -> RGB, HWC -> CHW, normalize
        img_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        img_float = img_rgb.astype(np.float32) / 255.0
        img_chw = np.transpose(img_float, (2, 0, 1))
        img_batch = np.expand_dims(img_chw, axis=0)

        input_name = self._model.get_inputs()[0].name
        t0 = time.perf_counter()
        outputs = self._model.run(None, {input_name: img_batch})
        inference_ms = (time.perf_counter() - t0) * 1000

        # Parse Ultralytics ONNX output
        # Shape is typically [1, 84, N] where 84 = 4 box coords + 80 class scores
        # or [1, N, 84] depending on export version
        pred = outputs[0]
        if pred.ndim == 3:
            if pred.shape[1] < pred.shape[2]:
                pred = pred[0].T  # [84, N] -> [N, 84]
            else:
                pred = pred[0]    # already [N, 84]
        elif pred.ndim == 2:
            pass  # already [N, 84]

        detections = []
        for row in pred:
            if len(row) < 5:
                continue
            cx, cy, w, h = row[0], row[1], row[2], row[3]
            class_scores = row[4:]
            cls_id = int(np.argmax(class_scores))
            confidence = float(class_scores[cls_id])
            if confidence < 0.01:
                continue

            # Undo letterbox to get original image coords
            x1 = (cx - w / 2 - pad_x) / scale
            y1 = (cy - h / 2 - pad_y) / scale
            x2 = (cx + w / 2 - pad_x) / scale
            y2 = (cy + h / 2 - pad_y) / scale

            label = COCO_LABELS[cls_id] if cls_id < len(COCO_LABELS) else str(cls_id)
            detections.append({
                "label": label,
                "confidence": confidence,
                "x_min": float(max(0, x1)),
                "y_min": float(max(0, y1)),
                "x_max": float(min(orig_w, x2)),
                "y_max": float(min(orig_h, y2)),
            })

        return DetectionResult(detections, inference_ms, info.id)

    def _detect_torchhub(self, image_bytes: bytes) -> DetectionResult:
        """YOLOv5 inference via torch.hub."""
        import numpy as np
        import cv2
        import torch

        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        t0 = time.perf_counter()
        results = self._model(img_rgb)
        inference_ms = (time.perf_counter() - t0) * 1000

        detections = []
        # YOLOv5 results.xyxy[0] is [x1, y1, x2, y2, conf, cls]
        for *xyxy, conf, cls_id in results.xyxy[0].tolist():
            label = self._model.names[int(cls_id)]
            detections.append({
                "label": label,
                "confidence": float(conf),
                "x_min": float(xyxy[0]),
                "y_min": float(xyxy[1]),
                "x_max": float(xyxy[2]),
                "y_max": float(xyxy[3]),
            })

        return DetectionResult(detections, inference_ms, self._model_info.id)

    def _detect_torchvision(self, image_bytes: bytes) -> DetectionResult:
        """Torchvision detection model inference."""
        import numpy as np
        import cv2
        import torch

        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image")

        # Convert to RGB float tensor [C, H, W] in range [0, 1]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        tensor = tensor.to(self._device)

        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = self._model([tensor])
        inference_ms = (time.perf_counter() - t0) * 1000

        detections = []
        output = outputs[0]
        boxes   = output["boxes"].cpu().tolist()
        scores  = output["scores"].cpu().tolist()
        labels  = output["labels"].cpu().tolist()
        labels_list = self._tv_labels or []

        for box, score, label_idx in zip(boxes, scores, labels):
            if label_idx < len(labels_list):
                label = labels_list[label_idx]
            else:
                label = str(label_idx)
            detections.append({
                "label": label,
                "confidence": float(score),
                "x_min": float(box[0]),
                "y_min": float(box[1]),
                "x_max": float(box[2]),
                "y_max": float(box[3]),
            })

        return DetectionResult(detections, inference_ms, self._model_info.id)


# ---------------------------------------------------------------------------
# ONNX output parsing
# ---------------------------------------------------------------------------

# COCO 80-class labels (index 0-79)
COCO_LABELS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

# MobileNet SSD VOC labels (21 classes, index 0 = background)
MOBILENET_LABELS = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
    "tvmonitor",
]


def _parse_onnx_outputs(outputs, info: ModelInfo, img_shape: tuple) -> list:
    """Best-effort ONNX output parsing for supported model families."""
    import numpy as np

    h, w = img_shape[:2]
    detections = []

    try:
        if info.family == "MobileNet":
            # Outputs: [num_detections, detection_boxes, detection_scores, detection_classes]
            if len(outputs) >= 4:
                num = int(outputs[0][0])
                boxes = outputs[1][0]   # [N, 4] ymin, xmin, ymax, xmax normalized
                scores = outputs[2][0]
                classes = outputs[3][0]
                for i in range(min(num, len(scores))):
                    conf = float(scores[i])
                    if conf < 0.01:
                        continue
                    cls_id = int(classes[i])
                    label = MOBILENET_LABELS[cls_id] if cls_id < len(MOBILENET_LABELS) else str(cls_id)
                    ymin, xmin, ymax, xmax = boxes[i]
                    detections.append({
                        "label": label,
                        "confidence": conf,
                        "x_min": float(xmin * w),
                        "y_min": float(ymin * h),
                        "x_max": float(xmax * w),
                        "y_max": float(ymax * h),
                    })

        elif info.family == "EfficientDet":
            # Output: [batch, num_detections, 7] — [batch, image_id, y1, x1, y2, x2, score, class]
            if len(outputs) >= 1:
                preds = outputs[0]
                if preds.ndim == 3:
                    preds = preds[0]
                for pred in preds:
                    if len(pred) >= 7:
                        score = float(pred[5])
                        cls_id = int(pred[6])
                    elif len(pred) >= 6:
                        score = float(pred[4])
                        cls_id = int(pred[5])
                    else:
                        continue
                    if score < 0.01:
                        continue
                    label = COCO_LABELS[cls_id] if cls_id < len(COCO_LABELS) else str(cls_id)
                    y1, x1, y2, x2 = pred[1], pred[2], pred[3], pred[4]
                    detections.append({
                        "label": label,
                        "confidence": score,
                        "x_min": float(x1),
                        "y_min": float(y1),
                        "x_max": float(x2),
                        "y_max": float(y2),
                    })

        else:
            # Generic ONNX fallback — try to interpret first output as detection array
            arr = outputs[0]
            if arr.ndim == 3:
                arr = arr[0]
            for pred in arr:
                if len(pred) >= 6:
                    conf = float(pred[4])
                    if conf < 0.01:
                        continue
                    cls_id = int(pred[5])
                    label = COCO_LABELS[cls_id] if cls_id < len(COCO_LABELS) else str(cls_id)
                    detections.append({
                        "label": label,
                        "confidence": conf,
                        "x_min": float(pred[0]),
                        "y_min": float(pred[1]),
                        "x_max": float(pred[2]),
                        "y_max": float(pred[3]),
                    })

    except Exception as e:
        logger.warning(f"ONNX output parsing error: {e}")

    return detections


# ---------------------------------------------------------------------------
# Legacy GPU detection
# ---------------------------------------------------------------------------

_legacy_gpu_cache: Optional[bool] = None

def _is_legacy_gpu() -> bool:
    """
    Returns True if the detected NVIDIA GPU has compute capability < 7.5.
    Cached after first check. Returns False if no NVIDIA GPU or CC unknown.
    """
    global _legacy_gpu_cache
    if _legacy_gpu_cache is not None:
        return _legacy_gpu_cache

    # Try PyTorch first
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            _legacy_gpu_cache = props.major < 7 or (props.major == 7 and props.minor < 5)
            return _legacy_gpu_cache
    except Exception:
        pass

    # Try nvidia-smi
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(".")
            major = int(parts[0])
            minor = int(parts[1]) if len(parts) > 1 else 0
            _legacy_gpu_cache = major < 7 or (major == 7 and minor < 5)
            return _legacy_gpu_cache
    except Exception:
        pass

    _legacy_gpu_cache = False
    return False


# ---------------------------------------------------------------------------
# Device resolution helpers
# ---------------------------------------------------------------------------

def _resolve_device(backend: str, info: ModelInfo) -> str:
    """Resolve backend string to a concrete device."""
    if backend == "auto":
        return _auto_device(info)
    return backend


def _auto_device(info: ModelInfo) -> str:
    """Pick the best available device for the given model."""
    preferred = info.best_hardware

    # Try CUDA first
    if "cuda" in preferred:
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass

    # Try DirectML (Windows, any DX12 GPU) — only useful for ONNX-engine models
    if "directml" in preferred and info.engine == "onnx":
        try:
            import onnxruntime as ort
            if "DmlExecutionProvider" in ort.get_available_providers():
                return "directml"
        except Exception:
            pass

    # Try OpenVINO
    if "openvino" in preferred:
        try:
            from openvino.runtime import Core
            core = Core()
            if "GPU" in core.available_devices:
                return "openvino"
        except Exception:
            pass

    return "cpu"


def _device_to_ultralytics(device: str) -> str:
    """Map our device names to ultralytics predict() device parameter."""
    mapping = {
        "cuda": "0",        # first CUDA device
        "cpu": "cpu",
        "openvino": "cpu",  # ultralytics handles openvino via export; we use CPU path
        "directml": "cpu",  # ultralytics .pt has no DML path; ONNX models use DML directly
        "rocm": "0",
    }
    return mapping.get(device, "cpu")


def _onnx_providers(backend: str) -> list:
    """Return ONNX Runtime execution providers list."""
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
    except Exception:
        return ["CPUExecutionProvider"]

    if backend in ("cuda", "auto"):
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    if backend in ("directml", "auto"):
        # DmlExecutionProvider ships with the onnxruntime-directml package and
        # runs on any DirectX 12 GPU (NVIDIA, AMD, Intel) on Windows.
        if "DmlExecutionProvider" in available:
            return ["DmlExecutionProvider", "CPUExecutionProvider"]

    if backend == "openvino":
        if "OpenVINOExecutionProvider" in available:
            return ["OpenVINOExecutionProvider", "CPUExecutionProvider"]

    return ["CPUExecutionProvider"]


# ---------------------------------------------------------------------------
# ALPR engine (license plate detection + OCR via fast-alpr)
# ---------------------------------------------------------------------------

class ALPRResult:
    """One recognized plate: text, overall confidence, and bounding box."""

    def __init__(self, plate: str, confidence: float,
                 x_min: float, y_min: float, x_max: float, y_max: float):
        self.plate = plate
        self.confidence = confidence
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max


class ALPREngine:
    """
    Wraps the fast-alpr pipeline (plate detection + OCR).

    fast-alpr downloads its ONNX weights from Hugging Face on first load and
    caches them under the user's HF cache, mirroring how the YOLOv5 (torch.hub)
    and torchvision paths already self-manage their weights. Kept separate from
    InferenceEngine so the two can be loaded independently — BlueIris hits ALPR
    on its own endpoint and may run it alongside object detection.
    """

    def __init__(self):
        self._alpr = None
        self._lock = threading.Lock()
        self._loading = False
        self._load_error: Optional[str] = None
        self._detector_model: Optional[str] = None
        self._ocr_model: Optional[str] = None
        self.log_callback = None

    @property
    def is_loaded(self) -> bool:
        return self._alpr is not None

    @property
    def is_loading(self) -> bool:
        return self._loading

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    @property
    def active_models(self) -> dict:
        return {"detector": self._detector_model, "ocr": self._ocr_model}

    def _log(self, msg: str):
        logger.info(msg)
        if self.log_callback:
            try:
                self.log_callback(msg)
            except Exception:
                pass

    def load(self, detector_model: str, ocr_model: str) -> bool:
        """
        Load (or reload) the ALPR pipeline. First call may download weights.
        Returns True on success.
        """
        self._loading = True
        self._load_error = None
        try:
            try:
                from fast_alpr import ALPR
            except ImportError:
                self._load_error = (
                    "fast-alpr not installed. Install it from the Dependencies tab "
                    "or run: pip install fast-alpr onnxruntime"
                )
                logger.error(self._load_error)
                return False

            self._log(
                f"Loading ALPR pipeline (detector={detector_model}, ocr={ocr_model}). "
                f"First load downloads model weights — this may take a moment..."
            )
            alpr = ALPR(detector_model=detector_model, ocr_model=ocr_model)
            with self._lock:
                self._alpr = alpr
                self._detector_model = detector_model
                self._ocr_model = ocr_model
            self._log("ALPR pipeline ready.")
            return True
        except Exception as e:
            self._load_error = str(e)
            logger.exception(f"Failed to load ALPR pipeline: {e}")
            return False
        finally:
            self._loading = False

    def unload(self):
        with self._lock:
            self._alpr = None
            self._detector_model = None
            self._ocr_model = None
            self._load_error = None
        logger.info("ALPR pipeline unloaded")

    def recognize(self, image_bytes: bytes) -> tuple:
        """
        Run ALPR on raw image bytes.
        Returns (list[ALPRResult], inference_ms). Empty list if nothing found.
        """
        if not self.is_loaded:
            return [], 0.0

        import numpy as np
        import cv2

        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image")

        with self._lock:
            t0 = time.perf_counter()
            raw = self._alpr.predict(img)
            inference_ms = (time.perf_counter() - t0) * 1000

        results = []
        for r in raw:
            # fast-alpr returns objects with .detection (bbox + conf) and
            # .ocr (text + conf). Guard each field — OCR can be None if the
            # plate was detected but unreadable.
            try:
                det = getattr(r, "detection", None)
                ocr = getattr(r, "ocr", None)
                if det is None or ocr is None or not getattr(ocr, "text", None):
                    continue
                bbox = det.bounding_box
                det_conf = float(getattr(det, "confidence", 1.0) or 1.0)
                ocr_conf = float(getattr(ocr, "confidence", 1.0) or 1.0)
                results.append(ALPRResult(
                    plate=str(ocr.text).strip().upper(),
                    confidence=det_conf * ocr_conf,
                    x_min=float(bbox.x1),
                    y_min=float(bbox.y1),
                    x_max=float(bbox.x2),
                    y_max=float(bbox.y2),
                ))
            except Exception as e:
                logger.warning(f"Skipping malformed ALPR result: {e}")
                continue

        return results, inference_ms


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
engine = InferenceEngine()
alpr_engine = ALPREngine()
