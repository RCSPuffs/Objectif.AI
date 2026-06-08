"""
Download manager for Objectif.AI.
Handles model file downloads with progress tracking via WebSocket.
"""

import asyncio
import json
import logging
import threading
import time
from pathlib import Path
from typing import Optional, Callable
import hashlib
import urllib.request
import urllib.error

from model_registry import get_model, ModelInfo

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


class DownloadState:
    IDLE = "idle"
    DOWNLOADING = "downloading"
    COMPLETE = "complete"
    ERROR = "error"
    CANCELLED = "cancelled"


class DownloadManager:
    def __init__(self):
        self._active: dict[str, dict] = {}     # model_id -> state dict
        self._clients: set = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop

    def add_client(self, ws):
        self._clients.add(ws)

    def remove_client(self, ws):
        self._clients.discard(ws)

    def get_status(self, model_id: str) -> dict:
        return self._active.get(model_id, {"state": DownloadState.IDLE})

    def get_all_statuses(self) -> dict:
        return dict(self._active)

    def is_downloaded(self, model_id: str) -> bool:
        info = get_model(model_id)
        if info is None:
            return False
        if info.engine == "torchhub":
            # torch.hub caches .pt files under models/ultralytics_yolov5_master/
            # Check both the hub subfolder and direct models/ location
            hub_path = MODELS_DIR / "ultralytics_yolov5_master" / info.filename
            direct_path = MODELS_DIR / info.filename
            return hub_path.exists() or direct_path.exists()
        if info.engine == "torchvision":
            # PyTorch saves to different subfolders depending on version/platform.
            # Scan all subdirectories under MODELS_DIR for a file matching the
            # model name prefix (ignoring the hash suffix PyTorch appends).
            stem = info.filename.replace(".pth", "").split("_coco")[0]
            for f in MODELS_DIR.rglob("*.pth"):
                if f.name.startswith(stem):
                    return True
            return False
        return (MODELS_DIR / info.filename).exists()

    def cancel(self, model_id: str):
        if model_id in self._active:
            self._active[model_id]["cancelled"] = True

    def download_model(self, model_id: str) -> bool:
        """
        Start downloading a model in a background thread.
        Returns False if already downloading or model unknown.
        """
        info = get_model(model_id)
        if info is None:
            logger.error(f"Unknown model: {model_id}")
            return False

        if model_id in self._active and self._active[model_id]["state"] == DownloadState.DOWNLOADING:
            logger.warning(f"Already downloading: {model_id}")
            return False

        state = {
            "state": DownloadState.DOWNLOADING,
            "model_id": model_id,
            "filename": info.filename,
            "total_bytes": 0,
            "downloaded_bytes": 0,
            "percent": 0.0,
            "speed_kbps": 0.0,
            "cancelled": False,
            "error": None,
        }
        self._active[model_id] = state

        # Route to appropriate download method based on engine
        if info.engine == "torchhub":
            target = self._download_torchhub_thread
        elif info.engine == "torchvision":
            target = self._download_torchvision_thread
        else:
            target = self._download_thread

        thread = threading.Thread(
            target=target,
            args=(info, state),
            daemon=True,
        )
        thread.start()
        return True

    def _download_thread(self, info: ModelInfo, state: dict):
        dest = MODELS_DIR / info.filename
        tmp = dest.with_suffix(".tmp")

        try:
            logger.info(f"Starting download: {info.name} from {info.download_url}")

            if not info.download_url:
                state["state"] = DownloadState.ERROR
                state["error"] = "No download URL — this model requires manual installation. Place the file in the models/ folder."
                logger.error(f"No download URL for {info.id}")
                self._emit_progress(state)
                return

            req = urllib.request.Request(
                info.download_url,
                headers={"User-Agent": "ObjectifAI/0.3"}
            )

            with urllib.request.urlopen(req, timeout=30) as response:
                total = int(response.headers.get("Content-Length", 0))
                state["total_bytes"] = total
                downloaded = 0
                chunk_size = 65536  # 64KB chunks
                t_start = time.monotonic()
                t_last_report = t_start

                with open(tmp, "wb") as f:
                    while True:
                        if state.get("cancelled"):
                            logger.info(f"Download cancelled: {info.id}")
                            state["state"] = DownloadState.CANCELLED
                            tmp.unlink(missing_ok=True)
                            self._emit_progress(state)
                            return

                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        state["downloaded_bytes"] = downloaded

                        now = time.monotonic()
                        elapsed = now - t_start
                        if elapsed > 0:
                            state["speed_kbps"] = (downloaded / elapsed) / 1024
                        if total > 0:
                            state["percent"] = round((downloaded / total) * 100, 1)

                        # Emit progress ~every 250ms
                        if now - t_last_report >= 0.25:
                            self._emit_progress(state)
                            t_last_report = now

            # Verify SHA-256 if we have an expected hash
            if info.sha256:
                actual = _sha256_file(tmp)
                if actual != info.sha256.lower():
                    state["state"] = DownloadState.ERROR
                    state["error"] = (
                        f"SHA-256 mismatch — file may be corrupt or tampered with. "
                        f"Expected {info.sha256[:16]}..., got {actual[:16]}..."
                    )
                    logger.error(f"SHA-256 mismatch for {info.id}")
                    tmp.unlink(missing_ok=True)
                    self._emit_progress(state)
                    return

            # Move tmp to final location
            tmp.rename(dest)
            state["state"] = DownloadState.COMPLETE
            state["percent"] = 100.0
            logger.info(f"Download complete: {info.name} -> {dest}")
            self._emit_progress(state)

        except urllib.error.URLError as e:
            state["state"] = DownloadState.ERROR
            state["error"] = f"Network error: {e.reason}"
            logger.error(f"Download failed ({info.id}): {e}")
            tmp.unlink(missing_ok=True)
            self._emit_progress(state)

        except Exception as e:
            state["state"] = DownloadState.ERROR
            state["error"] = str(e)
            logger.exception(f"Download error ({info.id}): {e}")
            tmp.unlink(missing_ok=True)
            self._emit_progress(state)

    def _emit_progress(self, state: dict):
        """Broadcast download progress to all connected WebSocket clients."""
        if not self._clients:
            return
        payload = json.dumps({"type": "download_progress", "data": state})
        dead = set()
        for ws in list(self._clients):
            try:
                if self._loop and self._loop.is_running():
                    asyncio.run_coroutine_threadsafe(
                        _safe_send(ws, payload), self._loop
                    )
            except Exception:
                dead.add(ws)
        self._clients -= dead

    def _download_torchhub_thread(self, info: ModelInfo, state: dict):
        """Download YOLOv5 weights via torch.hub into models/ folder."""
        import os
        try:
            import torch
        except ImportError:
            state["state"] = DownloadState.ERROR
            state["error"] = "torch not installed. Run: pip install torch"
            self._emit_progress(state)
            return

        try:
            os.environ["TORCH_HOME"] = str(MODELS_DIR)
            torch.hub.set_dir(str(MODELS_DIR))

            parts = info.download_url.replace("torchhub:", "").split(":")
            repo, model_name = parts[0], parts[1]

            logger.info(f"Downloading torch.hub model: {repo}/{model_name}")
            state["state"] = DownloadState.DOWNLOADING
            state["percent"] = 10.0
            self._emit_progress(state)

            # torch.hub.load downloads repo + weights — no granular progress available
            # We emit indeterminate progress while it runs
            import threading as _threading
            done = [False]

            def _pulse():
                pct = 10.0
                while not done[0]:
                    import time as _time
                    _time.sleep(1.0)
                    pct = min(pct + 5.0, 90.0)
                    if not done[0]:
                        state["percent"] = pct
                        self._emit_progress(state)

            pulse_thread = _threading.Thread(target=_pulse, daemon=True)
            pulse_thread.start()

            try:
                # Just download, don't keep model in memory
                model = torch.hub.load(repo, model_name, pretrained=True,
                                       verbose=False, trust_repo=True)
                del model
                import gc; gc.collect()
            finally:
                done[0] = True

            state["state"] = DownloadState.COMPLETE
            state["percent"] = 100.0
            logger.info(f"torch.hub download complete: {info.name}")
            self._emit_progress(state)

        except Exception as e:
            state["state"] = DownloadState.ERROR
            state["error"] = str(e)
            logger.exception(f"torch.hub download error ({info.id}): {e}")
            self._emit_progress(state)

    def _download_torchvision_thread(self, info: ModelInfo, state: dict):
        """Download torchvision model weights. PyTorch saves to a checkpoints subfolder under MODELS_DIR."""
        import os
        try:
            import torch
            import torchvision
        except ImportError as e:
            state["state"] = DownloadState.ERROR
            state["error"] = f"Missing package: {e}"
            self._emit_progress(state)
            return

        try:
            os.environ["TORCH_HOME"] = str(MODELS_DIR)
            model_name = info.download_url.replace("torchvision:", "")

            loader_map = {
                "fasterrcnn_resnet50_fpn":
                    (torchvision.models.detection.fasterrcnn_resnet50_fpn,
                     torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1),
                "fasterrcnn_mobilenet_v3_large_fpn":
                    (torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn,
                     torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1),
                "ssd300_vgg16":
                    (torchvision.models.detection.ssd300_vgg16,
                     torchvision.models.detection.SSD300_VGG16_Weights.COCO_V1),
                "ssdlite320_mobilenet_v3_large":
                    (torchvision.models.detection.ssdlite320_mobilenet_v3_large,
                     torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.COCO_V1),
                "retinanet_resnet50_fpn":
                    (torchvision.models.detection.retinanet_resnet50_fpn,
                     torchvision.models.detection.RetinaNet_ResNet50_FPN_Weights.COCO_V1),
            }

            if model_name not in loader_map:
                state["state"] = DownloadState.ERROR
                state["error"] = f"Unknown torchvision model: {model_name}"
                self._emit_progress(state)
                return

            loader_fn, weights = loader_map[model_name]
            # Ensure TORCH_HOME is set so torchvision saves to models/checkpoints/
            os.environ["TORCH_HOME"] = str(MODELS_DIR)
            logger.info(f"Downloading torchvision model: {model_name}")
            state["percent"] = 10.0
            self._emit_progress(state)

            import threading as _threading
            done = [False]

            def _pulse():
                pct = 10.0
                while not done[0]:
                    import time as _time
                    _time.sleep(1.0)
                    pct = min(pct + 4.0, 90.0)
                    if not done[0]:
                        state["percent"] = pct
                        self._emit_progress(state)

            pulse_thread = _threading.Thread(target=_pulse, daemon=True)
            pulse_thread.start()

            try:
                model = loader_fn(weights=weights)
                del model
                import gc; gc.collect()
            finally:
                done[0] = True

            state["state"] = DownloadState.COMPLETE
            state["percent"] = 100.0
            logger.info(f"torchvision download complete: {info.name}")
            self._emit_progress(state)

        except Exception as e:
            state["state"] = DownloadState.ERROR
            state["error"] = str(e)
            logger.exception(f"torchvision download error ({info.id}): {e}")
            self._emit_progress(state)

    def delete_model(self, model_id: str) -> bool:
        """Delete a downloaded model file."""
        info = get_model(model_id)
        if info is None:
            return False
        # Check all possible locations
        candidates = [MODELS_DIR / info.filename]
        if info.engine == "torchhub":
            candidates.append(MODELS_DIR / "ultralytics_yolov5_master" / info.filename)
        if info.engine == "torchvision":
            # Scan all subfolders — PyTorch saves to varying locations by version
            stem = info.filename.replace(".pth", "").split("_coco")[0]
            for f in MODELS_DIR.rglob("*.pth"):
                if f.name.startswith(stem):
                    candidates.append(f)
        for path in candidates:
            if path.exists():
                path.unlink()
                logger.info(f"Deleted model file: {info.filename}")
                return True
        return False


async def _safe_send(ws, payload: str):
    try:
        await ws.send_text(payload)
    except Exception:
        pass


def _sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
download_manager = DownloadManager()
