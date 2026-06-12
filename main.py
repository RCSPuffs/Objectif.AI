"""
Objectif.AI — Main FastAPI application.
BlueIris-compatible AI object detection — objectif.ai.
Listens on port 32168 by default.
"""

import asyncio
import collections
import hashlib
import io
import ipaddress
import json
import logging
import os
import re
import secrets as _secrets
import subprocess
import sys
import threading
import time
import warnings

# Suppress YOLOv5 torch.cuda.amp FutureWarning — harmless deprecation in YOLOv5 repo code
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.cuda.amp.autocast.*")

# Tell Ultralytics/YOLOv5 not to attempt auto-installing dependencies
# (avoids malformed pip marker errors from urllib3 requirement string)
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("ULTRALYTICS_AUTOINSTALL", "false")
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, Form, Request, UploadFile, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

# Local modules
from config import get_config, update_config, get_value, get_coco_classes_grouped, get_all_coco_classes
from console_buffer import console
from detector import engine, alpr_engine, MODELS_DIR
from downloader import download_manager
from hardware import detect_hardware, hardware_info_to_dict
from model_registry import get_all_models, get_model, model_to_dict
from auth import verify_api_key, verify_websocket_key, get_or_create_api_key, is_first_run
from dependencies import check_all_dependencies, get_python_info

# ---------------------------------------------------------------------------
# Logging setup — console + rotating file logs
# ---------------------------------------------------------------------------
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# File handler — server.log (rotating, keep 5 x 1MB)
from logging.handlers import RotatingFileHandler as _RFH
_file_handler = _RFH(LOG_DIR / "server.log", maxBytes=1_000_000, backupCount=5, encoding="utf-8")
_file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%H:%M:%S"))
logging.getLogger().addHandler(_file_handler)

def _startup_log(msg: str):
    """Append a line to logs/startup.log (forever, no rotation)."""
    try:
        with open(LOG_DIR / "startup.log", "a", encoding="utf-8") as f:
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{ts}] {msg}\n")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Startup state
# ---------------------------------------------------------------------------
_hardware_info = None
_start_time = time.time()
_service_running = False  # True once lifespan startup completes

# ---------------------------------------------------------------------------
# Inference stats -- rolling window for average, plus last value
# ---------------------------------------------------------------------------
_inference_times: collections.deque = collections.deque(maxlen=20)
_last_inference_ms: float = 0.0
_inference_ws_clients: set = set()


def _record_inference(ms: float):
    global _last_inference_ms
    _last_inference_ms = ms
    _inference_times.append(ms)
    _broadcast_inference_stats()


def _broadcast_inference_stats():
    global _inference_ws_clients
    if not _inference_ws_clients:
        return
    avg = sum(_inference_times) / len(_inference_times) if _inference_times else 0.0
    payload = json.dumps({
        "type": "inference_stats",
        "last_ms": round(_last_inference_ms, 1),
        "avg_ms": round(avg, 1),
    })
    dead = set()
    for ws in list(_inference_ws_clients):
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.run_coroutine_threadsafe(_safe_ws_send(ws, payload), loop)
        except Exception:
            dead.add(ws)
    _inference_ws_clients -= dead


async def _safe_ws_send(ws, payload: str):
    try:
        await ws.send_text(payload)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Security primitives — Host-header allowlist, rate limiter, WS tickets.
# Designed to harden the LAN-only deployment without changing the BlueIris
# workflow. None of these add new dependencies.
# ---------------------------------------------------------------------------

# Host header allowlist — accept only localhost and RFC1918 private addresses
# (plus an optional :port suffix). Blocks DNS-rebinding attacks where a public
# domain resolves to a LAN IP, because the browser would still send Host: <domain>.
_LOCALHOST_HOSTS = {"localhost", "127.0.0.1", "::1", "[::1]"}

def _host_is_allowed(host_header: str) -> bool:
    if not host_header:
        return False
    # Strip optional port
    host = host_header
    if host.startswith("["):
        # IPv6 literal — only ::1 allowed
        end = host.find("]")
        if end == -1:
            return False
        return host[: end + 1].lower() in _LOCALHOST_HOSTS
    if ":" in host:
        host = host.rsplit(":", 1)[0]
    host_l = host.lower()
    if host_l in _LOCALHOST_HOSTS:
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    # Accept loopback and private (RFC1918 / link-local / ULA) addresses only.
    return ip.is_loopback or ip.is_private or ip.is_link_local


class HostHeaderMiddleware(BaseHTTPMiddleware):
    """Reject requests whose Host header isn't localhost or a private IP."""
    async def dispatch(self, request: Request, call_next):
        host = request.headers.get("host", "")
        if not _host_is_allowed(host):
            return JSONResponse(
                {"error": "Host header rejected — Objectif.AI only accepts requests on LAN/loopback addresses."},
                status_code=400,
            )
        return await call_next(request)


# Per-IP sliding-window rate limiter for /v1/vision/detection.
# Default 100 req/sec — high enough for a multi-camera BlueIris deployment,
# low enough to make brute-force flooding ineffective.
_RATE_LIMIT_PER_SEC = 100
_RATE_WINDOW_S = 1.0
_rate_buckets: dict = {}
_rate_lock = threading.Lock()


def _rate_limit_check(client_ip: str) -> bool:
    """Return True if the request is allowed, False if it should be rejected."""
    now = time.monotonic()
    with _rate_lock:
        bucket = _rate_buckets.get(client_ip)
        if bucket is None:
            bucket = collections.deque()
            _rate_buckets[client_ip] = bucket
        cutoff = now - _RATE_WINDOW_S
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        if len(bucket) >= _RATE_LIMIT_PER_SEC:
            return False
        bucket.append(now)
        # Opportunistic cleanup: prevent unbounded growth from one-shot clients.
        if len(_rate_buckets) > 10_000:
            for k in [k for k, v in _rate_buckets.items() if not v or v[-1] < cutoff]:
                _rate_buckets.pop(k, None)
        return True


# Short-lived single-use WebSocket tickets. The dashboard exchanges its API key
# for a ticket via POST /api/auth/ws-ticket, then connects with ?ticket=...
# This keeps the long-lived API key out of WebSocket URLs (and therefore out of
# uvicorn access logs, browser history, and DevTools network panels).
_WS_TICKET_TTL_S = 30.0
_ws_tickets: dict = {}  # ticket -> expiry monotonic
_ws_ticket_lock = threading.Lock()


def issue_ws_ticket() -> str:
    """Mint a one-shot ticket valid for _WS_TICKET_TTL_S seconds."""
    now = time.monotonic()
    ticket = _secrets.token_urlsafe(32)
    with _ws_ticket_lock:
        # Drop expired tickets opportunistically
        expired = [t for t, exp in _ws_tickets.items() if exp < now]
        for t in expired:
            _ws_tickets.pop(t, None)
        _ws_tickets[ticket] = now + _WS_TICKET_TTL_S
    return ticket


def consume_ws_ticket(ticket: str) -> bool:
    """Return True iff the ticket exists, hasn't expired, and hasn't been used."""
    if not ticket:
        return False
    now = time.monotonic()
    with _ws_ticket_lock:
        exp = _ws_tickets.pop(ticket, None)
    return exp is not None and exp >= now


# ---------------------------------------------------------------------------
# Package installer — streams pip output via WebSocket.
# Supports chained sequential installs.
# Includes a timeout guard so the lock can never stay stuck permanently.
# ---------------------------------------------------------------------------
_install_ws_clients: set = set()
_active_install = None
_install_start_time: float = 0.0
_INSTALL_TIMEOUT_S: float = 1800.0  # 30 minutes


def _is_install_stuck() -> bool:
    if _active_install is None:
        return False
    return (time.time() - _install_start_time) > _INSTALL_TIMEOUT_S


def _install_lock_clear():
    global _active_install, _install_start_time
    if _active_install is not None:
        try:
            _active_install.kill()
        except Exception:
            pass
    _active_install = None
    _install_start_time = 0.0


def _broadcast_install(msg: dict):
    global _install_ws_clients
    if not _install_ws_clients:
        return
    payload = json.dumps({"type": "install_output", **msg})
    dead = set()
    for ws in list(_install_ws_clients):
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.run_coroutine_threadsafe(_safe_ws_send(ws, payload), loop)
        except Exception:
            dead.add(ws)
    _install_ws_clients -= dead


def _run_install_chain(steps: list):
    """
    Run a chain of pip operations sequentially in a background thread.
    steps = [(packages_list, label_str), ...]

    A step whose packages_list begins with the sentinel "--uninstall" is run as
    `pip uninstall -y <rest>` and a non-zero return code is tolerated (the
    packages may simply not be installed). All other steps run as
    `pip install <packages>` and stop the chain on failure.
    """
    global _active_install, _install_start_time

    total = len(steps)
    for step_idx, (packages, label) in enumerate(steps):
        step_label = f"{label} ({step_idx+1}/{total})" if total > 1 else label

        is_uninstall = bool(packages) and packages[0] == "--uninstall"
        if is_uninstall:
            cmd = [sys.executable, "-m", "pip", "uninstall", "-y"] + packages[1:]
        else:
            cmd = [sys.executable, "-m", "pip", "install"] + packages

        console.system(f"{'Uninstalling' if is_uninstall else 'Installing'}: {step_label}")
        _broadcast_install({
            "status": "start",
            "label": step_label,
            "cmd": " ".join(cmd),
            "step": step_idx + 1,
            "total_steps": total,
        })
        _install_start_time = time.time()
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            _active_install = proc
            for line in proc.stdout:
                line = line.rstrip()
                if line:
                    _broadcast_install({"status": "line", "text": line})
            proc.wait()
            # Uninstall steps are best-effort: not-installed -> non-zero is fine.
            success = proc.returncode == 0 or is_uninstall
            _broadcast_install({
                "status": "step_done",
                "label": step_label,
                "success": success,
                "returncode": proc.returncode,
                "step": step_idx + 1,
                "total_steps": total,
                "chain_complete": success and step_idx == total - 1,
            })
            if success:
                console.success(f"Step complete: {step_label}")
            else:
                console.error(f"Step failed (code {proc.returncode}): {step_label}")
                _broadcast_install({"status": "done", "success": False, "returncode": proc.returncode})
                return
        except Exception as e:
            _broadcast_install({"status": "error", "text": str(e)})
            console.error(f"Install error: {e}")
            return
        finally:
            _active_install = None
            _install_start_time = 0.0

    _broadcast_install({"status": "done", "success": True, "returncode": 0})


# ---------------------------------------------------------------------------
# CUDA version -> torch wheel index URL mapping
# ---------------------------------------------------------------------------
def _cuda_wheel_url(cuda_version: str) -> str:
    """Map detected CUDA version string to the correct PyTorch wheel URL."""
    try:
        major, minor = [int(x) for x in cuda_version.split(".")[:2]]
        version_int = major * 10 + minor
    except Exception:
        return "https://download.pytorch.org/whl/cu121"
    if version_int >= 128:
        return "https://download.pytorch.org/whl/cu128"
    elif version_int >= 124:
        return "https://download.pytorch.org/whl/cu124"
    elif version_int >= 121:
        return "https://download.pytorch.org/whl/cu121"
    elif version_int >= 118:
        return "https://download.pytorch.org/whl/cu118"
    else:
        return "https://download.pytorch.org/whl/cu117"


@asynccontextmanager
def _silence_proactor_pipe_errors(loop, context):
    """Suppress harmless Windows TCP reset noise on client disconnect."""
    exception = context.get("exception")
    if isinstance(exception, (ConnectionResetError, BrokenPipeError)):
        return
    loop.default_exception_handler(context)


async def lifespan(app: FastAPI):
    global _hardware_info, _service_running

    loop = asyncio.get_event_loop()
    loop.set_exception_handler(_silence_proactor_pipe_errors)
    console.set_loop(loop)
    download_manager.set_loop(loop)

    cfg = get_config()
    port = cfg["server"]["port"]

    _startup_log("=" * 50)
    _startup_log("Objectif.AI v0.7.9 starting up")

    console.system("=" * 55)
    console.system("  Objectif.AI v0.7.9 starting up...")
    console.system("=" * 55)

    # Hardware detection
    console.system("Detecting hardware...")
    _startup_log("Detecting hardware...")
    _hardware_info = detect_hardware()
    hw = hardware_info_to_dict(_hardware_info)

    cpu_msg = f"CPU: {_hardware_info.cpu_name} ({_hardware_info.cpu_cores}c/{_hardware_info.cpu_threads}t) — {_hardware_info.ram_gb} GB RAM"
    console.system(cpu_msg)
    _startup_log(cpu_msg)

    if _hardware_info.has_nvidia:
        gpu_msg = f"NVIDIA GPU: {_hardware_info.nvidia_name} ({_hardware_info.nvidia_vram_gb} GB VRAM) — CUDA {_hardware_info.cuda_version}"
        console.success(gpu_msg)
        _startup_log(gpu_msg)
    else:
        console.info("No NVIDIA GPU detected")
        _startup_log("No NVIDIA GPU detected")

    if _hardware_info.has_intel_gpu:
        console.success(f"Intel GPU: {_hardware_info.intel_gpu_name}")
        if _hardware_info.has_openvino:
            console.success(f"OpenVINO: {_hardware_info.openvino_version}")
    
    if _hardware_info.has_amd_gpu:
        if _hardware_info.has_rocm:
            console.success(f"AMD GPU: {_hardware_info.amd_gpu_name} — ROCm {_hardware_info.rocm_version}")
        else:
            console.warning(f"AMD GPU detected ({_hardware_info.amd_gpu_name}) but ROCm not installed — CPU fallback will be used")

    for warning in _hardware_info.warnings:
        console.warning(warning)

    console.system(f"Recommended backend: {_hardware_info.recommended_backend.upper()}")

    # Auto-select backend if config says "auto"
    if cfg["detection"]["backend"] == "auto":
        update_config("detection.backend", _hardware_info.recommended_backend)
        console.system(f"Backend set to: {_hardware_info.recommended_backend.upper()}")

    # Load last-used model
    active_model_id = get_value("detection.active_model")
    if active_model_id:
        model_path = MODELS_DIR / (get_model(active_model_id).filename if get_model(active_model_id) else "")
        if get_model(active_model_id) and model_path.exists():
            console.system(f"Loading last-used model: {active_model_id}...")
            backend = get_value("detection.backend", "cpu")
            ok = engine.load_model(active_model_id, backend)
            if ok:
                console.success(f"Model ready: {engine.model_info.name}")
                _startup_log(f"Model loaded: {engine.model_info.name}")
            else:
                console.error(f"Failed to load model: {engine.load_error}")
                _startup_log(f"ERROR: Failed to load model: {engine.load_error}")
        else:
            console.warning(f"Last-used model '{active_model_id}' not found on disk — please download it")
            update_config("detection.active_model", None)
    else:
        console.info("No model loaded. Select one from the Model Browser.")

    # Wire console log callback to inference engine for legacy GPU messages
    engine.log_callback = console.system
    alpr_engine.log_callback = console.system

    # Restore the ALPR pipeline if it was active in a previous session.
    if get_value("alpr.active"):
        det_m = get_value("alpr.detector_model")
        ocr_m = get_value("alpr.ocr_model")
        console.system("Restoring ALPR pipeline from previous session...")
        if not alpr_engine.load(det_m, ocr_m):
            console.warning(f"ALPR restore failed: {alpr_engine.load_error}")

    console.system(f"Server listening on http://0.0.0.0:{port}")
    console.system("Ready.")
    _startup_log(f"Ready — listening on port {port}")

    _service_running = True

    yield

    # Shutdown
    _service_running = False
    _startup_log("Server shut down cleanly")
    console.system("Shutting down...")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Objectif.AI",
    description="Drop-in replacement for CodeProject.AI — BlueIris compatible",
    version="0.7.9",
    lifespan=lifespan,
)

# Reject requests whose Host header isn't a LAN/loopback address. This blocks
# DNS-rebinding attacks where a public domain points at the user's LAN IP.
app.add_middleware(HostHeaderMiddleware)

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

# Verify static dir exists — no path output to protect user privacy
if not (STATIC_DIR / 'index.html').exists():
    logger.warning("static/index.html not found — dashboard will not load")


# ---------------------------------------------------------------------------
# CODEPROJECT.AI COMPATIBLE ENDPOINTS
# BlueIris calls these — must match CPAI's API exactly.
# ---------------------------------------------------------------------------

@app.post("/v1/vision/detection")
async def detect_objects(
    request: Request,
    image: UploadFile = File(...),
    min_confidence: Optional[float] = Form(None),
):
    """
    Primary detection endpoint — CodeProject.AI compatible.
    BlueIris POSTs images here as multipart/form-data.
    """
    # Per-IP rate limit. This endpoint is intentionally unauthenticated for
    # BlueIris compatibility, so a flood-control layer is the only defense.
    client_ip = request.client.host if request.client else "unknown"
    if not _rate_limit_check(client_ip):
        return JSONResponse(
            {"success": False, "error": "Rate limit exceeded — slow down",
             "predictions": [], "count": 0, "inferenceMs": 0, "code": 429},
            status_code=429,
        )

    cfg = get_config()
    det_cfg = cfg["detection"]

    # Read image — 20MB hard limit
    image_bytes = await image.read()
    if len(image_bytes) > 20 * 1024 * 1024:
        return JSONResponse({"success": False, "error": "Image too large (20 MB max)",
            "predictions": [], "count": 0, "inferenceMs": 0, "code": 413}, status_code=413)

    size_kb = len(image_bytes) / 1024
    shape_idx = console.request_received(size_kb)

    if not engine.is_loaded:
        console.no_model()
        return JSONResponse({"success": False, "error": "No model loaded",
            "predictions": [], "count": 0, "inferenceMs": 0, "code": 500})

    # Resolve and clamp confidence threshold to [0.0, 1.0]
    threshold = min_confidence if min_confidence is not None else det_cfg["min_confidence"]
    threshold = max(0.0, min(1.0, float(threshold)))

    try:
        result = engine.detect(image_bytes)
        if result is None:
            return JSONResponse({
                "success": False,
                "error": "Inference failed",
                "predictions": [],
                "count": 0,
                "inferenceMs": 0,
                "code": 500,
            })

        response = result.to_cpai_response(
            min_confidence=threshold,
            filter_enabled=det_cfg["class_filter_enabled"],
            allowed_classes=det_cfg["allowed_classes"],
        )

        # Log detection result (paired with request via shape)
        console.detection_result(
            shape_idx=shape_idx,
            detections=response["predictions"],
            inference_ms=result.inference_ms,
        )

        # Record for header inference stats
        _record_inference(result.inference_ms)

        return JSONResponse(response)

    except Exception as e:
        logger.exception(f"Detection error: {e}")
        console.error(f"Detection error: {e}")
        logger.exception(f"Detection error: {e}")
        return JSONResponse({"success": False,
            "error": "Detection failed — check server logs",
            "predictions": [], "count": 0, "inferenceMs": 0, "code": 500})


@app.post("/v1/vision/alpr")
async def detect_alpr(
    request: Request,
    image: UploadFile = File(...),
    min_confidence: Optional[float] = Form(None),
):
    """
    License plate recognition endpoint — CodeProject.AI ALPR compatible.

    BlueIris posts an image here (configured as a separate AI server entry with
    path /v1/vision/alpr) and receives recognized plate strings with bounding
    boxes. Unauthenticated like /v1/vision/detection, so the same per-IP rate
    limit applies.
    """
    client_ip = request.client.host if request.client else "unknown"
    if not _rate_limit_check(client_ip):
        return JSONResponse(
            {"success": False, "error": "Rate limit exceeded — slow down",
             "predictions": [], "count": 0, "inferenceMs": 0, "code": 429},
            status_code=429,
        )

    image_bytes = await image.read()
    if len(image_bytes) > 20 * 1024 * 1024:
        return JSONResponse({"success": False, "error": "Image too large (20 MB max)",
            "predictions": [], "count": 0, "inferenceMs": 0, "code": 413}, status_code=413)

    size_kb = len(image_bytes) / 1024
    shape_idx = console.request_received(size_kb)

    if not alpr_engine.is_loaded:
        console.no_model()
        return JSONResponse({"success": False, "error": "ALPR not loaded",
            "predictions": [], "count": 0, "inferenceMs": 0, "code": 500})

    cfg = get_config()
    alpr_cfg = cfg.get("alpr", {})
    threshold = min_confidence if min_confidence is not None else alpr_cfg.get("min_confidence", 0.30)
    threshold = max(0.0, min(1.0, float(threshold)))

    try:
        results, inference_ms = alpr_engine.recognize(image_bytes)

        predictions = []
        for r in results:
            if r.confidence < threshold:
                continue
            predictions.append({
                "label": r.plate,
                "plate": r.plate,
                "confidence": round(r.confidence, 4),
                "x_min": int(r.x_min),
                "y_min": int(r.y_min),
                "x_max": int(r.x_max),
                "y_max": int(r.y_max),
            })

        response = {
            "success": True,
            "predictions": predictions,
            "count": len(predictions),
            "inferenceMs": round(inference_ms, 1),
            "processMs": round(inference_ms, 1),
            "moduleId": "ObjectifALPR",
            "moduleName": "Objectif.AI ALPR",
            "code": 200,
            "command": "alpr",
            "requestId": "",
        }

        console.detection_result(
            shape_idx=shape_idx,
            detections=predictions,
            inference_ms=inference_ms,
        )
        _record_inference(inference_ms)
        return JSONResponse(response)

    except Exception as e:
        logger.exception(f"ALPR error: {e}")
        console.error(f"ALPR error: {e}")
        return JSONResponse({"success": False,
            "error": "ALPR failed — check server logs",
            "predictions": [], "count": 0, "inferenceMs": 0, "code": 500})


@app.get("/v1/status")
async def status():
    """CPAI-compatible status endpoint."""
    return {
        "status": "OK",
        "version": "0.7.9",
        "platform": "Windows",
        "systemInfo": {
            "model": engine.model_info.name if engine.model_info else "None",
        }
    }


@app.get("/v1/vision/detection/list")
async def list_models_cpai():
    """CPAI-compatible model list."""
    return {"models": [engine.model_info.id] if engine.model_info else []}


# ---------------------------------------------------------------------------
# DASHBOARD API ENDPOINTS
# ---------------------------------------------------------------------------

@app.get("/api/status")
async def api_status(api_key: str = Depends(verify_api_key)):
    """Dashboard status summary."""
    cfg = get_config()
    uptime_s = int(time.time() - _start_time)
    hours, rem = divmod(uptime_s, 3600)
    mins, secs = divmod(rem, 60)

    return {
        "uptime": f"{hours:02d}:{mins:02d}:{secs:02d}",
        "model": model_to_dict(engine.model_info) if engine.model_info else None,
        "model_loading": engine.is_loading,
        "model_error": engine.load_error,
        "backend": cfg["detection"]["backend"],
        "legacy_mode": engine.legacy_mode,
        "hardware": hardware_info_to_dict(_hardware_info) if _hardware_info else {},
        "server_port": cfg["server"]["port"],
    }


@app.get("/api/hardware")
async def api_hardware(api_key: str = Depends(verify_api_key)):
    """Full hardware info."""
    if _hardware_info is None:
        raise HTTPException(status_code=503, detail="Hardware not yet detected")
    return hardware_info_to_dict(_hardware_info)


@app.get("/api/models")
async def api_models(api_key: str = Depends(verify_api_key)):
    """All available models with download status."""
    models = get_all_models()
    for m in models:
        m["downloaded"] = download_manager.is_downloaded(m["id"])
        m["is_active"] = (engine.model_info is not None and engine.model_info.id == m["id"])
        dl_status = download_manager.get_status(m["id"])
        m["download_state"] = dl_status.get("state", "idle")
        m["download_percent"] = dl_status.get("percent", 0)
    return {"models": models}


@app.post("/api/models/{model_id}/download")
async def api_download_model(model_id: str, api_key: str = Depends(verify_api_key)):
    """Start downloading a model."""
    info = get_model(model_id)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model_id}")
    if download_manager.is_downloaded(model_id):
        return {"success": True, "message": "Already downloaded"}
    console.system(f"Download started: {info.name} ({info.size_mb:.0f} MB)")
    ok = download_manager.download_model(model_id)
    if ok:
        return {"success": True, "message": f"Downloading {info.name}"}
    return {"success": False, "message": "Download already in progress"}


@app.delete("/api/models/{model_id}/download")
async def api_cancel_download(model_id: str, api_key: str = Depends(verify_api_key)):
    """Cancel an in-progress download."""
    download_manager.cancel(model_id)
    return {"success": True}


@app.delete("/api/models/{model_id}")
async def api_delete_model(model_id: str, api_key: str = Depends(verify_api_key)):
    """Delete a downloaded model file."""
    if engine.model_info and engine.model_info.id == model_id:
        engine.unload()
        update_config("detection.active_model", None)
        console.warning(f"Active model unloaded before deletion")
    ok = download_manager.delete_model(model_id)
    if ok:
        console.system(f"Model deleted: {model_id}")
        return {"success": True}
    raise HTTPException(status_code=404, detail="Model file not found")


@app.post("/api/models/{model_id}/activate")
async def api_activate_model(model_id: str, api_key: str = Depends(verify_api_key)):
    """Load and activate a model."""
    info = get_model(model_id)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model_id}")
    if not download_manager.is_downloaded(model_id):
        raise HTTPException(status_code=400, detail="Model not downloaded yet")

    backend = get_value("detection.backend", "cpu")
    console.system(f"Loading model: {info.name} on {backend.upper()}...")

    # Run in threadpool to avoid blocking event loop
    loop = asyncio.get_event_loop()
    ok = await loop.run_in_executor(None, engine.load_model, model_id, backend)

    if ok:
        update_config("detection.active_model", model_id)
        console.success(f"Model active: {info.name}")
        return {"success": True, "model": model_to_dict(info)}
    else:
        console.error(f"Failed to load {info.name}: {engine.load_error}")
        raise HTTPException(status_code=500, detail=engine.load_error)


@app.post("/api/models/unload")
async def api_unload_model(api_key: str = Depends(verify_api_key)):
    """Unload the current model."""
    engine.unload()
    update_config("detection.active_model", None)
    console.system("Model unloaded")
    return {"success": True}


@app.get("/api/config")
async def api_get_config(api_key: str = Depends(verify_api_key)):
    """Get current config."""
    cfg = get_config()
    return {
        "config": cfg,
        "coco_classes": get_coco_classes_grouped(),
    }


@app.post("/api/config")
async def api_update_config(body: dict, api_key: str = Depends(verify_api_key)):
    """Update one or more config values. body = {path: value, ...}"""
    from config import API_UPDATABLE_PATHS, validate_config_value
    updated = {}
    rejected = {}
    for path, value in body.items():
        if path not in API_UPDATABLE_PATHS:
            console.warning(f"Config update rejected — path not allowed: {path}")
            rejected[path] = "path not allowed"
            continue
        ok, err = validate_config_value(path, value)
        if not ok:
            console.warning(f"Config update rejected — {err}")
            rejected[path] = err
            continue
        update_config(path, value)
        updated[path] = value

    if updated:
        console.system(f"Config updated: {', '.join(updated.keys())}")

    return {"success": True, "updated": updated, "rejected": rejected}


@app.get("/api/log/history")
async def api_log_history(n: int = 200, api_key: str = Depends(verify_api_key)):
    """Get recent log history."""
    return {"entries": console.get_history(n)}


# ---------------------------------------------------------------------------
# SERVER CONTROL
# ---------------------------------------------------------------------------

@app.post("/api/server/restart")
async def api_server_restart(api_key: str = Depends(verify_api_key)):
    """Restart the server. Only works when running as a scheduled task; otherwise
    returns an error so the dashboard can tell the user to restart manually."""
    # Check up front whether the scheduled task exists. If not, the previous
    # behavior was to silently SIGKILL ourselves and pretend we'd come back —
    # which we wouldn't. Better to be honest.
    result = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: subprocess.run(
            ["schtasks", "/query", "/tn", "ObjectifAI"],
            capture_output=True, timeout=5
        )
    )
    if result.returncode != 0:
        return JSONResponse(
            {"success": False,
             "error": "Not running as a scheduled task — auto-restart unavailable. Stop the server (Ctrl+C) and re-run start.bat."},
            status_code=400,
        )

    console.system("Restart requested — restarting...")
    _startup_log("Restart requested from dashboard")

    async def _do_restart():
        await asyncio.sleep(1.5)
        # Restart via Task Scheduler: end the task then run it again.
        # Use a detached shell command so it outlives this process.
        subprocess.Popen(
            'cmd /c schtasks /end /tn "ObjectifAI" & timeout /t 3 /nobreak & schtasks /run /tn "ObjectifAI"',
            shell=False,
            creationflags=subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS,
            close_fds=True
        )

    asyncio.create_task(_do_restart())
    return {"success": True, "message": "Restarting..."}


@app.get("/api/service/status")
async def api_service_status(api_key: str = Depends(verify_api_key)):
    """Check if the ObjectifAI scheduled task exists."""
    import subprocess as _sp
    try:
        result = _sp.run(
            ["schtasks", "/query", "/tn", "ObjectifAI"],
            capture_output=True, text=True, timeout=5
        )
        exists = result.returncode == 0
        return {"exists": exists}
    except Exception:
        return {"exists": False}


@app.post("/api/service/remove")
async def api_service_remove(api_key: str = Depends(verify_api_key)):
    """Remove the ObjectifAI scheduled task."""
    import subprocess as _sp
    try:
        result = _sp.run(
            ["schtasks", "/delete", "/tn", "ObjectifAI", "/f"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            console.system("Startup service removed")
            _startup_log("Startup service removed by user")
            return {"success": True}
        else:
            return {"success": False, "error": result.stderr.strip()}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# BACKEND SWITCH  (switch hardware + reload model)
# ---------------------------------------------------------------------------

@app.post("/api/backend/switch")
async def api_switch_backend(body: dict, api_key: str = Depends(verify_api_key)):
    """Switch compute backend and reload active model if one is loaded."""
    backend = body.get("backend")
    if backend not in ("auto", "cpu", "cuda", "directml", "openvino", "rocm"):
        raise HTTPException(status_code=400, detail=f"Unknown backend: {backend}")

    prev_model_id = engine.model_info.id if engine.model_info else None
    update_config("detection.backend", backend)
    console.system(f"Backend switching to: {backend.upper()}")

    if prev_model_id:
        engine.unload()
        console.system(f"Reloading model on {backend.upper()}...")
        loop = asyncio.get_event_loop()
        ok = await loop.run_in_executor(None, engine.load_model, prev_model_id, backend)
        if ok:
            console.success(f"Model reloaded on {backend.upper()}: {engine.model_info.name}")
            return {"success": True, "reloaded": True, "model": engine.model_info.name}
        else:
            console.error(f"Failed to reload on {backend.upper()}: {engine.load_error}")
            raise HTTPException(status_code=500, detail=engine.load_error)
    else:
        console.system(f"Backend set to {backend.upper()} (no model loaded)")
        return {"success": True, "reloaded": False}


# ---------------------------------------------------------------------------
# ALPR MANAGEMENT
# ---------------------------------------------------------------------------

# Available fast-alpr model choices surfaced in the dashboard. Detector sizes
# trade speed for accuracy (256 fastest, 608 most accurate).
ALPR_DETECTOR_MODELS = [
    "yolo-v9-t-256-license-plate-end2end",
    "yolo-v9-t-384-license-plate-end2end",
    "yolo-v9-t-512-license-plate-end2end",
    "yolo-v9-t-640-license-plate-end2end",
    "yolo-v9-s-608-license-plate-end2end",
]
ALPR_OCR_MODELS = [
    "cct-xs-v2-global-model",
    "cct-s-v1-global-model",
    "global-plates-mobile-vit-v2-model",
]


@app.get("/api/alpr/status")
async def api_alpr_status(api_key: str = Depends(verify_api_key)):
    """ALPR pipeline status for the dashboard."""
    # fast-alpr availability — drives whether the UI shows an install prompt.
    try:
        import fast_alpr  # noqa: F401
        installed = True
    except ImportError:
        installed = False

    return {
        "installed": installed,
        "loaded": alpr_engine.is_loaded,
        "loading": alpr_engine.is_loading,
        "error": alpr_engine.load_error,
        "active": bool(get_value("alpr.active")),
        "models": alpr_engine.active_models,
        "min_confidence": get_value("alpr.min_confidence", 0.30),
        "available_detectors": ALPR_DETECTOR_MODELS,
        "available_ocr": ALPR_OCR_MODELS,
    }


@app.post("/api/alpr/load")
async def api_alpr_load(body: dict, api_key: str = Depends(verify_api_key)):
    """Load (or reload) the ALPR pipeline. First load downloads weights."""
    detector = body.get("detector_model") or get_value("alpr.detector_model")
    ocr = body.get("ocr_model") or get_value("alpr.ocr_model")

    if detector not in ALPR_DETECTOR_MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown detector model: {detector}")
    if ocr not in ALPR_OCR_MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown OCR model: {ocr}")

    loop = asyncio.get_event_loop()
    ok = await loop.run_in_executor(None, alpr_engine.load, detector, ocr)
    if not ok:
        raise HTTPException(status_code=500, detail=alpr_engine.load_error or "ALPR load failed")

    update_config("alpr.detector_model", detector)
    update_config("alpr.ocr_model", ocr)
    update_config("alpr.active", True)
    console.success(f"ALPR ready: {detector} + {ocr}")
    return {"success": True, "models": alpr_engine.active_models}


@app.post("/api/alpr/unload")
async def api_alpr_unload(api_key: str = Depends(verify_api_key)):
    """Unload the ALPR pipeline and stop auto-loading it on startup."""
    alpr_engine.unload()
    update_config("alpr.active", False)
    console.system("ALPR pipeline unloaded")
    return {"success": True}


# ---------------------------------------------------------------------------
# PACKAGE INSTALLER
# ---------------------------------------------------------------------------

@app.get("/api/install/status")
async def api_install_status(api_key: str = Depends(verify_api_key)):
    """Check install prerequisites for each backend."""
    hw = hardware_info_to_dict(_hardware_info) if _hardware_info else {}
    cuda_version = hw.get("nvidia", {}).get("cuda_version", "")

    # Auto-clear if stuck
    if _is_install_stuck():
        _install_lock_clear()
        console.warning("Install lock was stuck — cleared automatically")

    torch_available = False
    torch_has_cuda = False
    torch_version = ""
    try:
        import torch
        torch_available = True
        torch_version = torch.__version__
        torch_has_cuda = torch.cuda.is_available()
    except ImportError:
        pass

    openvino_available = False
    openvino_version = ""
    try:
        import openvino
        openvino_available = True
        openvino_version = openvino.__version__
    except ImportError:
        pass

    onnxruntime_gpu = False
    onnxruntime_directml = False
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        onnxruntime_gpu = "CUDAExecutionProvider" in providers
        onnxruntime_directml = "DmlExecutionProvider" in providers
    except ImportError:
        pass

    wheel_url = _cuda_wheel_url(cuda_version) if cuda_version else ""
    install_running = _active_install is not None and not _is_install_stuck()

    return {
        "torch": {"available": torch_available, "version": torch_version, "has_cuda": torch_has_cuda},
        "openvino": {"available": openvino_available, "version": openvino_version},
        "onnxruntime_gpu": onnxruntime_gpu,
        "onnxruntime_directml": onnxruntime_directml,
        "cuda_version": cuda_version,
        "cuda_wheel_url": wheel_url,
        "install_running": install_running,
    }


@app.post("/api/install/reset")
async def api_install_reset(api_key: str = Depends(verify_api_key)):
    """Force-clear a stuck install lock."""
    _install_lock_clear()
    console.warning("Install lock force-cleared by user")
    return {"success": True}


@app.post("/api/install/cuda")
async def api_install_cuda(api_key: str = Depends(verify_api_key)):
    """Install PyTorch CUDA then onnxruntime-gpu sequentially."""
    if _active_install is not None and not _is_install_stuck():
        raise HTTPException(status_code=409, detail="Another install is already running")
    if _is_install_stuck():
        _install_lock_clear()

    hw = hardware_info_to_dict(_hardware_info) if _hardware_info else {}
    cuda_version = hw.get("nvidia", {}).get("cuda_version", "")
    if not cuda_version:
        raise HTTPException(status_code=400, detail="No CUDA version detected")
    wheel_url = _cuda_wheel_url(cuda_version)
    tag = wheel_url.split("/")[-1]  # e.g. "cu124"

    # Install PyTorch CUDA only.
    # onnxruntime-gpu must be installed manually by the user due to Python
    # environment differences — the dashboard will show the manual command.
    steps = [
        (["torch", "torchvision", "--index-url", wheel_url],
         f"PyTorch CUDA ({tag})"),
    ]
    import threading
    threading.Thread(
        target=_run_install_chain,
        args=(steps,),
        daemon=True,
    ).start()
    return {"success": True, "wheel_url": wheel_url, "steps": 1}


@app.post("/api/install/onnxruntime-gpu")
async def api_install_onnxruntime_gpu(api_key: str = Depends(verify_api_key)):
    """Install onnxruntime-gpu only (use when PyTorch CUDA already installed)."""
    if _active_install is not None and not _is_install_stuck():
        raise HTTPException(status_code=409, detail="Another install is already running")
    if _is_install_stuck():
        _install_lock_clear()
    import threading
    threading.Thread(
        target=_run_install_chain,
        args=([( ["onnxruntime-gpu"], "onnxruntime-gpu")],),
        daemon=True,
    ).start()
    return {"success": True}


@app.post("/api/install/openvino")
async def api_install_openvino(api_key: str = Depends(verify_api_key)):
    """Install OpenVINO + onnxruntime-openvino."""
    if _active_install is not None and not _is_install_stuck():
        raise HTTPException(status_code=409, detail="Another install is already running")
    if _is_install_stuck():
        _install_lock_clear()
    steps = [(["openvino", "onnxruntime-openvino"], "OpenVINO + onnxruntime-openvino")]
    import threading
    threading.Thread(
        target=_run_install_chain,
        args=(steps,),
        daemon=True,
    ).start()
    return {"success": True}


@app.post("/api/install/directml")
async def api_install_directml(api_key: str = Depends(verify_api_key)):
    """
    Install onnxruntime-directml for GPU inference on any DirectX 12 GPU
    (NVIDIA, AMD, or Intel) on Windows.

    Note: ONNX Runtime ships its CPU, CUDA (onnxruntime-gpu), and DirectML
    builds as separate, mutually exclusive packages. Installing DirectML
    replaces any existing onnxruntime / onnxruntime-gpu install. The chain
    therefore uninstalls the conflicting builds first.
    """
    if _active_install is not None and not _is_install_stuck():
        raise HTTPException(status_code=409, detail="Another install is already running")
    if _is_install_stuck():
        _install_lock_clear()

    steps = [
        (["--uninstall", "onnxruntime", "onnxruntime-gpu", "onnxruntime-openvino"],
         "Removing conflicting ONNX Runtime builds"),
        (["onnxruntime-directml"], "onnxruntime-directml"),
    ]
    import threading
    threading.Thread(
        target=_run_install_chain,
        args=(steps,),
        daemon=True,
    ).start()
    return {"success": True}


# ---------------------------------------------------------------------------
# WEBSOCKET
# ---------------------------------------------------------------------------

@app.get("/api/dependencies")
async def api_dependencies(api_key: str = Depends(verify_api_key)):
    """Return full dependency status for the Dependencies tab."""
    return {
        "python": get_python_info(),
        "dependencies": check_all_dependencies(),
    }


@app.post("/api/dependencies/install/{dep_id}")
async def api_install_dependency(dep_id: str, api_key: str = Depends(verify_api_key)):
    """Install a dependency by its catalog ID."""
    from dependencies import DEPENDENCY_CATALOG
    dep = next((d for d in DEPENDENCY_CATALOG if d["id"] == dep_id), None)
    if dep is None:
        raise HTTPException(status_code=404, detail=f"Unknown dependency: {dep_id}")
    if dep.get("manual_note") and not dep.get("install_cmd"):
        raise HTTPException(status_code=400, detail="Manual install required — see note")
    if not dep.get("install_cmd"):
        raise HTTPException(status_code=400, detail="No install command defined")
    if _active_install is not None and not _is_install_stuck():
        raise HTTPException(status_code=409, detail="Another install is already running")
    if _is_install_stuck():
        _install_lock_clear()

    packages = dep["install_cmd"]
    label = dep["name"]
    console.system(f"Installing dependency: {label}")
    import threading
    threading.Thread(
        target=_run_install_chain,
        args=([(packages, label)],),
        daemon=True,
    ).start()
    return {"success": True, "label": label}


@app.get("/api/auth/check")
async def api_auth_check(api_key: str = Depends(verify_api_key)):
    """Validate API key — used by dashboard on load."""
    return {"valid": True}


@app.get("/api/auth/setup")
async def api_auth_setup():
    """
    First-run setup — unprotected so dashboard can show the key before it is known.
    On first run: generates the key and returns it ONCE so the overlay can display it.
    On subsequent calls: returns first_run=False, key omitted.
    This endpoint is the only place the key is ever returned unauthenticated.
    """
    first = is_first_run()
    if first:
        key = get_or_create_api_key()
        return {"first_run": True, "key": key}
    return {"first_run": False}


@app.post("/api/auth/ws-ticket")
async def api_auth_ws_ticket(api_key: str = Depends(verify_api_key)):
    """
    Exchange the API key for a short-lived single-use WebSocket ticket.
    The dashboard uses this so the API key never appears in WS URLs (where it
    would otherwise leak into access logs and browser history).
    """
    return {"ticket": issue_ws_ticket(), "ttl": _WS_TICKET_TTL_S}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    # Prefer single-use ticket; fall back to legacy ?key= for direct WS clients.
    ticket = websocket.query_params.get("ticket", "")
    if ticket:
        if not consume_ws_ticket(ticket):
            await websocket.close(code=4401, reason="Invalid or expired ticket")
            return
    else:
        if not await verify_websocket_key(websocket):
            await websocket.close(code=4401, reason="Unauthorized")
            return

    await websocket.accept()
    console.add_client(websocket)
    download_manager.add_client(websocket)
    _inference_ws_clients.add(websocket)
    _install_ws_clients.add(websocket)

    try:
        # Send history to new client
        await console.send_history(websocket, last_n=200)

        # Send current inference stats immediately on connect
        if _inference_times:
            avg = sum(_inference_times) / len(_inference_times)
            await websocket.send_text(json.dumps({
                "type": "inference_stats",
                "last_ms": round(_last_inference_ms, 1),
                "avg_ms": round(avg, 1),
            }))

        # Keep alive — listen for pings from client
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30)
                if data == "ping":
                    await websocket.send_text('{"type":"pong"}')
            except asyncio.TimeoutError:
                # Send keepalive
                try:
                    await websocket.send_text('{"type":"ping"}')
                except Exception:
                    break

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.debug(f"WebSocket error: {e}")
    finally:
        console.remove_client(websocket)
        download_manager.remove_client(websocket)
        _inference_ws_clients.discard(websocket)
        _install_ws_clients.discard(websocket)


# ---------------------------------------------------------------------------
# DASHBOARD  (serve index.html for all non-API routes)
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """
    Serve the dashboard with the API key injected directly into the page.
    The key is substituted server-side so the user never has to enter it manually.
    Anyone who can reach this page on the LAN is already trusted enough to use it.
    """
    index = STATIC_DIR / "index.html"
    if index.exists():
        html = index.read_text(encoding="utf-8")
        # Inject the API key so the dashboard authenticates automatically
        api_key = get_or_create_api_key()
        if not api_key:
            logger.error("API key is empty — config.yaml may not be writable")
            return HTMLResponse("<html><body style='font-family:monospace;background:#0d0d0d;color:#e8e8e8;padding:40px'><h2 style='color:#ff4d4d'>Startup error</h2><p>Could not generate API key. Check that config.yaml is writable in the app directory.</p></body></html>")
        count = html.count("__OBJECTIF_API_KEY__")
        logger.info(f"Serving dashboard — injecting API key ({count} substitutions)")
        html = html.replace("__OBJECTIF_API_KEY__", api_key)
        return HTMLResponse(html)
    return HTMLResponse("""
        <html><body style="font-family:monospace;background:#0d0d0d;color:#e8e8e8;padding:40px">
        <h2 style="color:#f0a500">Dashboard not found</h2>
        <p>The <code>static/</code> folder must be in the same directory as <code>main.py</code>.</p>
        <pre style="color:#888">  Objectif.AI/
  &#x251c;&#x2500;&#x2500; main.py
  &#x2514;&#x2500;&#x2500; static/
      &#x2514;&#x2500;&#x2500; index.html   &lt;-- this file is missing</pre>
        </body></html>
    """)


# Mount static files AFTER routes so routes take priority
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = get_config()
    port = cfg["server"]["port"]
    host = cfg["server"]["host"]
    log_level = cfg["server"]["log_level"]

    print(f"\n  Objectif.AI")
    print(f"  Dashboard: http://localhost:{port}")
    print(f"  BlueIris API endpoint: http://localhost:{port}/v1/vision/detection")
    print(f"  Press Ctrl+C to stop\n")

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        log_level=log_level,
        reload=False,
    )
