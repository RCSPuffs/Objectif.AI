"""
Configuration management for Objectif.AI.
Reads and writes config.yaml. All settings default gracefully.
"""

import os
import logging
from pathlib import Path
from typing import Any, Optional
import yaml

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent / "config.yaml"

# COCO 80-class names grouped by category
COCO_CLASSES = {
    "People": ["person"],
    "Vehicles": [
        "bicycle", "car", "motorcycle", "airplane", "bus", "train",
        "truck", "boat"
    ],
    "Animals": [
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
        "bear", "zebra", "giraffe"
    ],
    "Outdoor": [
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench"
    ],
    "Sports": [
        "frisbee", "skis", "snowboard", "sports ball", "kite",
        "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket"
    ],
    "Kitchen": [
        "bottle", "wine glass", "cup", "fork", "knife", "spoon",
        "bowl"
    ],
    "Food": [
        "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
        "hot dog", "pizza", "donut", "cake"
    ],
    "Furniture": [
        "chair", "couch", "potted plant", "bed", "dining table", "toilet"
    ],
    "Electronics": [
        "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator"
    ],
    "Miscellaneous": [
        "backpack", "umbrella", "handbag", "tie", "suitcase",
        "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
    ],
}

# Flat list of all 80 classes in order
ALL_COCO_CLASSES = [cls for group in COCO_CLASSES.values() for cls in group]

DEFAULT_CONFIG = {
    "server": {
        "host": "0.0.0.0",
        "port": 32168,
        "log_level": "info",
    },
    "auth": {
        "api_key": "",   # auto-generated on first run
    },
    "detection": {
        "active_model": None,           # model id string or null
        "backend": "auto",              # auto, cpu, cuda, directml, openvino, rocm
        "min_confidence": 0.30,         # global minimum confidence threshold
        "class_filter_enabled": False,  # if False, return all classes
        "allowed_classes": ALL_COCO_CLASSES,  # used when filter is enabled
    },
    "alpr": {
        "active": False,                # whether the ALPR pipeline auto-loads on startup
        "min_confidence": 0.30,         # minimum plate-text confidence returned
        "detector_model": "yolo-v9-t-384-license-plate-end2end",
        "ocr_model": "cct-xs-v2-global-model",
    },
    "console": {
        "buffer_size": 1000,            # lines to keep in memory
    },
    "ui": {
        "theme": "dark",
        "inference_update_ms": 1000,    # how often header inference display updates (ms)
        "detection_expand": True,       # show each detection on its own indented line
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning new dict."""
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def load_config() -> dict:
    """Load config from disk, merging with defaults for any missing keys."""
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                on_disk = yaml.safe_load(f) or {}
            config = _deep_merge(DEFAULT_CONFIG, on_disk)
            logger.info("Config loaded")
            return config
        except Exception as e:
            logger.warning(f"Failed to load config ({e}), using defaults")
    else:
        logger.info("No config.yaml found, using defaults")
    return dict(DEFAULT_CONFIG)


def save_config(config: dict) -> bool:
    """Save config dict to disk."""
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        logger.info("Config saved")
        return True
    except Exception as e:
        logger.error(f"Failed to save config: {e}")
        return False


def get_coco_classes_grouped() -> dict:
    """Return COCO classes organized by group."""
    return COCO_CLASSES


def get_all_coco_classes() -> list:
    return ALL_COCO_CLASSES


# ---------------------------------------------------------------------------
# Runtime config object — loaded once, mutated in-place, saved on changes
# ---------------------------------------------------------------------------
_config: Optional[dict] = None


def get_config() -> dict:
    global _config
    if _config is None:
        _config = load_config()
    return _config


# Exact allowlist of paths that can be updated via the API.
# Internal paths (auth.api_key, etc.) are updated directly via update_config()
# and are NOT in this list — they cannot be changed via the /api/config endpoint.
API_UPDATABLE_PATHS: set = {
    "detection.min_confidence",
    "detection.active_model",
    "detection.backend",
    "detection.class_filter_enabled",
    "detection.allowed_classes",
    "alpr.min_confidence",
    "console.buffer_size",
    "ui.theme",
    "ui.inference_update_ms",
    "ui.detection_expand",
    "server.port",
    "server.log_level",
}

# Per-path value validators. Each returns True if the value is acceptable.
# Used by /api/config to reject malformed updates before they reach disk.
_VALIDATORS = {
    "detection.min_confidence":
        lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and 0.0 <= float(v) <= 1.0,
    "detection.active_model":
        lambda v: v is None or (isinstance(v, str) and 0 < len(v) <= 100),
    "detection.backend":
        lambda v: v in ("auto", "cpu", "cuda", "directml", "openvino", "rocm"),
    "detection.class_filter_enabled":
        lambda v: isinstance(v, bool),
    "detection.allowed_classes":
        lambda v: isinstance(v, list) and all(isinstance(x, str) for x in v) and len(v) <= 500,
    "alpr.min_confidence":
        lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and 0.0 <= float(v) <= 1.0,
    "console.buffer_size":
        lambda v: isinstance(v, int) and not isinstance(v, bool) and 100 <= v <= 100_000,
    "ui.theme":
        lambda v: isinstance(v, str) and 0 < len(v) <= 50,
    "ui.inference_update_ms":
        lambda v: isinstance(v, int) and not isinstance(v, bool) and 100 <= v <= 60_000,
    "ui.detection_expand":
        lambda v: isinstance(v, bool),
    "server.port":
        lambda v: isinstance(v, int) and not isinstance(v, bool) and 1024 <= v <= 65535,
    "server.log_level":
        lambda v: v in ("debug", "info", "warning", "error", "critical"),
}


def validate_config_value(path: str, value: Any) -> tuple:
    """
    Returns (ok, error_message). Used by /api/config before persisting.
    """
    if path not in API_UPDATABLE_PATHS:
        return False, f"Path not API-updatable: {path}"
    validator = _VALIDATORS.get(path)
    if validator is None:
        return False, f"No validator defined for: {path}"
    try:
        if not validator(value):
            return False, f"Invalid value for {path}"
    except Exception as e:
        return False, f"Validation error for {path}: {e}"
    return True, ""


def update_config(path: str, value: Any) -> dict:
    """
    Update a config value by dot-separated path and save.
    e.g. update_config("detection.min_confidence", 0.45)
    """
    config = get_config()
    keys = path.split(".")
    node = config
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value
    save_config(config)
    return config


def get_value(path: str, default: Any = None) -> Any:
    """Get a config value by dot-separated path."""
    config = get_config()
    keys = path.split(".")
    node = config
    try:
        for key in keys:
            node = node[key]
        return node
    except (KeyError, TypeError):
        return default
