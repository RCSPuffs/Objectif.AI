"""
Authentication for Objectif.AI.
Generates and manages a simple API key stored in config.yaml.
Management endpoints require the key. The BlueIris detection endpoint
is intentionally unauthenticated since BlueIris cannot send auth headers.
"""

import secrets
import logging
from typing import Optional
from fastapi import Header, HTTPException, status, WebSocket
from config import get_config, update_config

logger = logging.getLogger(__name__)

_API_KEY_LENGTH = 32  # bytes -> 64 hex chars


def get_or_create_api_key() -> str:
    """Return the existing API key or generate and save a new one."""
    from config import get_config as _get_config, update_config as _update_config, _config as _cfg_cache
    cfg = _get_config()
    key = cfg.get("auth", {}).get("api_key", "")
    if not key:
        key = secrets.token_hex(_API_KEY_LENGTH)
        # Update both the in-memory cache and disk
        _update_config("auth.api_key", key)
        # Also patch the cached dict directly so subsequent calls in this
        # process see the new key without re-reading from disk
        cfg.setdefault("auth", {})["api_key"] = key
        logger.info("New API key generated and saved to config.yaml")
    return key


def verify_api_key(x_api_key: Optional[str] = Header(default=None)) -> str:
    """
    FastAPI dependency for header-based auth.
    Use as: Depends(verify_api_key)
    """
    expected = get_or_create_api_key()
    if not x_api_key or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key. Include X-Api-Key header.",
        )
    return x_api_key


async def verify_websocket_key(websocket: WebSocket) -> bool:
    """
    Verify API key from WebSocket query parameter.
    Returns True if valid, False otherwise.
    Browsers cannot set headers on WebSocket connections so we use ?key=
    """
    expected = get_or_create_api_key()
    provided = websocket.query_params.get("key", "")
    return secrets.compare_digest(provided, expected)


def is_first_run() -> bool:
    """True if no API key exists yet — used to show first-run UI."""
    cfg = get_config()
    return not cfg.get("auth", {}).get("api_key", "")
