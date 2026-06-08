"""
Console buffer for Objectif.AI.
Maintains a circular in-memory log buffer and broadcasts to WebSocket clients.
Thread-safe. Designed to be used from async FastAPI context.
"""

import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional
import itertools

logger = logging.getLogger(__name__)


class LogLevel(str, Enum):
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    REQUEST = "request"       # Incoming image from BlueIris
    DETECTION = "detection"   # Detection result
    SYSTEM = "system"


# Shapes cycled through for request/detection pairing
# Fixed-width via CSS — these are the Unicode symbols, CSS handles alignment
SHAPES = ["●", "■", "▲", "◆", "★", "⬟", "◉", "⬡", "▼", "◈"]

_shape_counter = itertools.cycle(range(len(SHAPES)))
_shape_lock = asyncio.Lock()


@dataclass
class LogEntry:
    id: int
    timestamp: float
    level: str          # LogLevel value
    message: str
    shape_index: Optional[int] = None   # index into SHAPES, for request/detection pairs
    inference_ms: Optional[float] = None
    detections: Optional[list] = None   # list of {label, confidence} dicts


_entry_counter = itertools.count(1)


def _make_entry(level: LogLevel, message: str, **kwargs) -> LogEntry:
    return LogEntry(
        id=next(_entry_counter),
        timestamp=time.time(),
        level=level.value,
        message=message,
        **kwargs,
    )


def entry_to_dict(entry: LogEntry) -> dict:
    d = asdict(entry)
    d["shape"] = SHAPES[entry.shape_index] if entry.shape_index is not None else None
    return d


class ConsoleBuffer:
    """
    Thread-safe circular buffer of log entries.
    Broadcasts new entries to all connected WebSocket clients.
    """

    def __init__(self, max_size: int = 1000):
        self._max_size = max_size
        self._buffer: deque[LogEntry] = deque(maxlen=max_size)
        self._clients: set = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._next_shape_idx: int = 0

    def resize(self, new_size: int):
        self._max_size = new_size
        old = list(self._buffer)
        self._buffer = deque(old[-new_size:], maxlen=new_size)

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop

    # ------------------------------------------------------------------
    # Client management
    # ------------------------------------------------------------------

    def add_client(self, ws):
        self._clients.add(ws)
        logger.debug(f"Console client connected. Total: {len(self._clients)}")

    def remove_client(self, ws):
        self._clients.discard(ws)
        logger.debug(f"Console client disconnected. Total: {len(self._clients)}")

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _next_shape(self) -> int:
        idx = self._next_shape_idx
        self._next_shape_idx = (self._next_shape_idx + 1) % len(SHAPES)
        return idx

    def _append_and_broadcast(self, entry: LogEntry):
        self._buffer.append(entry)
        self._broadcast(entry)

    def _broadcast(self, entry: LogEntry):
        if not self._clients:
            return
        payload = json.dumps({"type": "log", "entry": entry_to_dict(entry)})
        dead = set()
        for ws in list(self._clients):
            try:
                # Schedule coroutine on the event loop if available
                if self._loop and self._loop.is_running():
                    asyncio.run_coroutine_threadsafe(
                        _safe_send(ws, payload), self._loop
                    )
                else:
                    # Sync fallback — best effort
                    pass
            except Exception:
                dead.add(ws)
        self._clients -= dead

    async def _broadcast_async(self, entry: LogEntry):
        if not self._clients:
            return
        payload = json.dumps({"type": "log", "entry": entry_to_dict(entry)})
        dead = set()
        for ws in list(self._clients):
            try:
                await _safe_send(ws, payload)
            except Exception:
                dead.add(ws)
        self._clients -= dead

    # ------------------------------------------------------------------
    # Public log methods
    # ------------------------------------------------------------------

    def info(self, message: str):
        self._append_and_broadcast(_make_entry(LogLevel.INFO, message))

    def success(self, message: str):
        self._append_and_broadcast(_make_entry(LogLevel.SUCCESS, message))

    def warning(self, message: str):
        self._append_and_broadcast(_make_entry(LogLevel.WARNING, message))

    def error(self, message: str):
        self._append_and_broadcast(_make_entry(LogLevel.ERROR, message))

    def system(self, message: str):
        self._append_and_broadcast(_make_entry(LogLevel.SYSTEM, message))

    def request_received(self, size_kb: float) -> int:
        """Log an incoming image request. Returns the shape_index for pairing."""
        shape_idx = self._next_shape()
        entry = _make_entry(
            LogLevel.REQUEST,
            f"Image received — {size_kb:.1f} KB",
            shape_index=shape_idx,
        )
        self._append_and_broadcast(entry)
        return shape_idx

    def detection_result(
        self,
        shape_idx: int,
        detections: list,
        inference_ms: float,
    ):
        """Log a detection result, paired with a prior request via shape_idx."""
        # Sort by confidence descending — frontend uses this order for display
        sorted_dets = sorted(detections, key=lambda d: d["confidence"], reverse=True)

        if sorted_dets:
            # Summary line always shows count + time
            msg = f"Detected {len(sorted_dets)} item{'s' if len(sorted_dets)!=1 else ''} — {inference_ms:.1f} ms"
        else:
            msg = f"No detections — {inference_ms:.1f} ms"

        entry = _make_entry(
            LogLevel.DETECTION,
            msg,
            shape_index=shape_idx,
            inference_ms=inference_ms,
            detections=sorted_dets,  # always full sorted list
        )
        self._append_and_broadcast(entry)

    def no_model(self):
        self._append_and_broadcast(
            _make_entry(LogLevel.WARNING, "No model loaded — request ignored")
        )

    # ------------------------------------------------------------------
    # History retrieval
    # ------------------------------------------------------------------

    def get_history(self, last_n: Optional[int] = None) -> list[dict]:
        entries = list(self._buffer)
        if last_n is not None:
            entries = entries[-last_n:]
        return [entry_to_dict(e) for e in entries]

    async def send_history(self, ws, last_n: int = 200):
        """Send recent history to a newly connected WebSocket client."""
        history = self.get_history(last_n)
        payload = json.dumps({"type": "history", "entries": history})
        try:
            await ws.send_text(payload)
        except Exception as e:
            logger.debug(f"Failed to send history: {e}")


async def _safe_send(ws, payload: str):
    try:
        await ws.send_text(payload)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Module-level singleton — imported everywhere
# ---------------------------------------------------------------------------
console = ConsoleBuffer()
