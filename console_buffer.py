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
    ALPR = "alpr"             # License plate recognition result
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
    model: Optional[str] = None         # model/engine that served this inference
    backend: Optional[str] = None       # active backend badge: CUDA/CPU/DML/OpenVINO
    timing: Optional[dict] = None       # {preprocess, inference, postprocess} or {decode, inference}
    source: Optional[str] = None        # endpoint type: "detection" | "alpr" | "onnx"


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
        # Rolling 24h plate-read tracker (in-memory; resets on restart).
        # deque of (timestamp, plate_text); pruned to 24h on each ALPR result.
        self._plate_reads: deque = deque()   # (timestamp, plate_text, confidence, thumbnail)
        self._last_plate: Optional[str] = None

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
        model: Optional[str] = None,
        backend: Optional[str] = None,
        timing: Optional[dict] = None,
        source: str = "detection",
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
            model=model,
            backend=backend,
            timing=timing,
            source=source,
        )
        self._append_and_broadcast(entry)

    def alpr_result(
        self,
        shape_idx: int,
        plates: list,
        inference_ms: float,
        model: Optional[str] = None,
        backend: Optional[str] = None,
        timing: Optional[dict] = None,
    ):
        """
        Log a license-plate recognition result, paired with a prior request.
        `plates` is a list of {label/plate, confidence} dicts (the ALPR predictions).
        Records each read into the rolling 24h tracker.
        """
        sorted_plates = sorted(plates, key=lambda d: d["confidence"], reverse=True)

        # Record reads into the 24h tracker (total reads, not unique).
        # Each read: (timestamp, plate_text, confidence, thumbnail).
        # Thumbnails are capped to the most recent THUMB_CAP reads to bound
        # memory — older reads keep text/confidence but drop the image.
        now = time.time()
        for p in sorted_plates:
            text = p.get("plate") or p.get("label") or "?"
            conf = float(p.get("confidence", 0.0))
            thumb = p.get("thumbnail", "") or ""
            self._plate_reads.append((now, text, conf, thumb))
            self._last_plate = text
        self._prune_plate_reads(now)
        self._prune_plate_thumbnails()

        if sorted_plates:
            plate_list = ", ".join((p.get("plate") or p.get("label") or "?") for p in sorted_plates)
            msg = f"Plate: {plate_list} — {inference_ms:.1f} ms"
        else:
            msg = f"No plate found — {inference_ms:.1f} ms"

        entry = _make_entry(
            LogLevel.ALPR,
            msg,
            shape_index=shape_idx,
            inference_ms=inference_ms,
            detections=sorted_plates,
            model=model,
            backend=backend,
            timing=timing,
            source="alpr",
        )
        self._append_and_broadcast(entry)

    def _prune_plate_thumbnails(self):
        """
        Keep base64 thumbnails only for the most recent THUMB_CAP plate reads.
        Strips the image off older entries to bound memory while preserving the
        text/confidence/timestamp record.
        """
        THUMB_CAP = 100
        thumbed = [i for i, r in enumerate(self._plate_reads) if len(r) > 3 and r[3]]
        if len(thumbed) <= THUMB_CAP:
            return
        # Strip thumbnails from all but the last THUMB_CAP thumbed entries
        to_strip = set(thumbed[:-THUMB_CAP])
        for i in to_strip:
            r = self._plate_reads[i]
            self._plate_reads[i] = (r[0], r[1], r[2], "")

    def _prune_plate_reads(self, now: Optional[float] = None):
        """Drop plate reads older than 24h. Cheap — runs on each ALPR result."""
        if now is None:
            now = time.time()
        cutoff = now - 86400
        while self._plate_reads and self._plate_reads[0][0] < cutoff:
            self._plate_reads.popleft()

    def alpr_stats(self) -> dict:
        """Header readout: last plate seen and total reads in the last 24h."""
        self._prune_plate_reads()
        return {
            "last_plate": self._last_plate,
            "count_24h": len(self._plate_reads),
        }

    def get_plate_history(self, limit: int = 1000) -> list:
        """
        Return the last `limit` plate reads, newest first.
        Each entry: {plate, timestamp, confidence, thumbnail}.
        Reads are kept for 24h in memory and reset on server restart.
        Only the most recent ~100 reads carry a thumbnail (older ones are "").
        """
        self._prune_plate_reads()
        reads = list(self._plate_reads)
        reads.reverse()
        out = []
        for r in reads[:limit]:
            thumb = r[3] if len(r) > 3 else ""
            out.append({
                "plate": r[1],
                "timestamp": r[0],
                "confidence": round(r[2], 4),
                "thumbnail": thumb,
            })
        return out

    def no_model(self):
        self._append_and_broadcast(
            _make_entry(LogLevel.WARNING, "No model loaded — request ignored")
        )

    def alpr_not_loaded(self):
        """
        Logged when an ALPR request arrives but ALPR is not enabled. This is
        common and benign (Blue Iris is configured to send plate requests but
        the user hasn't turned ALPR on), so it is rate-limited to at most once
        per 60 seconds to avoid flooding the console.
        """
        now = time.time()
        last = getattr(self, "_last_alpr_notice", 0)
        if now - last < 60:
            return
        self._last_alpr_notice = now
        self._append_and_broadcast(
            _make_entry(LogLevel.SYSTEM,
                "ALPR request received but ALPR is off — ignoring. "
                "Enable it in ALPR Settings, or remove the Plate Recognizer config in Blue Iris.")
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
