"""
plate_store.py — Persistent plate-read storage for Objectif.AI ALPR.

Responsibilities:
  - Lazily create a /plates directory and SQLite database on first use.
  - Save two files per plate read:
      YYYYMMDD_HHMMSS_mmm_{PLATE}_crop.jpg  — the 120px plate thumbnail
      YYYYMMDD_HHMMSS_mmm_{PLATE}_full.jpg  — the source image at reduced quality
  - Persist the last N reads to SQLite so history survives server restarts.
  - Serve plate reads back (newest-first) with file paths for the modal.
  - Enforce a configurable day-based retention policy (default 30 days).
  - Emit zero overhead if ALPR is not enabled: no directory, no DB, no background
    thread. Everything is created lazily on the first actual plate read.

Design notes:
  - SQLite is Python stdlib — no extra dependency.
  - All disk writes happen in a ThreadPoolExecutor so they never block the
    async request handler.
  - The store is a singleton accessed via module-level functions.
  - `PLATES_DIR` defaults to <app_dir>/plates. Configurable via
    config.yaml `alpr.plates_dir`.
  - File writes are best-effort: a write failure never fails the ALPR response.
"""

import base64
import logging
import os
import re
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level state — nothing is allocated until _ensure_ready() is called.
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_ready = False
_plates_dir: Optional[Path] = None
_db_path: Optional[Path] = None
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="plate_store")

# Maximum records to keep in SQLite (beyond this the oldest are pruned).
MAX_RECORDS = 10_000
# Default retention in days (0 = keep forever).
DEFAULT_RETENTION_DAYS = 30


def _ensure_ready(config: dict) -> bool:
    """
    Lazily initialise the plates directory and SQLite DB.
    Called on first plate write — a no-op on subsequent calls.
    Returns True if ready, False if initialisation failed.
    """
    global _ready, _plates_dir, _db_path

    if _ready:
        return True

    with _lock:
        if _ready:
            return True
        try:
            alpr_cfg = config.get("alpr", {})
            # Resolve plates directory
            app_dir = Path(__file__).parent
            plates_dir_cfg = alpr_cfg.get("plates_dir", "")
            if plates_dir_cfg:
                pd = Path(plates_dir_cfg)
            else:
                pd = app_dir / "plates"
            pd.mkdir(parents=True, exist_ok=True)
            _plates_dir = pd

            # Initialise SQLite
            _db_path = pd / "plates.db"
            _init_db(_db_path)

            _ready = True
            logger.info(f"Plate store ready at {pd}")
            return True
        except Exception as e:
            logger.error(f"Plate store init failed: {e}")
            return False


def _init_db(db_path: Path):
    """Create the plates table and indexes if they don't exist."""
    con = sqlite3.connect(str(db_path))
    try:
        con.execute("""
            CREATE TABLE IF NOT EXISTS plate_reads (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                plate       TEXT    NOT NULL,
                confidence  REAL    NOT NULL,
                timestamp   REAL    NOT NULL,   -- Unix epoch (float)
                crop_file   TEXT,               -- filename only, relative to plates dir
                full_file   TEXT,               -- filename only, relative to plates dir
                source      TEXT                -- endpoint: 'alpr' | 'detection'
            )
        """)
        con.execute("CREATE INDEX IF NOT EXISTS idx_ts ON plate_reads (timestamp DESC)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_plate ON plate_reads (plate)")
        con.commit()
    finally:
        con.close()


# ---------------------------------------------------------------------------
# File naming
# ---------------------------------------------------------------------------

def _make_stem(plate: str, ts: float) -> str:
    """
    Build a filesystem-safe stem for a plate read at timestamp ts.
    Format: YYYYMMDD_HHMMSS_mmm_{PLATE}
    """
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    ms = int(dt.microsecond / 1000)
    safe_plate = re.sub(r"[^A-Z0-9]", "_", plate.upper())
    return f"{dt.strftime('%Y%m%d_%H%M%S')}_{ms:03d}_{safe_plate}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_plate_read(
    plate: str,
    confidence: float,
    thumbnail_b64: str,
    full_image_bytes: bytes,
    config: dict,
    source: str = "alpr",
):
    """
    Persist a plate read asynchronously. Called from the ALPR endpoint.

    thumbnail_b64 — base64 data-URI (or raw base64) of the 120px crop.
                    May be empty if fast-alpr couldn't crop.
    full_image_bytes — the raw JPEG bytes received from Blue Iris, re-encoded
                       at reduced quality before saving.
    config — the current app config dict (passed in so we don't import
             get_config here and create a circular dependency).
    """
    ts = time.time()
    _executor.submit(
        _write_plate_read,
        plate, confidence, thumbnail_b64, full_image_bytes, config, source, ts,
    )


def _write_plate_read(
    plate: str,
    confidence: float,
    thumbnail_b64: str,
    full_image_bytes: bytes,
    config: dict,
    source: str,
    ts: float,
):
    """Runs in the thread pool — all disk and DB I/O here."""
    if not _ensure_ready(config):
        return

    stem = _make_stem(plate, ts)
    crop_file: Optional[str] = None
    full_file: Optional[str] = None

    # --- Save crop ---
    try:
        if thumbnail_b64:
            # Strip data-URI prefix if present
            b64_data = thumbnail_b64.split(",", 1)[-1] if "," in thumbnail_b64 else thumbnail_b64
            crop_bytes = base64.b64decode(b64_data)
            crop_fname = f"{stem}_crop.jpg"
            crop_path = _plates_dir / crop_fname
            with open(crop_path, "wb") as f:
                f.write(crop_bytes)
            crop_file = crop_fname
    except Exception as e:
        logger.warning(f"Plate store: crop save failed for {plate}: {e}")

    # --- Save full image (re-encoded at reduced quality) ---
    try:
        if full_image_bytes:
            import cv2, numpy as np  # only imported here — lazy, ALPR-only path

            nparr = np.frombuffer(full_image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                full_fname = f"{stem}_full.jpg"
                full_path = _plates_dir / full_fname
                # Encode at 55% JPEG quality — typically 80–150 KB from a 500–700 KB source
                cv2.imwrite(str(full_path), img, [cv2.IMWRITE_JPEG_QUALITY, 55])
                full_file = full_fname
    except Exception as e:
        logger.warning(f"Plate store: full image save failed for {plate}: {e}")

    # --- Write to SQLite ---
    try:
        con = sqlite3.connect(str(_db_path))
        try:
            con.execute(
                "INSERT INTO plate_reads (plate, confidence, timestamp, crop_file, full_file, source) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (plate.upper(), round(confidence, 4), ts, crop_file, full_file, source),
            )
            con.commit()
        finally:
            con.close()
    except Exception as e:
        logger.warning(f"Plate store: DB write failed for {plate}: {e}")
        return

    # --- Prune old records ---
    _prune(config)


def _prune(config: dict):
    """
    Enforce retention policy. Deletes associated files for pruned records.
    Runs after every write — cheap because it uses indexed queries.
    """
    if not _ready or _db_path is None:
        return

    alpr_cfg = config.get("alpr", {})
    retention_days = int(alpr_cfg.get("plates_retention_days", DEFAULT_RETENTION_DAYS))

    try:
        con = sqlite3.connect(str(_db_path))
        try:
            to_delete = []

            # Day-based pruning
            if retention_days > 0:
                cutoff = time.time() - (retention_days * 86400)
                rows = con.execute(
                    "SELECT id, crop_file, full_file FROM plate_reads WHERE timestamp < ?",
                    (cutoff,),
                ).fetchall()
                to_delete.extend(rows)

            # Hard cap — keep newest MAX_RECORDS, delete the rest
            count = con.execute("SELECT COUNT(*) FROM plate_reads").fetchone()[0]
            if count > MAX_RECORDS:
                excess = count - MAX_RECORDS
                rows = con.execute(
                    "SELECT id, crop_file, full_file FROM plate_reads "
                    "ORDER BY timestamp ASC LIMIT ?",
                    (excess,),
                ).fetchall()
                to_delete.extend(rows)

            if not to_delete:
                return

            ids = list({r[0] for r in to_delete})
            # Delete files first
            for _, crop_f, full_f in to_delete:
                for fname in (crop_f, full_f):
                    if fname:
                        try:
                            (_plates_dir / fname).unlink(missing_ok=True)
                        except Exception:
                            pass
            # Delete DB rows
            con.execute(
                f"DELETE FROM plate_reads WHERE id IN ({','.join('?' * len(ids))})",
                ids,
            )
            con.commit()
            logger.debug(f"Plate store: pruned {len(ids)} records")
        finally:
            con.close()
    except Exception as e:
        logger.warning(f"Plate store: prune failed: {e}")


def get_history(limit: int = 1000) -> list:
    """
    Return the last `limit` plate reads from SQLite, newest-first.
    Each entry: {id, plate, confidence, timestamp, crop_file, full_file}.
    Returns [] if the store is not initialised (ALPR never ran this session).
    """
    if not _ready or _db_path is None:
        return []
    try:
        con = sqlite3.connect(str(_db_path))
        try:
            rows = con.execute(
                "SELECT id, plate, confidence, timestamp, crop_file, full_file "
                "FROM plate_reads ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [
                {
                    "id": r[0],
                    "plate": r[1],
                    "confidence": r[2],
                    "timestamp": r[3],
                    "crop_file": r[4] or "",
                    "full_file": r[5] or "",
                }
                for r in rows
            ]
        finally:
            con.close()
    except Exception as e:
        logger.warning(f"Plate store: get_history failed: {e}")
        return []


def get_plates_dir() -> Optional[Path]:
    """Return the plates directory path, or None if not yet initialised."""
    return _plates_dir
