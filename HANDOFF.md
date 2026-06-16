# Objectif.AI — Project Handoff

Read this before making any changes. It explains architecture decisions, recurring
bugs, known quirks, and why things are done the way they are.

**Current version:** v0.8.5.1
**Working directory convention:** all source in one flat folder + `static/` subfolder.

---

## What This Is

A Python/FastAPI server that replaces CodeProject.AI for BlueIris NVR. BlueIris
POSTs camera images to `/v1/vision/detection` and expects JSON bounding box
detections back. As of v0.7.9 it also serves license-plate recognition on
`/v1/vision/alpr`. Everything else (dashboard, model management, GPU setup) is
layered on top of those core contracts.

**The BlueIris API contract is sacred.** `/v1/vision/detection` must always accept
multipart/form-data and return the exact CodeProject.AI JSON format. The same
applies to `/v1/vision/alpr`. BlueIris users point at this server with zero
reconfiguration. Never break these endpoints.

---

## File Structure

```
main.py            FastAPI app, all HTTP + WebSocket endpoints (incl. /v1/vision/alpr + /api/alpr/*)
auth.py            API key generation and verification
config.py          YAML config read/write, in-memory singleton (incl. alpr.* block)
detector.py        Object-detection engine + separate ALPREngine (fast-alpr) singleton
downloader.py      Model file download manager with WebSocket progress streaming
hardware.py        Hardware detection (CUDA, OpenVINO, ROCm, DirectML, CC detection)
console_buffer.py  Circular in-memory log buffer, WebSocket broadcast
model_registry.py  Model catalog — names, URLs, metadata, engine types
dependencies.py    Package checker for the Dependencies tab (incl. fast-alpr, onnxruntime-directml)
_write_task_xml.py Helper called by setup-service.bat — generates start-silent.vbs
                   (with full python.exe path) and writes Task Scheduler XML UTF-16.
_stop_server.py    Helper called by remove-service.bat — finds and kills the running
                   python.exe by matching the app directory in its command line.
                   Uses wmic with taskkill, falls back to psutil.
static/index.html  Entire dashboard frontend — vanilla JS, no framework, no build
```

**Config files (gitignored, never commit):**
- `config.yaml` — user settings + API key
- `logs/startup.log` — append-forever startup events
- `logs/server.log` — rotating 5×1MB server log
- `models/` — downloaded weights + cached ONNX exports
- `.deps_installed` — flag file created by start.bat

---

## Authentication — Critical Details

**How it works:** Random 64-char hex key generated on first run, saved to
`config.yaml`. `serve_dashboard()` reads `index.html`, substitutes the literal
string `__OBJECTIF_API_KEY__` with the real key, and sends it to the browser.
All API calls use the `api()` JS wrapper which sends `X-Api-Key` header.

**THE SPLIT STRING BUG — DO NOT FIX:**
The JS fallback check uses:
```javascript
const placeholder = '__OBJECTIF_' + 'API_KEY__';
if (!state.apiKey || state.apiKey === placeholder)
```
This split is intentional and load-bearing. If you write `'__OBJECTIF_API_KEY__'`
as a whole string anywhere in `index.html`, the server's `.replace()` substitutes
it too — the check then compares the real key against itself and always passes,
so the error page always shows. This bug was hit twice during development.
Never "simplify" the split.

**WebSocket auth uses single-use tickets (v0.7.8)** — browsers cannot set
headers on WebSocket connections. Earlier versions passed the API key in the
`?key=` query string, which exposed it to uvicorn access logs and browser
history. Now the dashboard POSTs `/api/auth/ws-ticket` (authenticated with
the API key) and connects with `?ticket=`. Tickets are 30-second single-use.
The `?key=` path still works as a fallback for any direct WS client.

**`/v1/vision/detection` is intentionally unauthenticated** — BlueIris cannot
send auth headers. Do not add auth to this endpoint ever. Flood protection is
handled by a per-IP rate limiter (`_rate_limit_check`, default 100 req/s).

**Host-header allowlist (v0.7.8)** — `HostHeaderMiddleware` rejects requests
whose Host header isn't `localhost` or an RFC1918 / loopback / link-local IP.
This blocks DNS-rebinding attacks where a public domain resolves to a LAN IP.
If you ever need to host behind a custom hostname, extend `_host_is_allowed()`.

**Config update validation (v0.7.8)** — `/api/config` now calls
`validate_config_value()` from `config.py` per path. Type/range mismatches
are rejected before reaching disk, so a holder of the API key can't crash
the server by setting e.g. `server.port` to a string.

---

## Model Download Flow (v0.7.7)

All models now follow the same three-step flow: **Download Model → Load Model → Active.**
This was not always the case — torchhub/torchvision previously used a combined
"Load & Download" button that silently downloaded on first load. This was confusing
because switching to another model and back made the button revert to "Load & Download"
even though the model was already cached, making users think something was broken.

**Current button states (all engines):**
- Not downloaded → "Download Model" button (+ size note if >100 MB)
- Downloading → progress bar (torchvision shows indeterminate pulse, no Cancel; all others show real % with Cancel)
- Downloaded → "Load Model" + "Delete"
- Active → "Active" (disabled) + "Delete"

**Download routing in `downloader.py`:**
- `ultralytics` / `onnx` → `_download_thread()` — direct URL download with real % progress
- `torchvision` → `_download_torchvision_thread()` — calls torchvision loader with pulse progress
- `torchhub` — engine still exists in code but no models currently use it (see YOLOv5 note below)

---

## YOLOv5 Engine Change (v0.7.7)

**YOLOv5 models were switched from `engine="torchhub"` to `engine="ultralytics"` with
direct GitHub release download URLs.**

Previously they used `torch.hub.load("ultralytics/yolov5", ...)` which:
1. Cloned the entire YOLOv5 Python repo into `models/ultralytics_yolov5_master/` (confusing users)
2. Saved weights to `TORCH_HOME/hub/checkpoints/yolov5n.pt` — **not** in `MODELS_DIR` —
   causing `is_downloaded()` to always return `False` and the "Model not downloaded yet"
   error on every "Load Model" click

**The fix:** Direct download from GitHub v7.0 release assets:
`https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt`

The Ultralytics package (`from ultralytics import YOLO`) loads original YOLOv5 v7.0
`.pt` files natively — same code path as YOLO v8/v9/v10/v11. File lands at
`MODELS_DIR/yolov5n.pt` and the standard flat-file check in `is_downloaded()` works.

The `_download_torchhub_thread()` and torchhub branches in `is_downloaded()` /
`delete_model()` are retained in `downloader.py` in case of future use, but no
models in the registry use them.

Also fixed in v0.7.7: `/api/models` endpoint was computing `m["downloaded"]` as a
raw `(MODELS_DIR / filename).exists()` check, bypassing `is_downloaded()` entirely.
Now calls `download_manager.is_downloaded(m["id"])` so all engine types are handled
consistently and the UI state is always accurate.

---

## Torchvision File Location — Known Quirk

PyTorch saves torchvision weights to a checkpoints subfolder under `TORCH_HOME`,
but the exact subfolder path varies by PyTorch version:
- Some versions: `models/checkpoints/`
- Other versions: `models/hub/checkpoints/`

Filenames also include a hash suffix (e.g. `fasterrcnn_resnet50_fpn_coco-258fb6c6.pth`).

**The fix:** `is_downloaded()` in `downloader.py` uses `MODELS_DIR.rglob("*.pth")`
to scan the entire models tree and matches by filename prefix (stripping `_coco`
and hash suffix). Same approach used in `delete_model()`. Do not hardcode
subfolder paths for torchvision — they will break on a different PyTorch version.

---

## torchhub (YOLOv5) File Location

**This section is superseded by the YOLOv5 Engine Change section above.**
YOLOv5 models no longer use torch.hub. The torchhub engine code is retained in
`downloader.py` but unused. If you ever add a new torchhub model, note that
torch.hub saves weights to `TORCH_HOME/hub/checkpoints/` — not in `MODELS_DIR` —
so `is_downloaded()` will need updating.

---

## GPU Support Architecture

**Normal NVIDIA (CC ≥ 7.5):** Ultralytics runs directly on PyTorch CUDA.

**Legacy NVIDIA (CC < 7.5 — GTX 10-series, Tesla P4, GTX 1060, Pascal/Maxwell):**
Modern PyTorch dropped support. Our path:
1. `_is_legacy_gpu()` detects CC via PyTorch props or nvidia-smi (cached after first call)
2. On first YOLO model load, `_load_ultralytics_legacy()` exports `.pt` → `.legacy.onnx`
   via Ultralytics export (30–60 seconds, cached in `models/`)
3. Subsequent loads use ONNX Runtime CUDA execution provider
4. Header shows amber "LEGACY" badge when active
5. Console shows progress: "Exporting to ONNX... this may take up to 60 seconds"

**YOLOv5 and torchvision models cannot use the legacy path** — they need PyTorch
CUDA. They fall back to CPU with a console warning.

**Intel:** OpenVINO + onnxruntime-openvino
**AMD:** ROCm — experimental, not tested by dev team
**DirectML (v0.7.9):** see below

---

## v0.7.9 — DirectML, ALPR, and ONNX routing fix

### DirectML backend
`onnxruntime-directml` exposes `DmlExecutionProvider`, which runs ONNX models on
any DirectX 12 GPU (NVIDIA, AMD, Intel) on Windows. Wired through:
- `detector.py`: `_onnx_providers()` adds `DmlExecutionProvider`; `_auto_device()`
  returns `"directml"` only for `engine == "onnx"` models (it has no effect on
  PyTorch `.pt` / torchvision paths, so `_device_to_ultralytics()` maps it to CPU);
- `config.py`: `directml` added to the `detection.backend` validator;
- `main.py`: `/api/install/directml` endpoint and `directml` allowed in
  `/api/backend/switch`; install-status reports `onnxruntime_directml`;
- `hardware.py`: `_detect_directml()` + DirectML in the backend list and dict;
- `dependencies.py`: `onnxruntime_directml` catalog entry + provider check;
- `static/index.html`: DirectML hardware card + `installDirectml()`.

**Critical gotcha — ONNX Runtime builds are mutually exclusive.** `onnxruntime`,
`onnxruntime-gpu`, `onnxruntime-openvino`, and `onnxruntime-directml` all install
as the same import name (`onnxruntime`) and cannot coexist. The DirectML install
chain therefore runs an **uninstall step first**. This is why `_run_install_chain()`
was extended to recognise a `["--uninstall", ...]` sentinel as the first list
element (runs `pip uninstall -y` and tolerates a non-zero return code). Don't
"clean that up" — it's load-bearing.

**Critical gotcha #2 — native runtime swaps REQUIRE a server restart (v0.7.9.1).**
First DirectML builds appeared to "install" in ~2s and report success, but the
Hardware badge never flipped to ready. Root cause: a running Python process has
already imported `onnxruntime` (the dependency checker imports it), and you cannot
hot-swap a loaded native module's `.pyd`/`.dll` in a live process on Windows. pip's
metadata removal/install can succeed while the in-memory module — and therefore
`ort.get_available_providers()` — stays unchanged. Re-running just repeats the
no-op. Fixes applied: (1) the DirectML install now uses
`--force-reinstall --no-cache-dir` so pip can't silently treat it as "already
satisfied"; (2) `_run_install_chain(..., restart_required=True)` emits an explicit
RESTART REQUIRED warning and the `done` broadcast carries `restart_required`, which
the dashboard turns into a restart prompt; (3) the chain now watches pip output for
"Downloading/Installing collected/Successfully installed" and warns if a non-uninstall
step installed nothing. The same restart requirement applies to ANY future native
provider swap (e.g. switching back to onnxruntime-gpu) — reuse `restart_required`.

### ALPR (license plate recognition)
Uses the MIT-licensed `fast-alpr` package (plate detector from `open-image-models`
+ OCR from `fast-plate-ocr`). Architecture:
- **Separate engine.** `ALPREngine` in `detector.py` is its own singleton
  (`alpr_engine`), independent of the object-detection `InferenceEngine`, so both
  can be loaded at once. It has its own lock.
- **Self-managing weights.** `fast-alpr` downloads ONNX weights from Hugging Face
  on first `ALPR(...)` construction and caches them in the HF cache — same
  "lazy download" posture as the torchhub/torchvision paths. We deliberately do
  NOT host these or route them through `downloader.py` / `model_registry.py`.
  If HF is down, first load fails; cached weights keep working.
- **Separate endpoint.** `/v1/vision/alpr` (CPAI-ALPR-compatible: predictions with
  `label`/`plate`, confidence, bbox). Unauthenticated + rate-limited like detection.
  **As of v0.8.5.1, Blue Iris 5.9.9 does NOT use this endpoint** — it uses `POST /`
  via the Plate Recognizer SDK protocol. See v0.8.5.1 section.
- **Config.** `alpr.*` block: `active` (auto-load on startup), `min_confidence`,
  `detector_model`, `ocr_model`. Only `alpr.min_confidence` is in
  `API_UPDATABLE_PATHS`; the model choices are set via `/api/alpr/load`, not the
  generic config endpoint.
- **Dashboard.** Controls live in Settings → License Plate Recognition (not a new
  tab). `/api/alpr/status|load|unload` drive it.

**VERIFIED in v0.8.5.1.** `ALPREngine.recognize()` correctly reads fast-alpr's
result objects. Note: `ocr.confidence` is a list of per-character floats — see
v0.8.5.1 section for the fix.

### ONNX output routing fix
`InferenceEngine.detect()` previously routed every `onnx` engine to
`_detect_onnx_yolo()`, leaving `_detect_onnx()` (the MobileNet SSD / EfficientDet
parser) as dead code. Detection now dispatches `engine == "onnx"` models whose
`family` is `MobileNet` or `EfficientDet` to `_detect_onnx()`, and everything else
(legacy-exported YOLO, generic YOLO ONNX) to `_detect_onnx_yolo()`. Note those
ONNX families still have empty `download_url` (manual install only), so this path
is only exercised if a user manually drops an `.onnx` in `models/`.

---

## v0.8.5.1 — Blue Iris 5.9.9 ALPR, plate suppression, console observability

This release has two thrusts: (1) making ALPR actually work with Blue Iris 5.9.9
and adding plate management, and (2) a "show what's going on under the hood" pass
on the console and header. The ALPR/suppression work is documented in the
subsections below; the observability work is documented under
**"Console observability"** further down.

### Blue Iris 5.9.9 ALPR Protocol Fix

Blue Iris 5.9.9 changed how it sends ALPR requests. It no longer uses a separate
AI server entry with path `/v1/vision/alpr`. Instead, when "Plate Recognizer®" is
selected under Settings → AI → License plates (ALPR), it sends requests using the
**Plate Recognizer SDK protocol**:

- **Endpoint:** `POST /` (bare root) — discovered by live traffic capture; documented
  nowhere in Blue Iris's own docs
- **Image field name:** `upload` (not `image` as used by `/v1/vision/detection`)
- **Response format:** `{"results": [{plate, score, box: {xmin,ymin,xmax,ymax}, vehicle, region}], "processing_time": float}`

Two endpoints added in `main.py`:
- `POST /v1/plate-reader/` — canonical Plate Recognizer SDK path
- `POST /` — bare root, which is what Blue Iris 5.9.9 actually sends to

`POST /` delegates to `plate_reader_compat()`. The `POST /v1/vision/alpr` CPAI-style
endpoint is still present for backward compatibility.

**Blue Iris config (for users):**
- License plates (ALPR) dropdown → Plate Recognizer®
- Configure → On-Premise SDK Port → `<ip>:32168`
- Region/s field → blank or country code (e.g. `us`). NOT a path.
- Make/Model/Color analysis → unchecked (unsupported)

### fast-alpr ONNX Provider Conflict Fix

`fast-alpr` uses `open-image-models` which calls `ort.get_available_providers()`
to build its provider list before creating an `InferenceSession`. On machines with
`onnxruntime-gpu` installed (normal for Objectif.AI CUDA users), the GPU providers
fail to load inside `fast-alpr`'s ONNX session due to missing CUDA DLL dependencies
(the ort-gpu build expects specific cuBLAS versions). The result: `fast-alpr`'s
detector returns no detections — silently, with no error.

Fix in `detector.py` `ALPREngine.load()`: monkeypatch both `ort.get_available_providers`
and `ort.InferenceSession` to return/use only `["CPUExecutionProvider"]` during
`fast-alpr`'s `ALPR(...)` construction, then restore both immediately after. This
is safe because:
1. `fast-alpr` stores the provider list and session at init time — it doesn't
   re-call `get_available_providers()` on each inference
2. The main detection engine (`InferenceEngine`) creates its ONNX sessions elsewhere
   and is not affected
3. The patch is scoped to the `try/finally` block — no leakage

**Don't remove the monkeypatch** — it's load-bearing on every CUDA machine.

### OCR Confidence Fix

`fast-alpr` returns `ocr.confidence` as a **list of per-character floats**, not a
single float (e.g. `[0.085, 0.971, 0.785, ...]`). The original code did
`float(ocr.confidence)` which raised `TypeError: float() argument must be a string
or a real number, not 'list'`, causing all ALPR results to be silently dropped via
the `except` clause.

Fix in `detector.py` `ALPREngine.recognize()`:
```python
raw_ocr_conf = getattr(ocr, "confidence", 1.0)
if isinstance(raw_ocr_conf, (list, tuple)) and len(raw_ocr_conf) > 0:
    ocr_conf = sum(raw_ocr_conf) / len(raw_ocr_conf)
else:
    ocr_conf = float(raw_ocr_conf or 1.0)
```
Average of per-character confidences is used as the overall OCR confidence.

### Plate Suppression System

New `alpr.suppression` config block (persisted to `config.yaml`):
```yaml
alpr:
  suppression:
    global_cooldown_seconds: 0   # 0=always report, -1=suppress all, N=cooldown
    plates:
      ABC123: {cooldown_seconds: -1, label: "My Car"}
      XYZ789: {cooldown_seconds: 3600, label: "Neighbor"}
    last_seen:
      ABC123: 1718467200.0       # Unix timestamp of last report
```

`_plate_is_suppressed(plate)` is called in both ALPR endpoints before adding a
plate to the response. `_record_plate_seen(plate)` updates `last_seen` after a
plate passes the suppression check.

New API endpoints (all authenticated):
- `GET /api/alpr/suppression` — returns plates dict + global_cooldown_seconds
- `POST /api/alpr/suppression/global` — set global default cooldown
- `POST /api/alpr/suppression/plates` — add/update a plate entry
- `DELETE /api/alpr/suppression/plates/{plate}` — remove a plate

### Plate History

`console_buffer.py` extended:
- `_plate_reads` deque now stores `(timestamp, plate_text, confidence)` tuples
  (was `(timestamp, plate_text)`)
- New `get_plate_history(limit)` method returns newest-first list of
  `{plate, timestamp, confidence}` dicts

New API endpoint: `GET /api/alpr/history?limit=1000` — returns the last N plates
reported to Blue Iris this session (in-memory, resets on restart).

Dashboard: clicking the ALPR pill in the header opens a modal showing the plate
history table (plate, time, confidence). Closes on ✕, overlay click, or Escape.

### Dashboard — New ALPR Settings Tab

All ALPR configuration moved from Settings to a dedicated "ALPR Settings" tab:
- Engine enable/disable, model selectors, confidence slider, reload button
- Global cooldown slider (−1 to 86400 seconds)
- Known plates table with add form and per-plate cooldown select
- Step-by-step Blue Iris 5.9.9 setup guide (with the correct Plate Recognizer SDK
  instructions — the old "add a second AI server" instructions were wrong for 5.9.9).
  Ordered last so all the live settings sit at the top of the tab.

### Console log-level filter

The console toolbar has a **Filter ▾** dropdown (top-right) to show/hide each log
level independently. State lives in `state.hiddenLogLevels` (a Set), is applied in
`appendLogEntry()` (early return if the level is hidden), and persists to
`config.yaml` under `console.hidden_levels` (added to `API_UPDATABLE_PATHS` and the
validators in `config.py`). Loaded on boot in `loadConfig()`.

### ALPR "not loaded" no longer spams as a WARNING

Previously the ALPR endpoints called `console.no_model()` (a WARNING) when ALPR
wasn't enabled. Since most users don't run ALPR but Blue Iris may still be configured
to send Plate Recognizer requests, this flooded the console. Now they call
`console.alpr_not_loaded()` — a SYSTEM-level message, rate-limited to once per 60s,
explaining that ALPR is off and how to enable it or stop Blue Iris sending. The
detection endpoint still uses the real `no_model()` WARNING.

### Plate Storage and History (persistent, ALPR-gated)

**New module: `plate_store.py`** — a self-contained persistent store for plate reads.
Keeps main.py clean. Key design decisions:

**Lazy initialisation.** `_ensure_ready()` is called on the first actual plate write,
not at startup. If ALPR is never enabled: no `plates/` directory, no SQLite DB, no
background thread, zero overhead. This was a hard requirement — users running without
ALPR should pay nothing.

**Two files per read:**
- `{stem}_crop.jpg` — the 120px base64 thumbnail decoded back to bytes (already
  generated by `recognize()`, just saved)
- `{stem}_full.jpg` — source `image_bytes` decoded by OpenCV and re-encoded at
  JPEG quality 55 (typically 80–150 KB from a 500–700 KB source). cv2/numpy are
  imported lazily inside `_write_plate_read` so they don't load for non-ALPR users.

**File naming:** `YYYYMMDD_HHMMSS_mmm_{PLATE}_{crop|full}.jpg` in UTC. Safe for all
filesystems — non-alphanumeric plate chars replaced with `_`.

**SQLite (stdlib, no extra dep).** Schema: `plate_reads(id, plate, confidence,
timestamp, crop_file, full_file, source)`. Indexes on `timestamp DESC` and `plate`.
Created by `_init_db()` if not present. Connection per operation (not pooled) — plate
writes are low-frequency so this is simpler and avoids threading issues.

**ThreadPoolExecutor (max 2 workers).** All disk + DB I/O runs off the async thread.
`save_plate_read()` is fire-and-forget from the endpoint — a write failure never
fails the ALPR response to Blue Iris.

**Retention.** `_prune()` runs after every write. Two pruning passes:
1. Day-based: deletes rows older than `alpr.plates_retention_days` days (default 30,
   0 = keep forever, max 3650)
2. Hard cap: keeps newest `MAX_RECORDS` (10,000), deletes oldest beyond that
Files are unlinked first, then DB rows deleted. `missing_ok=True` on unlink.

**History endpoint** (`GET /api/alpr/history`) now reads from SQLite first, falls
back to in-memory `console.get_plate_history()` if the store hasn't been initialised
(ALPR never ran or no plates read yet). Response includes `source: "db"|"memory"`.

**Image serving** (`GET /api/alpr/image/{filename}`): authenticated (API key),
strict filename validation (no path separators, must match `[\w\-]+\.jpe?g`),
returns `FileResponse`. Frontend includes `?key=` in the src URL since `<img>` tags
can't set headers.

**Frontend changes:**
- ALPR Settings tab: retention slider (0–365 days), info note about lazy creation
- History modal: new "View Full" column, expandable row showing crop + full image
  side by side. `toggleFullImage()` shows/hides the expand row and relabels button.
- Modal header shows `(persistent)` vs `(this session)` based on `source` field.
- `fmtRetention()` formats slider value to human-readable (0→"Keep forever",
  1→"1 day", <30→"N days", <365→"N weeks", 365→"1 year")

**Config additions:**
- `alpr.plates_retention_days` (int, 0–3650) in `API_UPDATABLE_PATHS` + validator

**`.gitignore` note:** add `plates/` to `.gitignore` before the next commit.

---

## Console observability (v0.8.5.1)

A batch of "show what's happening under the hood" features. They share two plumbing
clusters: **per-inference enrichment** (tag, backend badge, timing, thumbnail) all
flow through the console entry + WS payload, and **header telemetry** (system stats)
rides the existing inference-stats WebSocket.

### LogEntry new fields

`console_buffer.py` `LogEntry` gained four optional fields: `model`, `backend`,
`timing` (dict), `source` (`"detection"` | `"alpr"` | `"onnx"`). `detection_result()`
and `alpr_result()` accept and pass them. They serialize through `entry_to_dict()`
automatically (it's `asdict`), so the WS payload carries them with no extra wiring.

### Model/endpoint tag + backend badge (per line)

- The endpoint sets `source` and `model`. For detection, `source` is `"onnx"` when
  `engine._engine_type in ("onnx","onnx_legacy")` else `"detection"`; `model` is the
  model display name. For ALPR, `source="alpr"`, `model="fast-alpr"`.
- `backend` comes from a new `InferenceEngine.active_backend` property (and the same
  on `ALPREngine`), which normalizes `self._device` to `CUDA`/`CPU`/`DML`/`OpenVINO`/
  `ROCm`. **It reflects the resolved device, not the requested one** — that's the whole
  point: a silent CPU fallback shows truthfully. ALPR is always CPU by design (the
  v0.8.5.1 fast-alpr CPU monkeypatch), so its badge reads CPU correctly.
- Frontend renders `.log-tag` and `.log-backend` pills. The backend badge can be
  hidden via `state.hideBackendBadge`, toggled in the filter menu and persisted to
  `console.hide_backend_badge`.

### Timing split

- `DetectionResult` gained a `timing` dict. Ultralytics path uses the native
  `results[0].speed` dict (preprocess/inference/postprocess). The ONNX YOLO path wraps
  `perf_counter` manually around the three phases. When a split exists, `inference_ms`
  is set to the **sum** so total and split agree.
- ALPR `recognize()` now returns a **3-tuple** `(results, inference_ms, timing)` where
  timing is `{decode, inference}` (fast-alpr doesn't expose a detector/OCR split, so
  decode-vs-predict is the honest granularity). **Both callers in `main.py` were
  updated** — if you add a third caller, unpack three values.
- Frontend shows the split as an indented detail line, expanded mode only.

### Plate crop thumbnails

- `ALPRResult` gained a `thumbnail` field. `recognize()` crops the plate bbox (with
  small padding, clamped to image bounds), resizes to ~120px wide, encodes JPEG q70,
  and base64-data-URIs it. Best-effort — a thumbnail failure never fails the read.
- `console_buffer` plate-history tuples went from 3-element to **4-element**
  `(timestamp, text, confidence, thumbnail)`. `_prune_plate_thumbnails()` strips the
  image off all but the most recent 100 thumbed reads to bound memory (text/conf/ts
  are kept). `get_plate_history()` returns the thumbnail; `_prune_plate_reads()` only
  indexes `[0]` so it was unaffected.
- The endpoints build a separate `plate_log` list (with thumbnails) for the console
  while keeping `predictions`/`pr_results` clean for Blue Iris.
- Frontend history modal gained a "Crop" column rendering the `<img>`.

### Header system stats

- `get_system_stats()` in `main.py` returns CPU%, RAM (% + used/total GB), app RSS
  (MB), and uptime via `psutil`. Returns `{available: False}` if psutil import fails.
- `_system_stats_loop()` is an asyncio task started in `lifespan` (cancelled on
  shutdown). It primes `cpu_percent()` once (first call always reads 0), then polls
  every 3s and broadcasts a `system_stats` WS message via `_broadcast_system_stats()`.
- `GET /api/system-stats` serves the same snapshot for the initial page load (so the
  strip populates immediately rather than waiting up to 3s).
- Reuses the existing module-level `_start_time` for uptime (don't add a duplicate).
- Frontend: header `#sysstats-pill` (hidden until first data), `handleSystemStats()`
  WS handler, `fetchSystemStats()` one-shot on boot. CPU/RAM go amber ≥80%, red ≥90%.
- `psutil>=5.9.0` is in `requirements.txt` and surfaced on the Dependencies tab.

### Config additions

`console.hidden_levels` (list), `console.hide_backend_badge` (bool) added to
`API_UPDATABLE_PATHS` and `_VALIDATORS` in `config.py`. Suppression paths
(`alpr.suppression.*`) are written via internal `update_config()` which does NOT
enforce the allowlist, so they don't need entries there.

### Future hook

Disk monitoring was deliberately deferred but the system-stats dict is structured to
extend — adding a `disk_percent` key + a header element is a small addition. Worth
doing once plate-snip storage (if added) can fill a disk.

---

## v0.7.9.2 — /api/dependencies crash fix

The Dependencies tab 500'd on some machines. Root cause: `_get_version()` did
`importlib.import_module(import_name)` to read a package's `__version__`, which
*executes that module's whole import chain*. On an environment with a broken
transitive dep (observed: a numpy/scipy/seaborn mismatch where seaborn's import
raised `AttributeError: ... no attribute '_blas_supports_fpe'`), that exception
propagated out of the endpoint — only `ImportError` was caught, not arbitrary
import-time errors.

Fixes in `dependencies.py`:
- `_get_version()` tries `importlib.metadata.version()` FIRST (reads metadata, no
  module execution), falling back to import only if metadata can't resolve the
  dist. The import fallback now catches ALL exceptions: genuinely missing ->
  `None`; present but broken import chain -> `"installed (version unavailable)"`
  (still counts as installed, since it IS on disk).
- `check_all_dependencies()` wraps each entry via `_check_single_dependency()`;
  one failing check degrades to status `"unknown"` / "Check failed" instead of
  breaking the whole page.

Lesson: never import a module just to inspect it — prefer metadata or
`find_spec`. seaborn/matplotlib are optional catalog entries (plotting features)
and are exactly the heavy/fragile imports that must not run during a status poll.

---

## Known Recurring Bugs

### System tray icon — disabled

pystray caused intractable problems across launch methods:
- `start.bat` (python.exe): worked inconsistently
- Task Scheduler via VBS: message pump issues, wrong Python env, restart broken

Tray.py is retained as a stub of no-op functions so main.py doesn't need changes.
If re-implementing: the core issue is pystray needs the main thread on some platforms,
which conflicts with uvicorn also wanting the main thread. The VBS launcher approach
(which solved the console window problem) runs python.exe normally so a background-
thread tray should work — but the restart menu item also needs a working API call
(use urllib not requests) and schtasks-based restart, not SIGTERM.

### YOLOv5 — seaborn/pandas/matplotlib missing
Only relevant if a `torchhub` engine model is ever added. The YOLOv5 repo imports
these at top level even for inference. `_load_torchhub()` in `detector.py` auto-installs
them silently if missing. Also in `requirements.txt`.
**Current YOLOv5 models use `engine="ultralytics"` and do not trigger this.**

### YOLOv5 — urllib3 auto-update warning
Only relevant if a `torchhub` engine model is ever added. Ultralytics tries to
auto-update with a malformed pip marker. Suppressed:
```python
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("ULTRALYTICS_AUTOINSTALL", "false")
```

### YOLOv5 — FutureWarning (torch.cuda.amp.autocast)
Only relevant if a `torchhub` engine model is ever added. YOLOv5 uses deprecated
PyTorch API. Suppressed:
```python
warnings.filterwarnings("ignore", category=FutureWarning,
                        message=".*torch.cuda.amp.autocast.*")
```

### Task Scheduler launch — VBS with full python.exe path

**Do not use `pythonw.exe`** — no stdout/stderr, crashes instantly, pystray/asyncio conflicts.

**Do not call `python.exe` directly from the task XML** — `<Hidden>true</Hidden>` in
Task Scheduler XML hides the task from the UI, NOT the console window. The console
still appears.

**Do not use a bare `python.exe` in VBScript** — Task Scheduler logon tasks run with
a stripped system PATH, not the user PATH. Bare `python.exe` finds the wrong Python
(Windows Store stub) and all packages appear missing.

**The correct approach:**
`_write_task_xml.py` is called at install time (while the user's full PATH is active).
It does two things:
1. Generates `start-silent.vbs` with the **full path to python.exe** hardcoded in
2. Writes the task XML to launch `wscript.exe start-silent.vbs`

`WScript.Shell.Run` with window style `0` (hidden) suppresses the console window.
The full Python path ensures the right interpreter and packages are always used.
`start-silent.vbs` is regenerated on every `setup-service.bat` run — don't edit it.

### Windows asyncio ConnectionResetError
BlueIris closing connections throws noise into the log. Suppressed via custom
exception handler in `lifespan()`. Must be set inside lifespan, not at module
import — event loop doesn't exist at import time.

### setup-service.bat — XML must be written by Python, not echo

The original bat used `echo` to build the Task Scheduler XML. This silently breaks
when the install path contains spaces (e.g. `C:\Users\Some User\...`) because
`echo >>` writes ANSI encoding while the XML declaration says `UTF-16` — so
`schtasks` misparsed the file and silently dropped `<WorkingDirectory>`. The task
registered and appeared to work, but Python couldn't find `main.py` or any imports.

Fixed: Python writes the XML (`python -c "..."`) with explicit `encoding='utf-16'`,
passing all paths as `sys.argv` arguments so spaces are handled by shell quoting,
not by string substitution into the XML body.

### onnxruntime-gpu can't be installed via in-app installer
On some Miniconda setups, pip installs to the wrong environment. The dashboard
shows a manual install command instead of an install button for this package.
Do not attempt to fix by running pip differently.

### API key placeholder self-reference (see Authentication section above)
The split string `'__OBJECTIF_' + 'API_KEY__'` is intentional. Do not simplify.

### BlueIris polls the dashboard URL
BlueIris periodically GETs `/` to check health. This produces repeated
"Serving dashboard — injecting API key (1 substitutions)" log lines. Normal.

### Script truncation bug
During patching, a leftover `const _unusedTabsPlaceholder = ` string was left
at the end of the script block, cutting off `init()` and closing tags. Dashboard
loaded blank with unresponsive tabs. Always verify script ends with `init();`
followed by `</script></body></html>`.

---

## Model Registry Notes

- `sha256` field exists on `ModelInfo` but is empty string for all models (no verification yet)
- YOLO v26 URLs are `experimental=True` — GitHub release assets may not exist yet
- ONNX models (EfficientDet, MobileNet SSD) have empty `download_url` — manual install only
- All COCO label sets are 80 classes; torchvision models map from their own label index

---

## Security Rules

- **Paths:** Never in user-facing output. Full paths go to `logs/server.log` only.
- **Exceptions:** Caught server-side, generic message in API response.
- **Config writes:** Exact allowlist `API_UPDATABLE_PATHS` in `config.py` — not prefix match.
  `auth.api_key` is not in the allowlist and cannot be changed via API.
- **Upload limit:** 20 MB cap on `/v1/vision/detection`.
- **Confidence:** Clamped to [0.0, 1.0] before use.

---

## License

AGPL-3.0 due to Ultralytics. The optional ALPR pipeline (`fast-alpr` and its
plate/OCR models) is MIT — it does not change the project's AGPL status, but its
weights are third-party downloads (document this if redistributing). If
commercialisation is ever wanted:
- Purchase an Ultralytics Enterprise License (current pricing on their site), OR
- Remove Ultralytics entirely (torchvision and the fast-alpr/ONNX paths would survive; YOLO `.pt` models would not, as they load via the Ultralytics engine)

**Trademark (separate from the code license):** "Objectif.AI" is claimed as a
common-law (unregistered) trademark — ™, not ®. The ® symbol must NOT be used
anywhere unless/until a registration is actually granted (using ® pre-registration
is itself a violation). Notices live in `README.md` (Trademark section) and the
dashboard Help → Trademark. The notices currently name **RCSPuffs** as owner —
this is fine for an unregistered claim, but note a GitHub handle cannot be the
owner on an actual registration; the maintainer plans to file under their legal
name (or a registered company) if/when registering, at which point update the
notices to match. The AGPL grants rights to the source, NOT to the Objectif.AI
name or logo as third-party branding.

---

## Development Setup

```
pip install -r requirements.txt
python main.py
```

Open `http://localhost:32168`. No build step, no Node.js, no npm.
Edit `static/index.html` directly — it's a single vanilla JS file.

For GPU: see README GPU Setup section.

---

## What's Next (v0.8.x+)

- **Plate history persistence** — currently resets on restart; consider writing to
  a local SQLite DB so history survives restarts
- **ALPR accuracy improvements** — `cct-xs-v2-global-model` OCR has ~92% first-char
  accuracy on North American plates; a dedicated US model would help
- **Custom model slot** — drag-drop a user model + pick engine + class list. Deferred
  intentionally; scope to Ultralytics `.pt`/`.onnx` and `[N,6]` ONNX layout first
- SHA-256 hash verification for downloaded model files (field exists, needs values)
- Legacy GPU broader testing (CC < 7.5 path is unverified on real hardware)
- DirectML real-hardware testing on AMD/Intel GPUs (wired but not yet tested end-to-end)
- Possibly: single .exe installer for non-technical users
- Possibly: per-detection webhook notifications (Discord, Pushover, etc.)
- Possibly: detection history / statistics tab
