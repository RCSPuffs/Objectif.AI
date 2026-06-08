# Objectif.AI — Project Handoff

Read this before making any changes. It explains architecture decisions, recurring
bugs, known quirks, and why things are done the way they are.

**Current version:** v0.7.8
**Working directory convention:** all source in one flat folder + `static/` subfolder.

---

## What This Is

A Python/FastAPI server that replaces CodeProject.AI for BlueIris NVR. BlueIris
POSTs camera images to `/v1/vision/detection` and expects JSON bounding box
detections back. Everything else (dashboard, model management, GPU setup) is
layered on top of that core contract.

**The BlueIris API contract is sacred.** `/v1/vision/detection` must always accept
multipart/form-data and return the exact CodeProject.AI JSON format. BlueIris
users point at this server with zero reconfiguration. Never break this endpoint.

---

## File Structure

```
main.py            FastAPI app, all HTTP + WebSocket endpoints
auth.py            API key generation and verification
config.py          YAML config read/write, in-memory singleton
detector.py        Inference engine (ultralytics, torchvision, onnx; torchhub retained but unused)
downloader.py      Model file download manager with WebSocket progress streaming
hardware.py        Hardware detection (CUDA, OpenVINO, ROCm, CC detection)
console_buffer.py  Circular in-memory log buffer, WebSocket broadcast
model_registry.py  Model catalog — names, URLs, metadata, engine types
dependencies.py    Package checker for the Dependencies tab
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

AGPL-3.0 due to Ultralytics. If commercialisation is ever wanted:
- Purchase an Ultralytics Enterprise License (current pricing on their site), OR
- Remove Ultralytics entirely (torchvision models would survive; YOLOv5 models would not, as their `.pt` files are also loaded via the Ultralytics engine)

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

## What's Next (v0.8+)

- SHA-256 hash verification for downloaded model files (field exists, needs values)
- Legacy GPU broader testing (legacy CC < 7.5 path is unverified on real hardware — tested indirectly)
- Possibly: single .exe installer for non-technical users
- Possibly: per-detection webhook notifications (Discord, Pushover, etc.)
- Possibly: detection history / statistics tab
