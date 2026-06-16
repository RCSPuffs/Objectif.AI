# Objectif.AI User Guide

**Version 0.8.0** | Open source under AGPL-3.0

---

## Quick Start

1. Run `start.bat` (or `setup-service.bat` to start automatically on login)
2. Open `http://localhost:32168` — dashboard loads automatically
3. Go to **Model Browser**, find a model, and click **Download Model**
4. Once downloaded, click **Load Model**
5. In BlueIris: AI → Configure → Server URL: `http://<this-machine-ip>:32168`

---

## Installation

### Manual (start.bat)
Double-click `start.bat`. Server runs while the console window is open. Close the window to stop the server. Good for testing.

### As a Scheduled Task (setup-service.bat)
Run `setup-service.bat` once. Objectif.AI starts automatically on Windows login with no console window. The **Restart Server** button in Settings becomes available. To remove, run `remove-service.bat` or use Settings → Server → Remove from Startup.

> The task runs as your Windows user account giving it full GPU access. Restarts automatically on crash (up to 10 times).

---

## API Key & Security

On first run a random API key is generated and saved to `config.yaml`. The server embeds it into the dashboard page automatically — no entry required from any device on your network. Just open the URL and the dashboard works.

The key protects management endpoints (downloads, settings, restart). The BlueIris detection endpoint `/v1/vision/detection` does **not** require the key.

> Keep `config.yaml` private. It contains your API key. Never commit it to a public repository.

---

## BlueIris Configuration — Object Detection

In BlueIris → AI → Configure:

| Setting | Value |
|---------|-------|
| Server | Custom server |
| URL | `http://<objectif-ai-machine-ip>:32168` |
| Path | `/v1/vision/detection` |

Set Objectif.AI's minimum confidence **lower** than your BlueIris trigger threshold so BlueIris receives all candidate detections and decides what to act on.

---

## License Plate Recognition (ALPR)

ALPR is optional and off by default. It uses the open-source [fast-alpr](https://github.com/ankandrew/fast-alpr) pipeline (MIT-licensed) — a YOLOv9-based plate detector plus an OCR model that reads the plate text.

All ALPR configuration lives in the **ALPR Settings** tab of the dashboard.

### Step 1 — Enable ALPR in Objectif.AI

1. Go to the **Dependencies** tab and confirm `fast-alpr` is installed. If not, click Install.
2. Go to the **ALPR Settings** tab.
3. Turn on **Enable ALPR**. The first time you enable it, the detector and OCR weights download from Hugging Face automatically (~50 MB total) — watch the console for progress.
4. Set the **Detector Model** and **OCR Model**. Recommended for North American plates:
   - Detector: `yolo-v9-t-384-license-plate-end2end`
   - OCR: `global-plates-mobile-vit-v2-model`
5. Click **Reload ALPR** if you change models.
6. Set **Minimum Plate Confidence** — plates below this combined detection×OCR score are not reported. Start at 20–30% and adjust based on real results.

### Step 2 — Configure Blue Iris 5.9.9

Blue Iris 5.9.9 uses the **Plate Recognizer SDK** protocol for ALPR. This is a separate connection from regular object detection — you do not need to add a second AI server entry.

1. In Blue Iris → **Settings → AI**, find the **License plates (ALPR)** section at the bottom.
2. Set the dropdown to **Plate Recognizer®**.
3. Click **Configure…** next to it.
4. Select **On-Premise SDK Port** and enter this machine's IP and port: `192.168.x.x:32168`
5. Leave the **Region/s** field **blank** (or enter a country code like `us`). Do **not** put a path or URL in this field.
6. Uncheck **Make/Model/Color analysis** — this feature is not supported.
7. Click OK.

### Step 3 — Camera Setup (Recommended)

**Clone your street-facing or driveway camera in Blue Iris.** This is the cleanest approach:

- **Original camera** — detects people, animals, etc. → triggers recordings and alerts as normal
- **Cloned camera** — detects cars and trucks → fires ALPR quietly, no recordings

To set this up on the cloned camera:
1. Right-click the camera → **Camera Properties → AI tab**
2. Tick **License plates**
3. Optionally tick **Only when vehicles are detected** — this sends images to ALPR only when a vehicle is also detected, reducing unnecessary calls
4. Go to the **Alert** tab and **disable recording** on the motion trigger for this clone

The clone shares the same video stream so there is no extra camera or bandwidth overhead.

### Plate Suppression

Use the **Known Plates** section of the ALPR Settings tab to manage which plates are reported to Blue Iris.

**Global Cooldown Default** — applies to all plates not explicitly listed. Set to 0 to always report every plate. Set to a duration to throttle re-reporting. Set to -1 (Never) to suppress all unlisted plates.

**Per-plate entries** — add a plate to set an individual cooldown:
- **Never report** — fully suppresses this plate (useful for your own car parked outside)
- **Always report** — always reports this plate regardless of the global default
- **Duration (5 min – 24h)** — throttles re-reporting; the plate is reported once, then suppressed until the cooldown expires

> Weights are fetched from Hugging Face on first use and cached locally. If that download host is ever unavailable, the first enable will fail until it is reachable again — already-cached weights keep working offline.

---

## Choosing a Model

| Hardware | Recommended |
|----------|-------------|
| Old CPU / no GPU | YOLO v11 Nano |
| Modern CPU only | YOLO v11 Small |
| NVIDIA GTX 1060 (legacy) | YOLO v11 Small — auto ONNX export |
| NVIDIA RTX 2070+ | YOLO v11 Medium or Large |
| NVIDIA RTX 3070+ | YOLO v11 Large or v26 Medium |
| Intel Arc / iGPU | YOLO v8 Small (OpenVINO) |

**Size guide:** Nano = fastest/least accurate. XLarge = slowest/most accurate. Small or Medium is best for most security camera setups — start there and only go larger if you are missing detections you care about.

---

## Downloading and Loading Models

Every model follows the same three-step flow regardless of type:

1. **Download Model** — downloads the weights file to the `models/` folder. A progress bar appears in the model card during download.
2. **Load Model** — loads the downloaded model into memory and makes it active for inference.
3. **Active** — the model is loaded and serving detections to BlueIris.

A **Delete** button appears alongside Load Model for any downloaded model that is not currently active.

---

## The Console

Each incoming image gets a shape symbol (●, ■, ▲, ◆, ★…). The matching detection result uses the same shape — lets you trace a request to its result even when multiple images arrive at the same time.

| Label | Meaning |
|-------|---------|
| IN (blue) | Image received from BlueIris |
| OUT (cyan) | Detection result returned |
| PLATE (green) | License plate recognized (ALPR) |
| SYS (purple) | Server status messages |
| WARN (amber) | Warnings |
| ERR (red) | Errors |

In **Settings → Display** toggle between expanded (each detected object on its own indented line, confidence color-coded green/amber/red) and compact (all on one line) display.

Use the **Filter ▾** button at the top-right of the console to choose which message types appear (requests, detections, plates, system, info, warnings, errors). The selection is saved and survives restarts — handy if you want a quiet console showing only plates, or a verbose one showing every request for troubleshooting.

---

## Hardware & GPU Setup

### Hardware tab
Shows status of each detected hardware component and whether required packages are installed. Use the **Set as Backend** button to switch to your GPU — the model reloads automatically.

### Dependencies tab
Shows every package Objectif.AI uses, whether it is installed, and provides install buttons. Check here first if something is not working.

### NVIDIA CUDA
Click **Install PyTorch CUDA** on the Hardware tab (~1.8 GB). After it finishes:
```
python -m pip install onnxruntime-gpu
```
Then restart the server.

### Legacy GPU — CC below 7.5 (GTX 10-series, Tesla P4, GTX 1060)
Handled automatically. On first YOLO model load, Objectif.AI exports to ONNX (up to 60 seconds — progress shown in console) and caches it in `models/`. Subsequent loads are instant. Requires `onnxruntime-gpu` — install via Dependencies tab.

### Intel OpenVINO
Click **Install OpenVINO** on the Hardware tab (~200 MB). Works on Intel Arc and Intel integrated graphics.

### DirectML — any DirectX 12 GPU (NVIDIA, AMD, or Intel)
Click **Install DirectML** on the DirectML card in the Hardware tab (~20 MB), then use **Set as Backend**. DirectML runs ONNX-engine models on any DX12 GPU and is the simplest path to GPU acceleration on AMD and Intel cards on Windows.

---

## Detection Settings

**Minimum Confidence:** Server-side floor — detections below this are not sent to BlueIris. Default 30%. Set lower than your BlueIris trigger threshold.

**Class Filter:** When enabled, only selected COCO classes are returned. Grouped by category — click a group header to toggle all at once.

---

## Themes

10 themes in **Settings → Display**: Dark, Light, Solarized, Dracula, Nord, Gruvbox, Monokai, Ocean, Forest, Rose, Hi-Contrast. Saved in `config.yaml`, applies to all browsers.

---

## Troubleshooting

### BlueIris not getting detections
- Check a model is loaded (green dot in header)
- Confirm BlueIris URL matches this machine's IP on port 32168
- Console tab — are IN/OUT lines appearing when BlueIris sends images?
- Lower the minimum confidence threshold
- Disable the class filter to rule it out

### ALPR not working — no PLATE lines in console
- Confirm **Enable ALPR** is on in the ALPR Settings tab and the status line reads "Active"
- Check `fast-alpr` is installed (Dependencies tab)
- Confirm the Blue Iris Plate Recognizer config has the correct IP:port and the Region/s field does not contain a path
- Check that **Make/Model/Color analysis** is unchecked in Blue Iris

### ALPR running but not finding plates
- Lower the **Minimum Plate Confidence** in ALPR Settings
- Plates that are too small, angled, or motion-blurred won't read — use a tighter area-of-interest or a camera clone aimed at the road
- Try a clearer plate image via the `/docs` test page at `http://localhost:32168/docs`

### My own car is being reported constantly
- Go to **ALPR Settings → Known Plates**, add your plate, and set cooldown to **Never report**

### GPU not being used
- Hardware tab — check status dots and package badges
- Dependencies tab — check for missing packages
- Use Set as Backend on the Hardware card
- Restart server — model reloads on new backend automatically

### Scheduled task not starting
- Check `logs/startup.log`
- Open Windows Task Scheduler → find ObjectifAI → check Last Run Result
- Run `setup-service.bat` again to re-register

### Server crashes on startup
Check `logs/startup.log` and `logs/server.log`. Common causes:
- Port 32168 already in use (another instance, or CodeProject.AI still running)
- Missing package — run `pip install -r requirements.txt`
- Corrupt model file — delete from `models/` and re-download

---

## Updating

1. Extract the new version zip
2. Copy all `.py` files, `static/index.html`, and any new files over your existing folder
3. **Do not overwrite** `config.yaml` — your settings are preserved
4. **Do not overwrite** the `models/` folder — your downloaded models are preserved
5. Restart the server

---

## For Developers / GitHub

Never commit: `config.yaml`, `logs/`, `models/`, `.deps_installed` — all covered by `.gitignore`.

All user-facing output must use relative paths only. Exception strings go to `logs/server.log` only, never to API responses. Keep `/v1/vision/detection` backward compatible with CodeProject.AI.

**License:** AGPL-3.0 due to Ultralytics. Any distribution must include complete source code.

**Trademark:** Objectif.AI™ is a trademark of RCSPuffs (common-law / unregistered). The AGPL license covers the source code, not the Objectif.AI name or logo — use your own name when redistributing.
