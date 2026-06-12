# Objectif.AI User Guide

**Version 0.7.9** | Open source under AGPL-3.0

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

## BlueIris Configuration

In BlueIris → AI → Configure:

| Setting | Value |
|---------|-------|
| Server | Custom server |
| URL | `http://<objectif-ai-machine-ip>:32168` |
| Path | `/v1/vision/detection` |

Set Objectif.AI's minimum confidence **lower** than your BlueIris trigger threshold so BlueIris receives all candidate detections and decides what to act on.

### License-plate recognition (ALPR)

ALPR runs on a **separate endpoint** from object detection, so BlueIris treats it as its own AI server entry. After enabling ALPR in Settings (see below), point a second BlueIris AI configuration at:

| Setting | Value |
|---------|-------|
| URL | `http://<objectif-ai-machine-ip>:32168` |
| Path | `/v1/vision/alpr` |

Each recognized plate comes back as a prediction whose label is the plate text, with a bounding box and a combined detection×OCR confidence. Apply ALPR only to the cameras (or camera clones with a tight area-of-interest) that actually watch a driveway or road — running it on every camera wastes GPU cycles.

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

**Model families:**
- **YOLO v11/v26** — newest generation, best overall
- **YOLO v8** — mature, well tested, excellent OpenVINO support
- **YOLO v5** — classic; requires seaborn/pandas/matplotlib (installed automatically on first load)
- **RT-DETR** — transformer-based, good for complex overlapping scenes, GPU recommended
- **Faster R-CNN** — strong accuracy on difficult scenes, higher latency
- **SSD / SSDLite / RetinaNet** — fast single-stage alternatives, good CPU performance

---

## Downloading and Loading Models

Every model follows the same three-step flow regardless of type:

1. **Download Model** — downloads the weights file to the `models/` folder. A progress bar appears in the model card during download.
2. **Load Model** — loads the downloaded model into memory and makes it active for inference.
3. **Active** — the model is loaded and serving detections to BlueIris.

A **Delete** button appears alongside Load Model for any downloaded model that is not currently active.

**YOLO v5 note:** Downloads the model weights directly from GitHub (same as all other YOLO models). Real percentage progress is shown during download.

**Torchvision note (Faster R-CNN, SSD, SSDLite, RetinaNet):** Same indeterminate progress bar during download. PyTorch saves these to a checkpoints subfolder under `models/` — the exact subfolder varies by PyTorch version but Objectif.AI finds them automatically.

---

## The Console

Each incoming image gets a shape symbol (●, ■, ▲, ◆, ★…). The matching detection result uses the same shape — lets you trace a request to its result even when multiple images arrive at the same time.

| Label | Meaning |
|-------|---------|
| IN (blue) | Image received from BlueIris |
| OUT (cyan) | Detection result returned |
| SYS (purple) | Server status messages |
| WARN (amber) | Warnings |
| ERR (red) | Errors |

In **Settings → Display** toggle between expanded (each detected object on its own indented line, confidence color-coded green/amber/red) and compact (all on one line) display.

The header shows **last** inference time and **avg** (rolling average of last 20). Update frequency is adjustable in Settings.

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

Note: YOLOv5 and Torchvision models require PyTorch CUDA and fall back to CPU on legacy GPUs.

### Intel OpenVINO
Click **Install OpenVINO** on the Hardware tab (~200 MB). Works on Intel Arc and Intel integrated graphics.

### DirectML — any DirectX 12 GPU (NVIDIA, AMD, or Intel)
Click **Install DirectML** on the DirectML card in the Hardware tab (~20 MB), then use **Set as Backend**. DirectML runs ONNX-engine models on any DX12 GPU and is the simplest path to GPU acceleration on AMD and Intel cards on Windows.

> ONNX Runtime ships its CPU, CUDA, and DirectML builds as **mutually exclusive** packages. Installing DirectML automatically removes the conflicting builds first. YOLO `.pt` and Torchvision models are unaffected (they use PyTorch, not ONNX Runtime) — only ONNX-engine models switch to DirectML.

### AMD ROCm
Experimental, not officially tested. See [rocm.docs.amd.com](https://rocm.docs.amd.com).

---

## Detection Settings

**Minimum Confidence:** Server-side floor — detections below this are not sent to BlueIris. Default 30%. Set lower than your BlueIris trigger threshold.

**Class Filter:** When enabled, only selected COCO classes are returned. Grouped by category — click a group header to toggle all at once.

---

## License-Plate Recognition (ALPR)

ALPR is optional and off by default. It uses the open-source [fast-alpr](https://github.com/ankandrew/fast-alpr) pipeline (MIT-licensed) — a YOLOv9-based plate detector plus an OCR model that reads the plate text.

**Enabling it:**

1. Make sure the `fast-alpr` package is installed (Dependencies tab — it is listed under Detection Engine).
2. In **Settings → License Plate Recognition**, turn on **Enable ALPR**. The first time you enable it, the detector and OCR weights download from Hugging Face automatically (a few MB total) — watch the console for progress.
3. Pick a **Detector Model** (larger input size = more accurate but slower) and an **OCR Model**, then click **Reload ALPR** if you change them.
4. Set the **Minimum Plate Confidence** — plates below this combined detection×OCR score are not returned.
5. Add the `/v1/vision/alpr` endpoint as a second AI server in BlueIris (see BlueIris Configuration above).

Once enabled, ALPR auto-loads on every server start until you turn it off. It runs independently of object detection, so you can serve both at once.

> Weights are fetched from Hugging Face on first use and cached locally. If that download host is ever unavailable, the first enable will fail until it is reachable again — already-cached weights keep working offline.

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

### GPU not being used
- Hardware tab — check status dots and package badges
- Dependencies tab — check for missing packages
- Use Set as Backend on the Hardware card
- Restart server — model reloads on new backend automatically

### Model shows "Download Model" after already downloading
If you downloaded a torchvision model (Faster R-CNN, SSD, etc.) and the button still shows "Download Model" after restarting, the file may have been saved to an unexpected location. Check the `models/` folder and all subfolders for `.pth` files. If found, the detection should work on the next model list refresh.

### Legacy GPU first load is slow
ONNX export takes up to 60 seconds per model, once only. Watch the console for progress. Subsequent loads are instant from the cached `.legacy.onnx` file.

### YOLOv5 fails to load
Needs seaborn, pandas, matplotlib — usually installed automatically. If not:
```
pip install seaborn pandas matplotlib
```

### Scheduled task not starting
- Check `logs/startup.log`
- Open Windows Task Scheduler → find ObjectifAI → check Last Run Result
- Run `setup-service.bat` again to re-register

### Model download fails
- YOLO v26 models are Experimental — URLs may not exist yet
- EfficientDet and MobileNet SSD are manual install only (place `.onnx` in `models/`)
- Interrupted downloads clean up `.tmp` files automatically

### Server crashes on startup
Check `logs/startup.log` and `logs/server.log`. Common causes:
- Port 32168 already in use (another instance, or CodeProject.AI still running)
- Missing package — run `pip install -r requirements.txt`
- Corrupt model file — delete from `models/` and re-download

### ALPR not returning plates
- Confirm **Enable ALPR** is on in Settings and the status line reads "Active"
- Check `fast-alpr` is installed (Dependencies tab)
- Confirm the BlueIris AI entry for ALPR uses path `/v1/vision/alpr`, not `/v1/vision/detection`
- Lower the minimum plate confidence
- Plates that are too small, angled, or motion-blurred won't read — use a tighter area-of-interest or a camera clone aimed at the road
- First enable failed? The weight download needs internet on first run — check the console for a download error

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
