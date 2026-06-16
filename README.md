# Objectif.AI

**Self-hosted AI object detection server for BlueIris NVR — a drop-in CodeProject.AI replacement.**

![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL_v3-blue.svg)
![Platform](https://img.shields.io/badge/platform-Windows%2010%2F11-lightgrey)
![Python](https://img.shields.io/badge/python-3.10+-blue)

Point BlueIris at it and it just works — no reconfiguration, no cloud. Supports 30+ object-detection models across YOLO v5–v26, RT-DETR, Faster R-CNN, SSD, and RetinaNet, plus license-plate recognition (ALPR) with plate suppression and per-plate cooldown timers. Runs on CPU, NVIDIA CUDA, Intel OpenVINO, AMD ROCm, or any DirectX 12 GPU via DirectML.

---

## Quick Start

1. Install [Python 3.10+](https://www.python.org/downloads/) — tick **Add to PATH**
2. Download the [latest release](../../releases/latest) and extract it
3. Run `start.bat`
4. Open <http://localhost:32168>, pick a model from **Model Browser**, click Download → Load
5. In BlueIris → AI → Configure, set the Server URL to `http://<this-machine-ip>:32168`

For auto-start on login: run `setup-service.bat`. Full docs in the dashboard Help tab or [USER_GUIDE.md](USER_GUIDE.md).

---

## Features

- **Drop-in BlueIris replacement** — same API as CodeProject.AI, zero reconfiguration
- **30+ detection models** — switch between them in the dashboard
- **License-plate recognition (ALPR)** — full Blue Iris 5.9.9 Plate Recognizer SDK support, with plate-crop thumbnails and full source image history that persists across restarts
- **Plate storage** — every read saved to disk (crop + compressed full image), browsable in a history modal with inline full-image expand; configurable day-based retention
- **Plate suppression list** — suppress known plates entirely or set per-plate cooldown timers (0–24h)
- **Broad GPU support** — one-click CUDA or OpenVINO, plus DirectML for any DirectX 12 GPU (NVIDIA, AMD, or Intel)
- **Legacy GPU support** — older NVIDIA cards (GTX 10-series, Tesla P4) work via ONNX Runtime
- **Diagnostic console** — per-line model/endpoint tag and backend badge (so a silent CPU fallback is obvious), preprocess/inference/postprocess timing split, per-detection confidence, and a per-type message filter
- **Live system stats** — CPU, RAM, app memory, and uptime in the header
- **Zero overhead when features are off** — ALPR storage, database, and background work only exist when ALPR is enabled and plates are being read
- **11 themes**, optional auto-start on login

---

## Security

LAN-only by design. **Do not expose port 32168 to the internet.** Built-in mitigations include a Host-header allowlist (blocks DNS rebinding), per-IP rate limiting on the detection endpoint, and single-use WebSocket tickets. See [USER_GUIDE.md](USER_GUIDE.md) for details.

---

## Development Notes

This project was built collaboratively with AI (Anthropic's Claude). The code and documentation were AI-generated; I directed the design and tested against real BlueIris deployments.

---

## Trademark

**Objectif.AI™** is a trademark of RCSPuffs. All rights reserved.

The Objectif.AI name and logo identify this project and may not be used to name, brand, or endorse third-party or derivative products without permission. This is a common-law (unregistered) trademark claim — the ™ symbol is used to assert it. The AGPL-3.0 license below grants rights to the *source code*; it does not grant any right to use the Objectif.AI name or logo as your own product's branding. If you fork or redistribute, please use your own name for the distributed product.

---

## License

**[AGPL-3.0](LICENSE)** — uses Ultralytics YOLO, which is also AGPL-3.0. Any redistribution must include source code. Commercial use requires an [Ultralytics Enterprise License](https://ultralytics.com/license). The optional ALPR pipeline (fast-alpr and its plate-detection/OCR models) is MIT-licensed; its weights download from Hugging Face on first use.

---

## Version History

| Version | Notes |
|---------|-------|
| v0.8.5.1 | **Persistent plate storage.** Every ALPR read saves a plate crop and compressed full source image to a `plates/` directory. SQLite database persists the last 1,000 reads across restarts. History modal adds inline "View Full" expand showing crop + full image side by side. Configurable day-based retention (default 30 days, 0 = keep forever). All storage is lazy-initialised — no files, no DB, zero overhead if ALPR is off. |
| v0.8.5 | **Blue Iris 5.9.9 ALPR + console observability.** Full Plate Recognizer SDK compatibility (`POST /` and `/v1/plate-reader/`); fixed ONNX Runtime provider conflict and per-character OCR confidence bug. New ALPR Settings tab (engine config, global cooldown, per-plate suppression, Blue Iris setup guide). Console now shows model/endpoint tag, backend badge, and timing split. Header system stats (CPU/RAM/App/uptime). Per-type console filter. |
| v0.7.9.2 | Fixed Dependencies tab 500 when an optional package (e.g. seaborn) has a broken import chain — version checks no longer execute modules |
| v0.7.9.1 | Fixed DirectML/GPU installs silently no-op'ing — native runtime swaps now force-reinstall and prompt for the required restart |
| v0.7.9 | DirectML backend (any DX12 GPU); ALPR via fast-alpr; fixed MobileNet/EfficientDet ONNX output routing |
| v0.7.8 | Security hardening — Host-header allowlist, rate limit, WS tickets, config validation |
| v0.7.7 | Removed system tray. Scheduled-task path fixes |
| v0.7.6 | YOLOv5 switched to direct GitHub release URLs |

Earlier versions documented in [HANDOFF.md](HANDOFF.md).
