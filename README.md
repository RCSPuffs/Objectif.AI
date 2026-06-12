# Objectif.AI

**Self-hosted AI object detection server for BlueIris NVR — a drop-in CodeProject.AI replacement.**

![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL_v3-blue.svg)
![Platform](https://img.shields.io/badge/platform-Windows%2010%2F11-lightgrey)
![Python](https://img.shields.io/badge/python-3.10+-blue)

Point BlueIris at it and it just works — no reconfiguration, no cloud. Supports 30+ object-detection models across YOLO v5–v26, RT-DETR, Faster R-CNN, SSD, and RetinaNet, plus license-plate recognition (ALPR). Runs on CPU, NVIDIA CUDA, Intel OpenVINO, AMD ROCm, or any DirectX 12 GPU via DirectML.

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
- **License-plate recognition (ALPR)** — optional plate detection + OCR on `/v1/vision/alpr`
- **Broad GPU support** — one-click CUDA or OpenVINO, plus DirectML for any DirectX 12 GPU (NVIDIA, AMD, or Intel)
- **Legacy GPU support** — older NVIDIA cards (GTX 10-series, Tesla P4) work via ONNX Runtime
- **Live console** with per-detection confidence, 11 themes, optional auto-start

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
| v0.7.9 | DirectML backend (any DX12 GPU); ALPR via fast-alpr on `/v1/vision/alpr`; fixed MobileNet/EfficientDet ONNX output routing |
| v0.7.8 | Security hardening — Host-header allowlist, rate limit, WS tickets, config validation |
| v0.7.7 | Removed system tray. Scheduled-task path fixes |
| v0.7.6 | YOLOv5 switched to direct GitHub release URLs |

Earlier versions documented in [HANDOFF.md](HANDOFF.md).
