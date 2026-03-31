# Document Scanner using OpenCV
Fast, CPU‑friendly document scanner that:
- detects the page (passport/ID/doc) as a quadrilateral,
- deskews to upright orientation,
- crops and saves a clean, padded result,
- exposes a simple CLI (`scan`) and an HTTP API (FastAPI).

## Contents

```
api.py                  # FastAPI app
main.py                 # CLI + core pipeline (process_once)
utils/                  # preprocessing, edges, Hough/contours, warp/pad, scoring
app/main.py             # Alt. API entry (kept for compatibility)
data/                   # sample images
```

## Prereqs

- **Python**: 3.10 – 3.12 (3.13 is too new for some wheels as of now)
- **Windows / macOS / Linux supported**
- (Optional) For OCR metadata cues, install the **Tesseract** binary:
  - Windows: https://github.com/UB-Mannheim/tesseract/wiki
  - macOS: `brew install tesseract`
  - Linux (Debian/Ubuntu): `sudo apt-get install tesseract-ocr`

## Setup (pip)

```bash
# From the repo root
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## Run the CLI

```bash
# Scan a single image
python main.py --input data/image1.jpg --output outputs

# Or an entire folder
python main.py --input data --output outputs

# Helpful flags
# --debug        writes intermediate images to ./debug/<file>/
# --use-ocr      uses OCR hints for orientation if available
# --strict-100ms favors the fastest path and skips slow fallbacks
```

Outputs include the cropped/padded image and a sidecar `__meta.json` with:
- `accepted` (bool), `score` (float), `route` used, and `angle` applied.

## Run the API (FastAPI)

```bash
# dev server
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

### Endpoints

- `GET /health` – health check
- `POST /scan` – single image upload
- `POST /scan/batch` – multiple files (form-data)
- `POST /scan/bytes` – image bytes download
- `POST /scan/angle` – quick angle/score metadata

**Example** (single file):

```bash
curl -X POST "http://127.0.0.1:8000/scan/angle" \
  -F "file=@data/image1.jpg" \
  -F "use_ocr=false" \
  -F "debug=false"
```

## Run with Docker

Build image:

```bash
docker build -t document-scanner:latest .
```

Run API container:

```bash
docker run --rm -p 8000:8000 \
  -v "$(pwd)/outputs_api:/app/outputs_api" \
  -v "$(pwd)/data:/app/data" \
  document-scanner:latest
```

PowerShell equivalent:

```powershell
docker run --rm -p 8000:8000 `
  -v "${PWD}\outputs_api:/app/outputs_api" `
  -v "${PWD}\data:/app/data" `
  document-scanner:latest
```

Or with Compose:

```bash
docker compose up --build
```

Then open:

- Flask UI: `http://127.0.0.1:5000`
- FastAPI docs: `http://127.0.0.1:8000/docs`
- FastAPI health: `http://127.0.0.1:8000/health`

### Architecture

- `flask-ui` (port `5000`) is the browser-facing upload UI.
- `doc-scanner-api` (port `8000`) is the FastAPI backend for scanning.
- Flask calls FastAPI internally via Docker network (`http://doc-scanner-api:8000`).

## Common Issues

- **`ModuleNotFoundError: No module named 'cv2'`**: Ensure your venv is activated and run `pip install -r requirements.txt`.
- **`uvicorn` not found**: Use `python -m uvicorn api:app --reload --host 0.0.0.0 --port 8000`.
- **Import errors from `main` or `utils`**: Run commands from the project root so modules resolve correctly.

## Dev Notes

- Core entry: `main.py` with `process_once(img_bgr, cfg, ...)`.
- Configure speed/quality via the `DEFAULTS` dict in `main.py`.
- Hough + contour routes with page scoring (`utils/page_score.py`) select the best quad, then we warp & pad.
- If an image nearly fills the frame (edges cut off), the Hough “bright page” route can accept near‑full quads based on heuristics.

---

© 2025 Shadhil24
