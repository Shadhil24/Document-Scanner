# Document Scanner using OpenCV
Fast, CPU‑friendly document scanner that:
- detects the page (passport/ID/doc) as a quadrilateral,
- deskews to upright orientation,
- crops and saves a clean, padded result,
- exposes a simple CLI (`scan`) and an HTTP API (FastAPI).

## Contents

```
assesmenttest/
  api.py                # FastAPI app (POST /scan)
  main.py               # CLI + core pipeline (process_once)
  utils/                # helpers: preprocessing, edges, Hough/contours, warp/pad, scoring
app/
  main.py               # Alt. API entry (kept for compatibility)
data/                   # sample images
```

> **Note on package name**: the package folder is **`assesmenttest`** (two 's' after 'a'), and the Poetry
> project name is set to match. If you previously used `assessmenttest` (with an extra 's'),
> update imports/commands accordingly.

## Prereqs

- **Python**: 3.10 – 3.12 (3.13 is too new for some wheels as of now)
- **Windows / macOS / Linux supported**
- (Optional) For OCR metadata cues, install the **Tesseract** binary:
  - Windows: https://github.com/UB-Mannheim/tesseract/wiki
  - macOS: `brew install tesseract`
  - Linux (Debian/Ubuntu): `sudo apt-get install tesseract-ocr`

## Setup (Poetry)

```bash
# From the repo root (where pyproject.toml lives)
poetry env use 3.12        # or 3.11 / 3.10
poetry install
```

If you only want dependency management (no packaging), you can do:

```toml
# pyproject.toml
[tool.poetry]
package-mode = false
```

…but packaging is already configured correctly here, so you shouldn't need that.

## Run the CLI

```bash
# Scan a single image
poetry run scan --input data/image1.jpg --out outputs

# Or an entire folder
poetry run scan --input data --out outputs

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
poetry run uvicorn assesmenttest.api:app --reload --host 0.0.0.0 --port 8000
```

### Endpoints

- `GET /` – health & version
- `POST /scan` – single image upload
- `POST /batch_scan` – multiple files (form-data)

**Example** (single file):

```bash
curl -X POST "http://127.0.0.1:8000/scan"   -F "file=@data/image1.jpg"   -F "use_ocr=false" -F "debug=false" -F "strict_100ms=false" -F "save=true"
```

The response returns a base64 JPEG and processing metadata. When `save=true` it saves to `./api_outputs/crops`.

## Common Issues

- **`ModuleNotFoundError: No module named 'cv2'`**: Ensure you are inside the Poetry venv and ran `poetry install` successfully.
- **Project not installed / script not found**: Run inside the project folder (where `pyproject.toml` is) and use `poetry run scan ...`.
- **Import errors from `main` or `utils`**: This README ships with fixed package‑relative imports (e.g., `from assesmenttest.main import ...`). Avoid running modules from inside subfolders; run from the project root or install the package in editable mode: `poetry install`.

## Dev Notes

- Core entry: `assesmenttest/main.py` with `process_once(img_bgr, cfg, ...)`.
- Configure speed/quality via the `DEFAULTS` dict in `main.py`.
- Hough + contour routes with page scoring (`utils/page_score.py`) select the best quad, then we warp & pad.
- If an image nearly fills the frame (edges cut off), the Hough “bright page” route can accept near‑full quads based on heuristics.

---

© 2025 Shadhil24
