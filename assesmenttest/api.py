# assesmenttest/api.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn
import os, io, json, time, shutil
import cv2
import numpy as np

# Pipeline entry: writes output to disk and returns (out_path, info)
try:
    from assesmenttest.main import process_image_path
except Exception:
    process_image_path = None

app = FastAPI(
    title="Document Scanner API",
    description="CV pipeline for deskew + crop + pad (always saves; no paths exposed)",
    version="1.6.0",
)

# --- CORS (dev-friendly; tighten for prod) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Internal paths (not exposed in responses) ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOAD_DIR = os.path.join(DATA_DIR, "api_uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs_api")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.get("/health")
def health():
    return {"status": "ok", "time": time.time()}

# ---------- Single image → JSON (meta only; auto-saved internally) ----------
@app.post("/scan")
async def scan_json(
    file: UploadFile = File(...),
    use_ocr: bool = Form(False),
    debug: bool = Form(False),
):
    """
    Processes one image. Always saves to disk internally.
    Returns only metadata (no file paths).
    """
    if process_image_path is None:
        return JSONResponse(status_code=500, content={"error": "assesmenttest.main.process_image_path not found"})

    ts = int(time.time() * 1000)
    in_name = f"{ts}_{file.filename}"
    in_path = os.path.join(UPLOAD_DIR, in_name)
    with open(in_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    _, info = process_image_path(in_path, OUTPUT_DIR, use_ocr=use_ocr, debug=debug)
    # Do not expose any saved paths
    return {"meta": info}

# ---------- Batch → JSON list (meta only; auto-saved internally) ----------
@app.post("/scan/batch")
async def scan_batch_json(
    files: list[UploadFile] = File(...),
    use_ocr: bool = Form(False),
    debug: bool = Form(False),
):
    """
    Processes multiple images. Always saves each output internally.
    Returns a list of { name, meta } without any file paths.
    """
    if process_image_path is None:
        return JSONResponse(status_code=500, content={"error": "assesmenttest.main.process_image_path not found"})

    results = []
    ts = int(time.time() * 1000)
    for idx, uf in enumerate(files):
        in_name = f"{ts}_{idx}_{uf.filename}"
        in_path = os.path.join(UPLOAD_DIR, in_name)
        with open(in_path, "wb") as f:
            shutil.copyfileobj(uf.file, f)
        _, info = process_image_path(in_path, OUTPUT_DIR, use_ocr=use_ocr, debug=debug)
        results.append({"name": uf.filename, "meta": info})
    return results

# ---------- Single image → BYTES (download; no paths exposed) ----------
@app.post("/scan/bytes")
async def scan_download(
    file: UploadFile = File(...),
    use_ocr: bool = Form(False),
    debug: bool = Form(False),
):
    """
    Returns processed image bytes and forces a download.
    No paths are exposed; meta returned in headers.
    """
    if process_image_path is None:
        return JSONResponse(status_code=500, content={"error": "assesmenttest.main.process_image_path not found"})

    ts = int(time.time() * 1000)
    safe_name = f"{ts}_{(file.filename or 'input').replace(chr(10),'_').replace(chr(13),'_')}"
    in_path = os.path.join(UPLOAD_DIR, safe_name)
    with open(in_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    out_path, info = process_image_path(in_path, OUTPUT_DIR, use_ocr=use_ocr, debug=debug)

    # Even if underlying pipeline failed to find a page, it should still save a best-effort.
    # If for some reason it didn't, return a JSON explanation (still no paths).
    if not out_path or not os.path.exists(out_path):
        return JSONResponse(status_code=200, content={"message": "No processed file produced", "meta": info})

    # Stream back the saved image as a download
    angle = info.get("angle", 0.0)
    stem = os.path.splitext(os.path.basename(file.filename or "processed"))[0]
    download_name = f"{stem}__angle_{angle:+.1f}.jpg"

    with open(out_path, "rb") as f:
        data = f.read()
    meta_json = json.dumps(info)

    return StreamingResponse(
        io.BytesIO(data),
        media_type="image/jpeg",
        headers={
            "Content-Disposition": f'attachment; filename="{download_name}"',
            "X-Doc-Angle": str(angle),
            "X-Doc-Meta": meta_json,
        },
    )

# ---------- Angle-only helper (meta-lite; no paths exposed) ----------
@app.post("/scan/angle")
async def scan_angle_only(
    file: UploadFile = File(...),
    use_ocr: bool = Form(False),
    debug: bool = Form(False),
):
    """
    Fast check: returns angle/score/route/timings, no paths.
    """
    if process_image_path is None:
        return JSONResponse(status_code=500, content={"error": "assesmenttest.main.process_image_path not found"})

    ts = int(time.time() * 1000)
    in_name = f"{ts}_{file.filename}"
    in_path = os.path.join(UPLOAD_DIR, in_name)
    with open(in_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    _, info = process_image_path(in_path, OUTPUT_DIR, use_ocr=use_ocr, debug=debug)
    return {
        "angle": info.get("angle", 0.0),
        "score": info.get("score", 0.0),
        "route": info.get("route", "unknown"),
        "elapsed_ms": info.get("elapsed_ms", None),
    }

# Local run: python -m assesmenttest.api
if __name__ == "__main__":
    uvicorn.run("assesmenttest.api:app", host="0.0.0.0", port=8000, reload=True)
