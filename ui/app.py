import base64
import io
import json
import os
import zipfile
from uuid import uuid4

import requests
from flask import Flask, Response, render_template, request

app = Flask(__name__)
BACKEND_URL = os.getenv("BACKEND_URL", "http://doc-scanner-api:8000")
BATCH_CACHE = {}


@app.get("/")
def index():
    return render_template("index.html", result=None, error=None, mode="file")


def _store_batch(items):
    batch_id = str(uuid4())
    BATCH_CACHE[batch_id] = items
    return batch_id


@app.get("/download/item/<batch_id>/<int:item_idx>/<kind>")
def download_item(batch_id, item_idx, kind):
    items = BATCH_CACHE.get(batch_id, [])
    if item_idx < 0 or item_idx >= len(items):
        return Response("Invalid item", status=404)
    item = items[item_idx]

    if kind == "input":
        data = item.get("input_bytes")
        if not data:
            return Response("No input bytes found", status=404)
        filename = f'{os.path.splitext(item["name"])[0]}__input.jpg'
    elif kind == "output":
        data = item.get("output_bytes")
        if not data:
            return Response("No output bytes found", status=404)
        filename = f'{os.path.splitext(item["name"])[0]}__output.jpg'
    else:
        return Response("Invalid kind", status=400)

    return Response(
        data,
        mimetype="image/jpeg",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/download/batch/<batch_id>")
def download_batch(batch_id):
    items = BATCH_CACHE.get(batch_id, [])
    if not items:
        return Response("Batch not found", status=404)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for idx, item in enumerate(items):
            stem = os.path.splitext(item["name"])[0] or f"item_{idx+1}"
            in_bytes = item.get("input_bytes")
            out_bytes = item.get("output_bytes")
            if in_bytes:
                zf.writestr(f"{stem}/{stem}__input.jpg", in_bytes)
            if out_bytes:
                zf.writestr(f"{stem}/{stem}__output.jpg", out_bytes)
            meta = item.get("meta", {})
            zf.writestr(f"{stem}/{stem}__meta.json", json.dumps(meta, indent=2))

    buf.seek(0)
    return Response(
        buf.getvalue(),
        mimetype="application/zip",
        headers={"Content-Disposition": 'attachment; filename="scan_results.zip"'},
    )


@app.post("/scan")
def scan():
    mode = request.form.get("mode", "file")
    use_ocr = str(request.form.get("use_ocr") == "on").lower()
    data = {"use_ocr": use_ocr, "debug": "false"}

    try:
        if mode == "folder":
            folder_files = [f for f in request.files.getlist("files") if f and f.filename]
            if not folder_files:
                return render_template(
                    "index.html",
                    result=None,
                    error="Please choose a folder with at least one image.",
                    mode="folder",
                )

            items = []
            accepted_count = 0
            for f in folder_files:
                input_bytes = f.read()
                files = {
                    "file": (f.filename, io.BytesIO(input_bytes), f.mimetype or "application/octet-stream")
                }
                resp = requests.post(f"{BACKEND_URL}/scan/bytes", files=files, data=data, timeout=180)
                if resp.status_code != 200:
                    items.append(
                        {
                            "name": f.filename,
                            "meta": {"accepted": False, "route": "error"},
                            "error": f"Backend error: {resp.status_code}",
                            "input_bytes": input_bytes,
                            "output_bytes": None,
                        }
                    )
                    continue

                meta = {}
                output_bytes = None
                content_type = resp.headers.get("Content-Type", "")
                if "application/json" in content_type:
                    payload = resp.json()
                    meta = payload.get("meta", {})
                else:
                    meta_raw = resp.headers.get("X-Doc-Meta", "{}")
                    try:
                        meta = json.loads(meta_raw)
                    except json.JSONDecodeError:
                        meta = {}
                    output_bytes = resp.content

                if meta.get("accepted"):
                    accepted_count += 1

                items.append(
                    {
                        "name": f.filename,
                        "meta": meta,
                        "error": None,
                        "input_bytes": input_bytes,
                        "output_bytes": output_bytes,
                    }
                )

            batch_id = _store_batch(items)
            ui_items = []
            for i, item in enumerate(items):
                ui_items.append(
                    {
                        "name": item["name"],
                        "meta": item["meta"],
                        "error": item["error"],
                        "input_b64": base64.b64encode(item["input_bytes"]).decode("utf-8") if item["input_bytes"] else None,
                        "output_b64": base64.b64encode(item["output_bytes"]).decode("utf-8") if item["output_bytes"] else None,
                        "index": i,
                    }
                )

            return render_template(
                "index.html",
                result={"mode": "folder", "items": ui_items, "accepted_count": accepted_count, "batch_id": batch_id},
                error=None,
                mode="folder",
            )

        file = request.files.get("file")
        if file is None or not file.filename:
            return render_template(
                "index.html",
                result=None,
                error="Please choose an image file.",
                mode="file",
            )

        input_bytes = file.read()
        files = {"file": (file.filename, io.BytesIO(input_bytes), file.mimetype or "application/octet-stream")}
        resp = requests.post(f"{BACKEND_URL}/scan/bytes", files=files, data=data, timeout=60)
        if resp.status_code != 200:
            return render_template(
                "index.html",
                result=None,
                error=f"Backend error: {resp.status_code} {resp.text[:200]}",
                mode="file",
            )

        content_type = resp.headers.get("Content-Type", "")
        if "application/json" in content_type:
            payload = resp.json()
            meta = payload.get("meta", {})
            item = {
                "name": file.filename,
                "meta": meta,
                "error": None,
                "input_bytes": input_bytes,
                "output_bytes": None,
            }
            batch_id = _store_batch([item])
            return render_template(
                "index.html",
                result={
                    "mode": "file",
                    "meta": meta,
                    "filename": file.filename,
                    "input_b64": base64.b64encode(input_bytes).decode("utf-8"),
                    "image_b64": None,
                    "batch_id": batch_id,
                },
                error=None,
                mode="file",
            )

        meta_raw = resp.headers.get("X-Doc-Meta", "{}")
        try:
            meta = json.loads(meta_raw)
        except json.JSONDecodeError:
            meta = {}

        output_bytes = resp.content
        item = {
            "name": file.filename,
            "meta": meta,
            "error": None,
            "input_bytes": input_bytes,
            "output_bytes": output_bytes,
        }
        batch_id = _store_batch([item])
        image_b64 = base64.b64encode(output_bytes).decode("utf-8")
        result = {
            "mode": "file",
            "meta": meta,
            "image_b64": image_b64,
            "input_b64": base64.b64encode(input_bytes).decode("utf-8"),
            "filename": file.filename,
            "batch_id": batch_id,
        }
        return render_template("index.html", result=result, error=None, mode="file")
    except requests.RequestException as exc:
        return render_template("index.html", result=None, error=f"Cannot reach backend: {exc}", mode=mode)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
