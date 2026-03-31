#!/usr/bin/env python3
import os, sys, json, argparse, time
import cv2
import numpy as np

from utils.io_utils import list_images, load_image, save_image, ensure_dir
from utils.preprocess import resize_long, to_gray, illum_normalize
from utils.edge_detect import canny_percentile, morph_close_then_dilate, kill_border
from utils.quad_detect import (
    find_quads_by_contours,
    quad_from_hough,
    quad_from_bright_page,
    quad_from_frame_fill,
    order_quad,
)
from utils.page_score import (
    composite_score,
    _edge_polarity_score,      # self-check
    _texture_gap_score,        # self-check
    _rectangularity,           # self-check
    _text_line_score,          # inner-photo rejection gate
)
from utils.warp_and_pad import warp_perspective_from_quad, add_padding, trim_warped_background
from utils.orientation import rotation_from_quad, ocr_orientation
from utils.retry_policy import build_retry_configs
from utils.vis_debug import draw_quad, save_step

# ------------------ Default configuration ------------------
DEFAULTS = dict(
    LONG_SIDE=768,
    P_LOW=15,
    P_HIGH=90,
    CLOSE_K=3,
    DILATE_K=3,
    BORDER_KILL=8,
    AREA_MIN_FRAC=0.05,
    AREA_MAX_FRAC=0.95,
    AR_MIN=0.5,
    AR_MAX=2.2,
    PAD_FRAC=0.03,
    JPEG_QUALITY=92,
    ILLUM_K=151,
    ILLUM_STRENGTH=0.88,
    CLAHE_CLIP=3.0,
    DESKEW_CLAMP_DEG=12.0,
    SCORE_MIN_ACCEPT=0.58,
    TRIM_WARP_BG=True,
    # If True, run Tesseract OSD whenever use_ocr is on (not only on low-confidence passes).
    OCR_ALWAYS=False,
)

cv2.setUseOptimized(True)
try:
    cv2.setNumThreads(max(1, os.cpu_count() or 1))
except Exception:
    pass


def _quad_mostly_inside(inner: np.ndarray, outer: np.ndarray) -> bool:
    """True when inner quad is largely inside outer quad."""
    poly = outer.astype(np.int32)
    inside = 0
    for pt in inner.astype(np.float32):
        if cv2.pointPolygonTest(poly, (float(pt[0]), float(pt[1])), False) >= 0:
            inside += 1
    cx = float(np.mean(inner[:, 0]))
    cy = float(np.mean(inner[:, 1]))
    if cv2.pointPolygonTest(poly, (cx, cy), False) >= 0:
        inside += 1
    return inside >= 4


def _suppress_inner_quads(quads):
    """Drop candidates enclosed by a significantly larger candidate."""
    if len(quads) <= 1:
        return quads
    keep = [True] * len(quads)
    areas = [float(cv2.contourArea(q.astype(np.int32))) for q in quads]
    for i in range(len(quads)):
        if not keep[i]:
            continue
        for j in range(len(quads)):
            if i == j or not keep[i]:
                continue
            if areas[j] > 1.20 * areas[i] and _quad_mostly_inside(quads[i], quads[j]):
                keep[i] = False
                break
    return [q for q, k in zip(quads, keep) if k]


# ------------------ Core pipeline ------------------
def process_once(img_bgr, cfg, use_ocr=False, debug_dir=None):
    t0 = time.time()

    # 1) Resize & grayscale
    img_small, scale = resize_long(img_bgr, cfg['LONG_SIDE'])
    gray = to_gray(img_small)
    save_step(debug_dir, gray, '01_gray.jpg')

    # 2) Illumination normalization (retry-controlled)
    if cfg.get('USE_ILLUM', False):
        gray = illum_normalize(
            gray,
            ksize=cfg['ILLUM_K'],
            strength=cfg['ILLUM_STRENGTH'],
            clahe_clip=cfg['CLAHE_CLIP']
        )
        save_step(debug_dir, gray, '02_bgfix.jpg')

    # 3) Edges
    edges = canny_percentile(gray, cfg['P_LOW'], cfg['P_HIGH'])
    edges = morph_close_then_dilate(edges, cfg['CLOSE_K'], cfg['DILATE_K'])
    edges = kill_border(edges, cfg['BORDER_KILL'])
    save_step(debug_dir, edges, '03_edges.jpg')

    # 4) Quad detection with fallbacks
    quads = []
    route_used = 'contours'
    tiny_mode = cfg.get('AREA_MIN_FRAC', 0.05) <= 0.005  # small-card mode

    if cfg.get('FORCE_HOUGH', False):
        q = quad_from_hough(edges, tiny=tiny_mode)
        if q is not None:
            quads = [q]
            route_used = 'hough'
    else:
        # primary: contours
        quads = find_quads_by_contours(
            edges, cfg['AREA_MIN_FRAC'], cfg['AREA_MAX_FRAC'],
            cfg['AR_MIN'], cfg['AR_MAX']
        )
        # fallback: Hough
        if not quads:
            q = quad_from_hough(edges, tiny=tiny_mode)
            if q is not None:
                quads = [q]
                route_used = 'hough'

    # fallback #2: bright-page (largest bright component)
    if not quads:
        q = quad_from_bright_page(gray)
        if q is not None:
            quads = [q]
            route_used = 'bright_page'

    # fallback #3: frame-fill — document fills entire frame, no visible border
    # Triggers when the image centre is bright + textured but no quad found above.
    if not quads:
        q = quad_from_frame_fill(gray)
        if q is not None:
            quads = [q]
            route_used = 'frame_fill'

    # 4.1) Remove inner candidates (photo box inside ID card, etc.)
    quads = _suppress_inner_quads(quads)

    # 5) Pick best quad by score
    best = None
    best_score = -1.0
    for q in quads:
        s = composite_score(gray, edges, q)
        if s > best_score:
            best_score, best = s, q

    accepted = best is not None and best_score >= cfg['SCORE_MIN_ACCEPT']

    # 5.05) Guardrail against selecting inner photo rectangles.
    # If candidate has weak text-line structure and is relatively small, reject it.
    if best is not None and accepted:
        H, W = gray.shape[:2]
        area_frac = cv2.contourArea(best.astype(np.int32)) / float(H * W + 1e-6)
        tscore = _text_line_score(gray, best)
        if area_frac < 0.45 and tscore < 0.20:
            accepted = False

    # 5.1) Self-check for fallback routes (photometric sanity)
    if best is not None and route_used in ('hough', 'bright_page', 'frame_fill', 'minrect'):
        pol = _edge_polarity_score(gray, best)
        tex = _texture_gap_score(gray, best)
        rect = _rectangularity(best)
        # frame_fill quads cover the whole image so polarity/texture gap is
        # naturally weak — use a relaxed threshold for that route only
        pol_thr = 0.20 if route_used == 'frame_fill' else 0.35
        tex_thr = 0.20 if route_used == 'frame_fill' else 0.35
        if (pol < pol_thr) or (tex < tex_thr) or (rect < 0.65):
            accepted = False

    # 6) OCR orientation fallback (optional)
    ocr_angle = None
    ocr_ran = False
    ocr_gate = False
    if use_ocr and (
        cfg.get('OCR_ALWAYS', False)
        or (not accepted or best_score < cfg['SCORE_MIN_ACCEPT'] + 0.1)
    ):
        ocr_gate = True
        ocr_ran = True
        ocr_angle = ocr_orientation(gray)

    # Overlay for debug
    overlay = img_small.copy()
    if best is not None:
        overlay = draw_quad(overlay, best, color=(0, 255, 255))
    save_step(debug_dir, overlay, '04_overlay.jpg')

    # 7) Last-ditch minAreaRect if still nothing
    if (best is None) or (not accepted):
        if best is None:
            cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                c = max(cnts, key=cv2.contourArea)
                rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rect).astype(np.float32)
                best = box
                best_score = 0.45
                route_used = 'minrect'
                # keep 'accepted' False here; self-check for fallbacks will run below

        # run self-check again if we changed best
        if best is not None and route_used in ('hough', 'bright_page', 'frame_fill', 'minrect'):
            pol = _edge_polarity_score(gray, best)
            tex = _texture_gap_score(gray, best)
            rect = _rectangularity(best)
            pol_thr = 0.20 if route_used == 'frame_fill' else 0.35
            tex_thr = 0.20 if route_used == 'frame_fill' else 0.35
            if (pol < pol_thr) or (tex < tex_thr) or (rect < 0.65):
                accepted = False

    if best is None:
        return None, dict(
            accepted=False, score=0.0, angle=None, route='none',
            elapsed_ms=round(1000*(time.time()-t0), 1), scale=round(float(scale), 4)
        )

    # 8) Order, angle, warp
    quad_ord = order_quad(best)
    angle_from_quad = rotation_from_quad(quad_ord)

    # Clamp large deskew only when confidence is not high; allow big rotations for strong quads
    if abs(angle_from_quad) > cfg['DESKEW_CLAMP_DEG'] and best_score < (cfg['SCORE_MIN_ACCEPT'] + 0.2):
        angle_from_quad = np.sign(angle_from_quad) * cfg['DESKEW_CLAMP_DEG']

    final_angle = angle_from_quad
    ocr_applied = False
    if ocr_angle is not None and abs(ocr_angle) <= 90:
        if abs(ocr_angle - angle_from_quad) < 30 or best_score < cfg['SCORE_MIN_ACCEPT'] + 0.05:
            final_angle = ocr_angle
            ocr_applied = True

    warped = warp_perspective_from_quad(img_small, quad_ord)
    # Keep saved image and returned metadata in sync: apply the selected final angle.
    if abs(final_angle) > 0.01:
        h, w = warped.shape[:2]
        center = (w / 2.0, h / 2.0)
        M = cv2.getRotationMatrix2D(center, final_angle, 1.0)
        cos = abs(M[0, 0])
        sin = abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        M[0, 2] += (new_w / 2.0) - center[0]
        M[1, 2] += (new_h / 2.0) - center[1]
        warped = cv2.warpAffine(
            warped,
            M,
            (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
    if cfg.get('TRIM_WARP_BG', True):
        warped = trim_warped_background(warped)
    padded = add_padding(warped, pad_frac=cfg['PAD_FRAC'], pad_color=255)

    elapsed = round(1000*(time.time()-t0), 1)
    info = dict(
        accepted=accepted,
        score=round(float(best_score), 3),
        angle=round(float(final_angle), 2),
        route=route_used,
        elapsed_ms=elapsed,
        scale=round(float(scale), 4),
    )
    if use_ocr:
        info["ocr_meta"] = dict(
            user_enabled=True,
            gate_passed=ocr_gate,
            osd_ran=ocr_ran,
            osd_deg=None if ocr_angle is None else round(float(ocr_angle), 2),
            quad_deg=round(float(angle_from_quad), 2),
            applied=ocr_applied,
            note=(
                "OCR orientation skipped: set DEFAULTS OCR_ALWAYS=True or lower acceptance so gate opens."
                if use_ocr and not ocr_gate
                else None
            ),
        )
    return padded, info


# ------------------ Wrapper for file input ------------------
def process_image_path(path, out_dir, use_ocr=False, debug=False, cfg_overrides=None):
    cfg = DEFAULTS.copy()
    if os.environ.get("OCR_ALWAYS", "").lower() in ("1", "true", "yes"):
        cfg = dict(cfg)
        cfg["OCR_ALWAYS"] = True
    if cfg_overrides:
        cfg.update(cfg_overrides)

    img = load_image(path)
    name = os.path.splitext(os.path.basename(path))[0]

    debug_dir = None
    if debug:
        debug_dir = os.path.join(out_dir, 'debug', name)
        ensure_dir(debug_dir)

    rconfs = build_retry_configs(cfg)
    for attempt, rcfg in enumerate(rconfs):
        # Slightly lower acceptance gate on late retries (never below 0.53)
        if attempt >= 2:
            rcfg = rcfg.copy()
            rcfg['SCORE_MIN_ACCEPT'] = max(0.53, cfg['SCORE_MIN_ACCEPT'] - 0.03)

        padded, info = process_once(img, rcfg, use_ocr=use_ocr, debug_dir=debug_dir)

        if padded is not None and (info['accepted'] or attempt == len(rconfs) - 1):
            angle_tag = f"angle_{info['angle']:+.1f}" if info.get('angle') is not None else "angle_+0.0"
            out_path = os.path.join(out_dir, f"{name}__{angle_tag}.jpg")
            ensure_dir(out_dir)
            save_image(out_path, padded, quality=cfg['JPEG_QUALITY'])

            meta_path = os.path.join(out_dir, f"{name}__meta.json")
            with open(meta_path, 'w') as f:
                json.dump(info, f, indent=2)
            return out_path, info

    return None, dict(accepted=False, score=0.0, angle=None, route='none')


# ------------------ CLI ------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='Folder or single image path')
    ap.add_argument('--output', required=True, help='Output folder')
    ap.add_argument('--debug', action='store_true', help='Save debug steps')
    ap.add_argument('--use_ocr', action='store_true', help='Enable OCR-based orientation fallback')
    ap.add_argument('--strict_100ms', action='store_true', help='Trade accuracy for speed: lower LONG_SIDE, fewer retries')
    args = ap.parse_args()

    if args.strict_100ms:
        DEFAULTS['LONG_SIDE'] = 640  # faster
        # you can also trim retries here if desired

    in_path = args.input
    out_dir = args.output
    ensure_dir(out_dir)

    paths = [in_path] if os.path.isfile(in_path) else list_images(in_path)
    if not paths:
        print('No images found.')
        sys.exit(1)

    results = []
    for p in paths:
        print(f"Processing: {p}")
        out_path, info = process_image_path(p, out_dir, use_ocr=args.use_ocr, debug=args.debug)
        print(f"  -> {info}")
        results.append(dict(name=os.path.basename(p), **info, out=out_path))

    ok = sum(1 for r in results if r['accepted'])
    print(f"Done. {ok}/{len(results)} accepted.")

if __name__ == '__main__':
    main()
