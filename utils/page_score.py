import cv2
import numpy as np

# Known long:short aspect ratios for common document types
_DOCUMENT_ARS = (1.364, 1.414, 1.420, 1.586, 1.75)


def _order_quad_pts(quad: np.ndarray) -> tuple:
    """Lightweight tl/tr/br/bl ordering for internal scoring helpers."""
    q = quad.astype(np.float32)
    s = q.sum(axis=1)
    d = np.diff(q, axis=1).reshape(-1)
    return q[np.argmin(s)], q[np.argmin(d)], q[np.argmax(s)], q[np.argmax(d)]


def _text_line_score(gray: np.ndarray, quad: np.ndarray) -> float:
    """
    Warp the quad region to a canonical rectangle and count horizontal
    gradient peaks (proxy for text lines spread across the document).

    Full documents (IDs, passports) produce 5-15+ spread peaks.
    A face photo or single embedded image produces 2-4 concentrated peaks.
    This is the main signal that distinguishes "whole ID card" from
    "just the photo on the ID card".
    """
    try:
        tl, tr, br, bl = _order_quad_pts(quad)
        w = float(np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2.0
        h = float(np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2.0
        if w < 20 or h < 20:
            return 0.5
        scale = 200.0 / max(w, h)
        Wo = max(int(w * scale), 20)
        Ho = max(int(h * scale), 20)
        src = np.array([tl, tr, br, bl], dtype=np.float32)
        dst = np.float32([[0, 0], [Wo - 1, 0], [Wo - 1, Ho - 1], [0, Ho - 1]])
        M = cv2.getPerspectiveTransform(src, dst)
        roi = cv2.warpPerspective(gray, M, (Wo, Ho))
        gy_map = cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3)
        proj = np.mean(np.abs(gy_map), axis=1)
        if proj.max() < 1.0:
            return 0.5
        proj = proj / (proj.max() + 1e-6)
        peaks = sum(
            1 for i in range(1, len(proj) - 1)
            if proj[i] > proj[i - 1] and proj[i] > proj[i + 1] and proj[i] > 0.22
        )
        # Face photo: ~2-4 peaks; full document: 5+ peaks
        return float(np.clip((peaks - 2) / 8.0, 0.0, 1.0))
    except Exception:
        return 0.5


def _document_ar_score(quad: np.ndarray) -> float:
    """
    Small bonus when the quad's aspect ratio matches a known document format
    (ID-1 card 1.586, A4/passport 1.414, business card 1.75, etc.).
    Face photos are typically AR 0.7-1.3, so they score near 0 here.
    """
    try:
        tl, tr, br, bl = _order_quad_pts(quad)
        w = float(np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2.0
        h = float(np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2.0
        if min(w, h) < 1e-3:
            return 0.5
        ar = max(w, h) / (min(w, h) + 1e-6)
        min_dist = min(abs(ar - k) for k in _DOCUMENT_ARS)
        return float(np.clip(1.0 - min_dist / 0.40, 0.0, 1.0))
    except Exception:
        return 0.5


def _rectangularity(quad: np.ndarray) -> float:
    """Corner angles close to 90° → score near 1."""
    q = quad.astype(np.float32)
    v = np.roll(q, -1, axis=0) - q
    angs = []
    for i in range(4):
        a = v[i]
        b = v[(i + 1) % 4]
        cosang = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)
        ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
        angs.append(ang)
    return float(1.0 - np.mean([abs(a - 90.0) / 90.0 for a in angs]))

def _edge_support(edges: np.ndarray, quad: np.ndarray, samples_per_edge: int = 50, tol: int = 3) -> float:
    """How much of the quad perimeter is supported by edge pixels."""
    q = quad.astype(np.float32)
    H, W = edges.shape[:2]
    total = 0
    hits = 0
    for i in range(4):
        p0 = q[i]
        p1 = q[(i + 1) % 4]
        for t in np.linspace(0, 1, samples_per_edge):
            x = int(round(p0[0] * (1 - t) + p1[0] * t))
            y = int(round(p0[1] * (1 - t) + p1[1] * t))
            x = np.clip(x, 0, W - 1)
            y = np.clip(y, 0, H - 1)
            y0, y1 = max(0, y - tol), min(H - 1, y + tol)
            x0, x1 = max(0, x - tol), min(W - 1, x + tol)
            if edges[y0:y1 + 1, x0:x1 + 1].max() > 0:
                hits += 1
            total += 1
    return float(hits / (total + 1e-6))

def _area_ratio(quad: np.ndarray, shape) -> float:
    """Prefer mid-range document coverage; downweight tiny/huge regions."""
    H, W = shape[:2]
    A = float(H * W)
    area = cv2.contourArea(quad.astype(np.int32))
    if area <= 0:
        return 0.0
    r = area / A
    t, u = 0.25, 0.85
    if r < t:
        return float(r / t)
    if r > u:
        return float(u / (r + 1e-6))
    return 1.0

def _border_contrast(gray: np.ndarray, quad: np.ndarray, delta: int = 8) -> float:
    """
    Contrast at the quad border: takes the maximum of two complementary signals.

    1. Intensity gap  — classic bright-doc-on-dark-background signal.
    2. Gradient-ring  — checks whether the gradient magnitude is elevated AT
       the document boundary vs inside the document interior.  This fires even
       when the background and document have similar luminance (e.g. white paper
       on a light-grey desk) because the physical paper edge still creates a
       local gradient transition.
    """
    mask = np.zeros_like(gray)
    cv2.fillConvexPoly(mask, quad.astype(np.int32), 255)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (delta, delta))
    dil = cv2.dilate(mask, k)
    er  = cv2.erode(mask, k)
    ring_out    = cv2.bitwise_and(dil, cv2.bitwise_not(mask))
    ring_in     = cv2.bitwise_and(mask, cv2.bitwise_not(er))
    border_ring = cv2.bitwise_and(dil, cv2.bitwise_not(er))  # thin strip straddling border

    m_out = cv2.mean(gray, ring_out)[0]
    m_in  = cv2.mean(gray, ring_in)[0]
    intensity_score = float(min(1.0, abs(m_in - m_out) / 50.0))

    gx  = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gys = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gys)
    b_mag = cv2.mean(mag, border_ring)[0]
    i_mag = float(cv2.mean(mag, er)[0]) if np.any(er) else 1.0
    gradient_score = float(min(1.0, b_mag / (i_mag + 5.0)))

    return float(max(intensity_score, 0.6 * gradient_score))

# -------- NEW: mild MRZ-band bonus for passport ID pages --------
def _mrz_band_score(gray: np.ndarray) -> float:
    """
    Look for strong dark text bands in the lower ~45% of the page (MRZ region).
    Returns a small bonus [0..1] that helps in busy backgrounds but won't dominate.
    """
    H, W = gray.shape[:2]
    y0 = int(0.55 * H)
    roi = gray[y0:H, :]
    # emphasize horizontal edges (text lines)
    blur = cv2.GaussianBlur(roi, (3, 3), 0)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    proj = np.mean(np.abs(gy), axis=1)        # row-wise activity
    # Normalize via robust stats to avoid outliers
    med = float(np.median(proj))
    std = float(np.std(proj) + 1e-6)
    z = (proj.max() - med) / std              # peak prominence
    # Scale to 0..1; typical good MRZ gives z~4..6
    return float(np.clip(z / 6.0, 0.0, 1.0))

def _edge_polarity_score(gray: np.ndarray, quad: np.ndarray, n=40, d=5) -> float:
    """
    Sample short normal profiles across each edge.
    Score is fraction of samples where inside is brighter than outside.
    """
    q = quad.astype(np.float32)
    ok = 0; tot = 0
    for i in range(4):
        p0, p1 = q[i], q[(i+1) % 4]
        edge = p1 - p0
        L = np.linalg.norm(edge) + 1e-6
        t = edge / L
        nrm = np.array([-t[1], t[0]])  # outward normal (direction doesn't matter; we compare inside vs outside)
        for s in np.linspace(0.05, 0.95, n):
            c = p0*(1-s) + p1*s
            inside = _sample_along(gray, c - nrm*(d//2), c, steps=d)
            outside = _sample_along(gray, c, c + nrm*(d//2), steps=d)
            if inside is None or outside is None: 
                continue
            if float(np.mean(inside)) > float(np.mean(outside)) + 2.0:  # 2 gray levels margin
                ok += 1
            tot += 1
    if tot == 0: 
        return 0.0
    return float(np.clip(ok / tot, 0.0, 1.0))

def _sample_along(img, p0, p1, steps=5):
    H, W = img.shape[:2]
    xs = np.linspace(p0[0], p1[0], steps)
    ys = np.linspace(p0[1], p1[1], steps)
    coords = np.stack([xs, ys], axis=1)
    vals = []
    for x, y in coords:
        xi = int(round(x)); yi = int(round(y))
        if xi < 0 or yi < 0 or xi >= W or yi >= H:
            return None
        vals.append(img[yi, xi])
    return np.array(vals, dtype=np.float32)

def _texture_gap_score(gray: np.ndarray, quad: np.ndarray, delta=6) -> float:
    """
    Laplacian variance outside ring minus inside ring, normalized to [0..1].
    Positive gap means background is more textured than paper.
    """
    mask = np.zeros_like(gray); cv2.fillConvexPoly(mask, quad.astype(np.int32), 255)
    dil = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (delta, delta)))
    er  = cv2.erode(mask,  cv2.getStructuringElement(cv2.MORPH_RECT, (delta, delta)))
    rin  = cv2.bitwise_and(mask, cv2.bitwise_not(er))
    rout = cv2.bitwise_and(dil,  cv2.bitwise_not(mask))

    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    v_in  = float(np.var(lap[rin > 0])) if np.any(rin) else 0.0
    v_out = float(np.var(lap[rout > 0])) if np.any(rout) else 0.0
    # map (v_out - v_in) with soft scaling so typical gaps land ~0.5..0.9
    diff = v_out - v_in
    return float(np.clip( (diff / (diff + 50.0)) if diff > 0 else 0.0, 0.0, 1.0 ))

def composite_score(gray: np.ndarray, edges: np.ndarray, quad: np.ndarray) -> float:
    """
    Final score in [0..1] combining geometry and photometric checks.

    Weight breakdown (must sum to 1.0):
      w_rect  — corner angles close to 90°
      w_edge  — edge pixels along perimeter
      w_area  — prefer mid-range document coverage
      w_con   — border contrast (intensity + gradient)
      w_mrz   — MRZ text-band bonus (passports)
      w_text  — horizontal text-line peaks in warped region  ← helps reject face photos
      w_ar    — aspect ratio close to known document formats ← helps reject face photos
    """
    w_rect = 0.25
    w_edge = 0.28
    w_area = 0.10
    w_con  = 0.15
    w_mrz  = 0.05
    w_text = 0.12
    w_ar   = 0.05
    s = (
        w_rect * _rectangularity(quad) +
        w_edge * _edge_support(edges, quad) +
        w_area * _area_ratio(quad, gray.shape) +
        w_con  * _border_contrast(gray, quad) +
        w_mrz  * _mrz_band_score(gray) +
        w_text * _text_line_score(gray, quad) +
        w_ar   * _document_ar_score(quad)
    )
    return float(max(0.0, min(1.0, s)))
