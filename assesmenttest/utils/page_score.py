import cv2
import numpy as np

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

def _border_contrast(gray: np.ndarray, quad: np.ndarray, delta: int = 6) -> float:
    """Mean intensity contrast just inside vs outside the quad border."""
    mask = np.zeros_like(gray)
    cv2.fillConvexPoly(mask, quad.astype(np.int32), 255)
    dil = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (delta, delta)))
    er = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (delta, delta)))
    ring_out = cv2.bitwise_and(dil, cv2.bitwise_not(mask))
    ring_in = cv2.bitwise_and(mask, cv2.bitwise_not(er))
    m_out = cv2.mean(gray, ring_out)[0]
    m_in = cv2.mean(gray, ring_in)[0]
    return float(min(1.0, abs(m_in - m_out) / 50.0))

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
    Final score in [0..1] combining geometry and simple photometric checks.
    We keep MRZ weight small to avoid overfitting to passports.
    """
    w_rect, w_edge, w_area, w_con, w_mrz = 0.30, 0.35, 0.10, 0.15, 0.10
    # w_rect, w_edge, w_area, w_con, w_mrz = 0.30, 0.35, 0.12, 0.15, 0.08
    s = (
        w_rect * _rectangularity(quad) +
        w_edge * _edge_support(edges, quad) +
        w_area * _area_ratio(quad, gray.shape) +
        w_con  * _border_contrast(gray, quad) +
        w_mrz  * _mrz_band_score(gray)
    )
    return float(max(0.0, min(1.0, s)))
