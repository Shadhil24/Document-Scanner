import cv2
import numpy as np

def _approx_quad(cnt, eps_frac: float = 0.02):
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, eps_frac * peri, True)
    if len(approx) == 4 and cv2.isContourConvex(approx):
        return approx.reshape(-1, 2).astype(np.float32)
    return None

def _area_frac(cnt, shape) -> float:
    h, w = shape[:2]
    A = float(h * w)
    return float(cv2.contourArea(cnt) / (A + 1e-6))

def find_quads_by_contours(edges: np.ndarray,
                           area_min_frac: float = 0.05,
                           area_max_frac: float = 0.95,
                           ar_min: float = 0.5,
                           ar_max: float = 2.2):
    """Primary route: external contours → approxPolyDP → 4-gons with basic sanity checks."""
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    quads = []
    for c in cnts:
        af = _area_frac(c, edges.shape)
        if af < area_min_frac or af > area_max_frac:
            continue
        q = _approx_quad(c)
        if q is None:
            continue
        x, y, w, h = cv2.boundingRect(q.astype(np.int32))
        ar = max(w, h) / float(min(w, h) + 1e-6)
        if ar < ar_min or ar > ar_max:
            continue
        quads.append(q)
    quads.sort(key=lambda q: cv2.contourArea(q.astype(np.int32)), reverse=True)
    return quads

def order_quad(pts: np.ndarray) -> np.ndarray:
    """Return points in (tl, tr, br, bl) order."""
    pts = np.asarray(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def _line_intersection(l1, l2):
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-6:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den
    return np.array([px, py], dtype=np.float32)

def quad_from_hough(edges: np.ndarray, tiny=False):
    """
    Fallback: build a quad hypothesis from long, near-orthogonal Hough lines.
    Tuned to be a bit more sensitive for desk/cloth scenes with faint edges.
    """
    H, W = edges.shape[:2]
    min_len = int((0.35 if not tiny else 0.18) * min(H, W))      # demand longer lines (helps reject clutter)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=80 if not tiny else 60,                               # slightly lower than 100 for sensitivity
        minLineLength=min_len,
        maxLineGap=int(0.04 * min(H, W))
    )
    if lines is None or len(lines) < 2:
        return None

    lines = lines[:, 0, :]  # (N, 4)
    # angles in degrees
    angs = np.degrees(np.arctan2(lines[:, 3] - lines[:, 1], lines[:, 2] - lines[:, 0]))

    # cluster into roughly horizontal/vertical groups
    g1_mask = (angs > -30) & (angs < 30)            # near horizontal
    g2_mask = (angs > 60) | (angs < -60)            # near vertical
    if not (np.any(g1_mask) and np.any(g2_mask)):
        return None

    # pick a small set of the longest lines to intersect
    lengths = np.hypot(lines[:, 2] - lines[:, 0], lines[:, 3] - lines[:, 1])
    order = np.argsort(-lengths)
    pick = lines[order[:8]]                         # up to 8 best to form intersections

    # intersect near-perpendicular pairs
    pts = []
    for i in range(len(pick)):
        for j in range(i + 1, len(pick)):
            # enforce near-perpendicular by angle difference
            a = angs[order[i]]
            b = angs[order[j]]
            da = abs(((a - b + 180) % 180) - 90)
            if da > 25:                             # allow some slack
                continue
            p = _line_intersection(pick[i], pick[j])
            if p is not None:
                pts.append(p)

    if len(pts) < 4:
        return None

    pts = np.array(pts, dtype=np.float32)
    hull = cv2.convexHull(pts)
    if len(hull) < 4:
        return None

    approx = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)
    if len(approx) != 4:
        return None

    return approx.reshape(-1, 2).astype(np.float32)

def quad_from_bright_page(gray: np.ndarray):
    """
    Fallback #2: assume the document page is the largest bright region.
    Works well on wood/cloth backgrounds when edges are weak.
    Returns a quad (4x2 float32) or None.
    """
    # 1) normalize and threshold
    g = gray
    # gentle local contrast
    g = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8)).apply(g)
    # adaptive thresh: bright page => foreground=255
    thr = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY, 35, -5)
    # clean small speckle, connect page
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, k, iterations=1)
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, k, iterations=2)

    # 2) largest contour
    cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)

    # reject tiny coverage
    H, W = gray.shape[:2]
    area_frac = cv2.contourArea(c) / float(H*W + 1e-6)
    if area_frac < 0.015:  # allow very small doc in big background
        return None

    # try quad approx, else minAreaRect hull -> approx
    q = _approx_quad(c, eps_frac=0.02)
    if q is None:
        hull = cv2.convexHull(c)
        approx = cv2.approxPolyDP(hull, 0.02*cv2.arcLength(hull, True), True)
        if len(approx) == 4:
            q = approx.reshape(-1,2).astype(np.float32)
        else:
            rect = cv2.minAreaRect(c)
            q = cv2.boxPoints(rect).astype(np.float32)

    return q