import cv2, numpy as np

def trim_warped_background(img_bgr, min_remain_frac=0.70, max_trim_frac=0.52):
    """
    After perspective warp + deskew, strip leftover background bands (e.g. table,
    wood grain) by comparing each edge row/column to a robust document interior
    reference (brightness + fine-structure energy).

    Conservative: if anything looks unsafe, returns the input unchanged.
    """
    if img_bgr is None or img_bgr.size == 0:
        return img_bgr
    h, w = img_bgr.shape[:2]
    if min(h, w) < 48:
        return img_bgr

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32)

    cy0, cy1 = int(0.22 * h), int(0.78 * h)
    cx0, cx1 = int(0.22 * w), int(0.78 * w)
    if cy1 <= cy0 + 5 or cx1 <= cx0 + 5:
        return img_bgr

    a = lab[:, :, 1].astype(np.float32)
    b = lab[:, :, 2].astype(np.float32)
    ref_L = float(np.median(L[cy0:cy1, cx0:cx1]))
    ref_a = float(np.median(a[cy0:cy1, cx0:cx1]))
    ref_b = float(np.median(b[cy0:cy1, cx0:cx1]))
    row_L = np.median(L, axis=1)
    col_L = np.median(L, axis=0)
    row_ab = np.sqrt((np.median(a, axis=1) - ref_a) ** 2 + (np.median(b, axis=1) - ref_b) ** 2)
    col_ab = np.sqrt((np.median(a, axis=0) - ref_a) ** 2 + (np.median(b, axis=0) - ref_b) ** 2)

    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    row_e = np.mean(np.abs(lap), axis=1)
    col_e = np.mean(np.abs(lap), axis=0)
    ksmooth = max(3, (min(h, w) // 50) | 1)
    ker = np.ones(ksmooth, dtype=np.float32) / ksmooth
    row_e = np.convolve(row_e, ker, mode="same")
    col_e = np.convolve(col_e, ker, mode="same")

    ref_er = float(np.median(row_e[cy0:cy1]))
    ref_ec = float(np.median(col_e[cx0:cx1]))
    ref_e = max(ref_er, ref_ec, 1e-3)

    tol_L = 20.0
    thr_e = 0.26 * ref_e
    thr_ab = 14.0
    consec = 2

    def row_doc_like(i):
        lum_ok = row_L[i] >= ref_L - tol_L
        tex_ok = row_e[i] >= thr_e
        col_ok = row_ab[i] <= thr_ab
        return lum_ok and (tex_ok or col_ok)

    def col_doc_like(i):
        lum_ok = col_L[i] >= ref_L - tol_L
        tex_ok = col_e[i] >= thr_e
        col_ok = col_ab[i] <= thr_ab
        return lum_ok and (tex_ok or col_ok)

    def run_from_top():
        ok = 0
        for i in range(h):
            if row_doc_like(i):
                ok += 1
                if ok >= consec:
                    return i - consec + 1
            else:
                ok = 0
        return 0

    def run_from_bottom():
        for i in range(h - consec, -1, -1):
            ok = True
            for j in range(consec):
                r = i + j
                if not row_doc_like(r):
                    ok = False
                    break
            if ok:
                return i + consec - 1
        return h - 1

    def run_from_left():
        ok = 0
        for i in range(w):
            if col_doc_like(i):
                ok += 1
                if ok >= consec:
                    return i - consec + 1
            else:
                ok = 0
        return 0

    def run_from_right():
        for i in range(w - consec, -1, -1):
            ok = True
            for j in range(consec):
                c = i + j
                if not col_doc_like(c):
                    ok = False
                    break
            if ok:
                return i + consec - 1
        return w - 1

    yt = run_from_top()
    yb = run_from_bottom()
    xl = run_from_left()
    xr = run_from_right()

    if yb <= yt + 10 or xr <= xl + 10:
        return img_bgr

    new_h = yb - yt + 1
    new_w = xr - xl + 1
    if new_h < min_remain_frac * h or new_w < min_remain_frac * w:
        return img_bgr
    trim_h = (h - new_h) / float(h)
    trim_w = (w - new_w) / float(w)
    if trim_h > max_trim_frac or trim_w > max_trim_frac:
        return img_bgr

    return img_bgr[yt : yb + 1, xl : xr + 1].copy()


def _target_size_from_quad(quad):
    (tl, tr, br, bl) = quad
    w1 = np.linalg.norm(tr - tl)
    w2 = np.linalg.norm(br - bl)
    h1 = np.linalg.norm(bl - tl)
    h2 = np.linalg.norm(br - tr)
    W = int(round(max(w1, w2)))
    H = int(round(max(h1, h2)))
    W = max(20, W)
    H = max(20, H)
    return W, H

def warp_perspective_from_quad(img, quad):
    quad = quad.astype(np.float32)
    tl, tr, br, bl = quad
    W, H = _target_size_from_quad(quad)
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(quad, dst)
    out = cv2.warpPerspective(img, M, (W, H), flags=cv2.INTER_LINEAR)
    return out

def add_padding(img, pad_frac=0.03, pad_color=255):
    H, W = img.shape[:2]
    p = int(round(pad_frac * min(H, W)))
    if p <= 0:
        return img
    if img.ndim == 2:
        ch = 1
    else:
        ch = img.shape[2]
    new = None
    if ch == 1:
        new = np.full((H + 2 * p, W + 2 * p), pad_color, dtype=img.dtype)
        new[p:p + H, p:p + W] = img
    else:
        new = np.full((H + 2 * p, W + 2 * p, ch), pad_color, dtype=img.dtype)
        new[p:p + H, p:p + W, :] = img
    return new