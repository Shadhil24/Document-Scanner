import cv2, numpy as np

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