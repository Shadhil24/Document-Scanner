import cv2, numpy as np

def canny_percentile(gray, p_low=15, p_high=90):
    # Compute thresholds from gradient magnitude percentiles
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    lo = np.percentile(mag, p_low)
    hi = np.percentile(mag, p_high)
    lo = max(1.0, lo)
    hi = max(lo + 1.0, hi)
    edges = cv2.Canny(gray, lo, hi, L2gradient=True)
    return edges

def morph_close_then_dilate(edges, k_close=3, k_dilate=3):
    k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (int(k_close)|1, int(k_close)|1))
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (int(k_dilate)|1, int(k_dilate)|1))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k1)
    dil = cv2.dilate(closed, k2, iterations=1)
    return dil

def kill_border(binary, n=8):
    if n <= 0:
        return binary
    h, w = binary.shape[:2]
    out = binary.copy()
    out[:n, :] = 0
    out[-n:, :] = 0
    out[:, :n] = 0
    out[:, -n:] = 0
    return out