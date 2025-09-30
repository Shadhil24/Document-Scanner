import cv2, numpy as np

def resize_long(img, long_side=768):
    h, w = img.shape[:2]
    if max(h, w) <= long_side:
        return img.copy(), 1.0
    if h >= w:
        scale = long_side / float(h)
    else:
        scale = long_side / float(w)
    out = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return out, scale

def to_gray(img):
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def _clahe(gray, clip=3.0, grid=8):
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    return clahe.apply(gray)

def illum_normalize(gray, ksize=151, strength=0.88, clahe_clip=3.0):
    # brightness-blur division + mild CLAHE
    ksize = int(ksize) | 1
    blur = cv2.GaussianBlur(gray, (ksize, ksize), 0)
    blur = np.maximum(blur, 1)
    div = cv2.divide(gray, blur, scale=255)
    if strength < 1.0:
        div = cv2.addWeighted(gray, 1.0 - strength, div, strength, 0)
    div = _clahe(div, clip=clahe_clip)
    return div