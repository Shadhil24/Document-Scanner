import os, cv2, numpy as np

def draw_quad(img, quad, color=(0,255,255), thickness=2):
    out = img.copy()
    q = quad.astype(int)
    for i in range(4):
        p0 = tuple(q[i])
        p1 = tuple(q[(i+1)%4])
        cv2.line(out, p0, p1, color, thickness, cv2.LINE_AA)
    return out

def save_step(debug_dir, img, name):
    if debug_dir is None:
        return
    os.makedirs(debug_dir, exist_ok=True)
    path = os.path.join(debug_dir, name)
    if img.ndim == 2:
        cv2.imwrite(path, img)
    else:
        cv2.imwrite(path, img)