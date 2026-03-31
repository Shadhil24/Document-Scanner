import os, cv2, glob

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def list_images(folder, exts=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
    out = []
    for ext in exts:
        out.extend(glob.glob(os.path.join(folder, f"**/*{ext}"), recursive=True))
    return sorted(out)

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img

def save_image(path, img, quality=92):
    params = []
    if path.lower().endswith(('.jpg', '.jpeg')):
        params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
    cv2.imwrite(path, img, params)