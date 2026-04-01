import numpy as np
try:
    import pytesseract
except Exception:
    pytesseract = None

def rotation_from_quad(quad):
    # angle of top edge wrt horizontal (degrees). Positive = rotate CCW to deskew.
    (tl, tr, br, bl) = quad
    dx = tr[0] - tl[0]
    dy = tr[1] - tl[1]
    angle = np.degrees(np.arctan2(dy, dx))
    return -angle # rotate by -angle to make top horizontal

def ocr_orientation(gray):
    if pytesseract is None:
        return None
    try:
        txt = pytesseract.image_to_osd(gray)
        # naive parse
        for line in txt.splitlines():
            if 'Rotate:' in line:
                deg = float(line.split(':')[-1].strip())
                # tesseract reports CW rotation; convert to our convention (CCW positive)
                return -deg
    except Exception:
        return None
    return None


def ocr_orientation_with_confidence(gray):
    """
    Returns (angle_ccw, orientation_confidence) from Tesseract OSD.
    angle_ccw can be None if OSD fails.
    confidence can be None if not present in output.
    """
    if pytesseract is None:
        return None, None
    try:
        txt = pytesseract.image_to_osd(gray)
        deg = None
        conf = None
        for line in txt.splitlines():
            line = line.strip()
            if line.startswith("Rotate:"):
                deg = float(line.split(":", 1)[1].strip())
            elif line.startswith("Orientation confidence:"):
                conf = float(line.split(":", 1)[1].strip())
        if deg is None:
            return None, conf
        # tesseract reports CW rotation; convert to our convention (CCW positive)
        return -deg, conf
    except Exception:
        return None, None