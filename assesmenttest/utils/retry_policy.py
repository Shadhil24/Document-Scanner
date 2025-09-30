def build_retry_configs(base):
    # Pass 1: fast, clean
    c0 = base.copy(); c0['USE_ILLUM'] = False

    # Pass 2: illumination normalization
    c1 = base.copy(); c1['USE_ILLUM'] = True

    # Pass 3: illum + relaxed edges + smaller min doc
    c2 = base.copy(); c2['USE_ILLUM'] = True
    c2['P_LOW']  = max(5,  base['P_LOW'] - 7)   # e.g., 8
    c2['P_HIGH'] = min(95, base['P_HIGH'] + 3)  # e.g., 93
    c2['CLOSE_K'] = max(3, base['CLOSE_K'] + 2) # e.g., 5
    c2['AREA_MIN_FRAC'] = 0.02                  # allow small docs in large backgrounds

    # Pass 4: Hough-only push for fragmented edges (slower but bounded)
    c3 = base.copy(); c3['USE_ILLUM'] = True
    c3['FORCE_HOUGH'] = True                   # <- new flag
    c3['P_LOW']  = c2['P_LOW']
    c3['P_HIGH'] = c2['P_HIGH']
    c3['CLOSE_K'] = c2['CLOSE_K']
    c3['AREA_MIN_FRAC'] = 0.015

    c4 = base.copy(); c4['USE_ILLUM'] = True
    c4['AREA_MIN_FRAC'] = 0.004
    c4['AR_MIN'] = 0.6
    c4['AR_MAX'] = 2.5  # tiny cards
    
    return [c0, c1, c2, c3, c4]
