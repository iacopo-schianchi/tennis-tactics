# from https://github.com/yastrebksv/TrackNet/blob/main/general.py

import cv2
import numpy as np

def postprocess(feature_map):
    feature_map = (feature_map.reshape((360, 640)) * 255).astype(np.uint8)
    _, heatmap = cv2.threshold(feature_map, 127, 255, cv2.THRESH_BINARY)

    # was minDist=1
    circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=5, param1=50, param2=2, minRadius=2, maxRadius=7)
    if circles is None: return []

    h, w = feature_map.shape
    yy, xx = np.mgrid[0:h, 0:w]
    candidates = []
    for x, y, r in circles[0]:
        # circular mask around candidate
        mask = (xx - x) ** 2 + (yy - y) ** 2 <= r ** 2
        vals = feature_map[mask].astype(np.float32) / 255
        if vals.size == 0:
            continue

        peak = float(vals.max())
        mean = float(vals.mean())
        mass = float(vals.sum())

        cx = float((xx[mask] * vals).sum() / vals.sum()) if vals.sum() > 0 else float(x)
        cy = float((yy[mask] * vals).sum() / vals.sum()) if vals.sum() > 0 else float(y)

        score = peak * mean
        candidates.append((cx, cy, score, dict(peak=peak, mean=mean, mass=mass, r=float(r))))

    return sorted(candidates, key=lambda c: c[2], reverse=True)