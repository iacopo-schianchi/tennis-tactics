# from https://github.com/yastrebksv/TrackNet/blob/main/general.py

import cv2
import numpy as np

BALL_PROB_THRESH = 0.5

def postprocess(ball_prob, W, H):
    ball_prob = ball_prob.reshape((H, W)).astype(np.float32)

    mask_u8 = (ball_prob > BALL_PROB_THRESH).astype(np.uint8) * 255

    circles = cv2.HoughCircles(mask_u8, cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=50, param2=2, minRadius=1, maxRadius=round(7 * H / 360))
    if circles is None: return []

    h, w = ball_prob.shape
    yy, xx = np.mgrid[0:h, 0:w]
    candidates = []
    for x, y, r in circles[0]:
        # circular mask around candidate
        mask = (xx - x) ** 2 + (yy - y) ** 2 <= r ** 2
        vals = ball_prob[mask]
        if vals.size == 0:
            continue

        s = vals.sum()
        cx = float((xx[mask] * vals).sum() / s) if s > 0 else float(x)
        cy = float((yy[mask] * vals).sum() / s) if s > 0 else float(y)

        conf = float(vals.max())
        candidates.append((cx, cy, conf))

    return sorted(candidates, key=lambda c: c[2], reverse=True)