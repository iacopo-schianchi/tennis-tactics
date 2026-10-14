# adapted from https://github.com/yastrebksv/TrackNet/blob/main/general.py

import cv2
import numpy as np

MIN_AREA = 2
MAX_AREA = 400

def postprocess(feature_map):
    print("Output shape:", feature_map.shape)
    print("Output min/max:", feature_map.min().item(), feature_map.max().item())

    feature_map = feature_map.reshape((360, 640))
    feature_map = np.clip(feature_map * 255, 0, 255).astype(np.uint8)

    _, binary = cv2.threshold(feature_map, 127, 255, cv2.THRESH_BINARY)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(binary)

    candidates = []

    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]

        if MIN_AREA <= area <= MAX_AREA:
            cx, cy = cents[i]
            confidence = float(feature_map[labels == i].max()) / 255.0
            candidates.append((float(cx), float(cy), confidence))

    candidates.sort(key=lambda c: c[2], reverse=True)
    candidates = [(x, y) for x, y, _ in candidates]

    return candidates