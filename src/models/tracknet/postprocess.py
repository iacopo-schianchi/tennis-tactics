# from https://github.com/yastrebksv/TrackNet/blob/main/general.py

import cv2
import numpy as np

def postprocess(feature_map):
    feature_map = (feature_map.reshape((360, 640)) * 255).astype(np.uint8)
    _, heatmap = cv2.threshold(feature_map, 127, 255, cv2.THRESH_BINARY)

    circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=50, param2=2, minRadius=2, maxRadius=7)
    if circles is None: return []

    candidates = [(float(x), float(y), float(feature_map[int(round(y)), int(round(x))]) / 255) for x, y, r in circles[0]]
    return sorted(candidates, key=lambda c: c[2], reverse=True)