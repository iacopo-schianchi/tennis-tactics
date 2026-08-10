import numpy as np
import cv2

PADDING_PIXELS = 40
GROUP_THRESH_PX = 20

def get_convex_hull_mask(raw_mask):
    mask = raw_mask.astype(np.uint8) * 255

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return np.zeros_like(mask)

    all_points = np.vstack(contours)
    hull = cv2.convexHull(all_points)

    hull_mask = np.zeros_like(mask)
    cv2.fillConvexPoly(hull_mask, hull, True)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * PADDING_PIXELS + 1, 2 * PADDING_PIXELS + 1))

    return cv2.dilate(hull_mask, kernel)
