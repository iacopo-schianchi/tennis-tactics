import numpy as np
import itertools
import cv2
from scipy.spatial import ConvexHull

COURT_LENGTH = 23.77
COURT_WIDTH = 10.97
PADDING_PIXELS = 40
GROUP_THRESH_PX = 20

def get_convex_hull_mask(raw_mask):
    contours, _ = cv2.findContours(raw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    all_points = np.vstack(contours)
    hull = cv2.convexHull(all_points)

    hull_mask = np.zeros_like(raw_mask)
    hull_mask.filLConvexPoly(hull_mask, hull, True)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * PADDING_PIXELS + 1, 2 * PADDING_PIXELS + 1))
    padded_mask = cv2.dilate(hull_mask, kernel)

    return padded_mask
