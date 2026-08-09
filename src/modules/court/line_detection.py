import cv2
import numpy as np
from .geometry import fit_line_through_segments

CANNY_THRESHOLD1 = 20
CANNY_THRESHOLD2 = 80

LINE_THRESHOLD = 40
MIN_LINE_LENGTH = 100
MAX_LINE_GAP = 120

HORIZ_ANGLE_THRESH = 25
VERT_ANGLE_THRESH = 60
MIN_WIDTH_FRACTION = 0.15
MIN_HEIGHT_FRACTION = 0.3

LINE_CLUSTER_THERSH = 25

def get_lines(mask):
    edges = cv2.Canny(mask, CANNY_THRESHOLD1, CANNY_THRESHOLD2)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=LINE_THRESHOLD,
        minLineLength=MIN_LINE_LENGTH,
        maxLineGap=MAX_LINE_GAP
    )

    return lines

def split_by_orientation(lines, img_w, img_h):
    horiz, vert = [], []

    min_length_w = img_w * MIN_WIDTH_FRACTION
    min_length_h = img_h * MIN_HEIGHT_FRACTION

    for line in lines:
        x1, y1, x2, y2 = line
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1))) % 180

        if (angle < HORIZ_ANGLE_THRESH or angle > 180 - HORIZ_ANGLE_THRESH) and length >= min_length_h:
            horiz.append(line)
        elif abs(angle - 90) < VERT_ANGLE_THRESH and length >= min_length_w:
            vert.append(line)
    
    return horiz, vert


def cluster_segments(centers, segments):
    clusters = []
    curr_segments = []

    for i in range(len(segments)):
        curr = centers[i]
        last = centers[i-1] if i != 0 else None

        if abs(curr - last) <= LINE_CLUSTER_THERSH:
            curr_segments.append(segments[i])
        else:
            clusters.append(curr_segments)
            curr_segments = []

    return clusters

    # for segment in segments:
    #     if len(clusters) == 1 or abs(clusters[-1] - segment) <= LINE_CLUSTER_THERSH:
    #         clusters.

def fit_line_through_segments(cluster):
    points = []
    for line in cluster:
        points.extend([(line[0], line[1]), (line[2], line[3])])

    vx, vy, x0, y0 = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    return vx, vy, x0, y0

def find_infinite_line_intersection(line1, line2):
    x1, y1, x2, y2 = line1[0]
    x3, y3, x4, y4 = line2[0]

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    # parallel
    if abs(denom) < 1e-9:
        return None

    det1 = x1 * y2 - y1 * x2
    det2 = x3 * y4 - y3 * x4

    x = (det1 * (x3 - x4) - (x1 - x2) * det2) / denom
    y = (det1 * (y3 - y4) - (y1 - y2) * det2) / denom

    return (round(x), round(y))

def get_corners(raw_vert, raw_horiz):
    sorted_vert = sorted([(l[1] + l[3]) / 2 for l in raw_vert])
    sorted_horiz = sorted([(l[0] + l[2]) / 2 for l in raw_horiz])

    v_clusters = cluster_segments(sorted_vert, raw_vert)
    h_clusters = cluster_segments(sorted_horiz, raw_horiz)

    if len(h_clusters) < 2 or len(v_clusters) < 2:
        return None

    top_cluster = min(h_clusters, key=lambda c: np.mean([(l[1]+l[3])/2 for l in c]))
    bottom_cluster = max(h_clusters, key=lambda c: np.mean([(l[1]+l[3])/2 for l in c]))
    left_cluster = min(v_clusters, key=lambda c: np.mean([(l[0]+l[2])/2 for l in c]))
    right_cluster = max(v_clusters, key=lambda c: np.mean([(l[0]+l[2])/2 for l in c]))

    top_line = fit_line_through_segments(top_cluster)
    bottom_line = fit_line_through_segments(bottom_cluster)
    left_line = fit_line_through_segments(left_cluster)
    right_line = fit_line_through_segments(right_cluster)
    
    tl = find_infinite_line_intersection(top_line, left_line)
    tr = find_infinite_line_intersection(top_line, right_line)
    br = find_infinite_line_intersection(bottom_line, right_line)
    bl = find_infinite_line_intersection(bottom_line, left_line)

    return [tl, tr, br, bl]