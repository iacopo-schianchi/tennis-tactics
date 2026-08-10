import cv2
import numpy as np

CANNY_THRESHOLD1 = 20
CANNY_THRESHOLD2 = 80

LINE_THRESHOLD = 40
MIN_LINE_LENGTH = 100
MAX_LINE_GAP = 120

HORIZ_ANGLE_THRESH = 25
VERT_ANGLE_THRESH = 60
MIN_WIDTH_FRACTION = 0.22
MIN_HEIGHT_FRACTION = 0.4

LINE_CLUSTER_THERSH = 25
RHO_THRESH = 20.0
THETHA_THRESH = np.radians(5)

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

    if lines is None: return []

    return lines

def split_by_orientation(lines, img_w, img_h):
    horiz, vert = [], []

    min_length_w = img_w * MIN_WIDTH_FRACTION
    min_length_h = img_h * MIN_HEIGHT_FRACTION

    for line in lines:
        x1, y1, x2, y2 = line.flatten()
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1))) % 180

        # dist from horiz
        horizontal_angle = min(angle, 180 - angle)

        # dist from vert
        vertical_angle = abs(angle - 90)

        if horizontal_angle < HORIZ_ANGLE_THRESH and length >= min_length_w:
            horiz.append(line.flatten())

        elif vertical_angle < VERT_ANGLE_THRESH and length >= min_length_h:
            vert.append(line.flatten())
            
    return horiz, vert

def line_to_rho_theta(line):
    x1, y1, x2, y2 = line
    
    theta = np.arctan2(y2 - y1, x2 - x1)
    
    if theta < 0:
        theta += np.pi
    
    rho = x1 * np.sin(theta) - y1 * np.cos(theta)
    
    if rho < 0:
        rho = -rho
        theta = (theta + np.pi) % np.pi
        
    return rho, theta

def cluster_segments(segments):
    if not segments:
        return []

    params = [line_to_rho_theta(seg) for seg in segments]
    
    clusters = []
    used = [False] * len(segments)

    for i in range(len(segments)):
        if used[i]:
            continue

        current_cluster = [segments[i]]
        used[i] = True
        
        rho1, theta1 = params[i]

        for j in range(i + 1, len(segments)):
            if used[j]:
                continue

            rho2, theta2 = params[j]

            d_theta = abs(theta1 - theta2)
            d_theta = min(d_theta, np.pi - d_theta)

            d_rho = abs(rho1 - rho2)

            if d_rho <= RHO_THRESH and d_theta <= THETHA_THRESH:
                current_cluster.append(segments[j])
                used[j] = True

        clusters.append(current_cluster)

    return clusters

def fit_line_through_segments(cluster):
    points = []
    for line in cluster:
        points.extend([(line[0], line[1]), (line[2], line[3])])

    points = np.asarray(points, dtype=np.float32)

    vx, vy, x0, y0 = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    return vx, vy, x0, y0

def find_infinite_line_intersection(line1, line2):
    vx1, vy1, x1, y1 = line1
    vx2, vy2, x2, y2 = line2

    denom = vx1 * vy2 - vy1 * vx2

    # parallel
    if abs(denom) < 1e-9:
        return None

    dx = x2 - x1
    dy = y2 - y1

    t = (dx * vy2 - dy * vx2) / denom

    x = x1 + t * vx1
    y = y1 + t * vy1

    return (round(x), round(y))

def y_at_x(line, x):
    vx, vy, x0, y0 = line

    if abs(vx) < 1e-8:
        return None

    t = (x - x0) / vx
    return y0 + t * vy


def x_at_y(line, y):
    vx, vy, x0, y0 = line

    if abs(vy) < 1e-8:
        return None

    t = (y - y0) / vy
    return x0 + t * vx

def get_boundary_lines(h_clusters, v_clusters, img_w, img_h):
    h_lines = [fit_line_through_segments(cluster) for cluster in h_clusters]
    v_lines = [fit_line_through_segments(cluster) for cluster in v_clusters]

    center_x = img_w / 2
    center_y = img_h / 2

    h_positions = [(y_at_x(line, center_x), line) for line in h_lines]
    v_positions = [(x_at_y(line, center_y), line)for line in v_lines]

    h_positions = [(pos, line) for pos, line in h_positions if pos is not None]
    v_positions = [(pos, line) for pos, line in v_positions if pos is not None]

    if len(h_positions) < 2 or len(v_positions) < 2:
        return None

    top_line = min(h_positions, key=lambda x: x[0])[1]
    bottom_line = max(h_positions, key=lambda x: x[0])[1]

    left_line = min(v_positions, key=lambda x: x[0])[1]
    right_line = max(v_positions, key=lambda x: x[0])[1]

    return top_line, bottom_line, left_line, right_line

def get_corners(raw_vert, raw_horiz, img_w, img_h):
    v_clusters = cluster_segments(raw_vert)
    h_clusters = cluster_segments(raw_horiz)

    top_line, bottom_line, left_line, right_line = get_boundary_lines(h_clusters, v_clusters, img_w, img_h)

    tl = find_infinite_line_intersection(top_line, left_line)
    tr = find_infinite_line_intersection(top_line, right_line)
    br = find_infinite_line_intersection(bottom_line, right_line)
    bl = find_infinite_line_intersection(bottom_line, left_line)

    if None in (tl, tr, br, bl):
        return None

    return [tl, tr, br, bl]