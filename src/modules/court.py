import cv2
import numpy as np
import itertools
from scipy.cluster.hierarchy import fclusterdata
from scipy.spatial import ConvexHull

LOWER = np.array([110] * 3)
UPPER = np.array([255] * 3)

MAX_AREA_THRESHOLD = 80000 # TODO: fine-tune based on resolution and distance from camera

POINT_TOLERANCE_PX = 10 # tolerance for line intersection
GROUP_THRESH_PX = 15 # distance threshold for merging points

class CourtDetector:
    def process(self, frame, frame_id, context):
        if frame_id != 0 and context.get("court_points") is not None: return {}

        court = self.detect_court(frame)
        return {"court_points": court}

    def detect_court(self, frame):
        mask = cv2.inRange(frame, LOWER, UPPER)
        
        # filter background noise
        num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask)
        clean_mask = np.zeros_like(mask)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]

            if area < MAX_AREA_THRESHOLD:
                clean_mask[labels == i] = 255
        mask = clean_mask

        edges = cv2.Canny(mask, 50, 150)

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=50,
            minLineLength=50,
            maxLineGap=10
        )

        if lines is None: return None

        # find intersections
        lines = [line[0] for line in lines]
        num_lines = len(lines)

        points = []

        for i in range(num_lines):
            for j in range(i + 1, num_lines):
                point = self.find_segment_intersection(lines[i], lines[j])
                if point is not None:
                    points.append(point)

        if len(points) < 4:
            return None

        # merge close points
        merged_points = []
        labels = fclusterdata(points, t=GROUP_THRESH_PX, criterion='distance', method='single')

        grouped_points = {}
        for point, label in zip(points, labels):
            grouped_points.setdefault(label, []).append(point)

        for cluster in grouped_points.values():
            avg_x = round(np.mean([p[0] for p in cluster]))
            avg_y = round(np.mean([p[1] for p in cluster]))
            merged_points.append((avg_x, avg_y))

        corners = self.find_court_corners(merged_points)
        
        if corners is None:
            return None
        
        # TODO: determine other line intersections from predefined court shape

        return corners

    def find_segment_intersection(self, line1, line2):
        x1, y1, x2, y2 = line1[0]
        x3, y3, x4, y4 = line2[0]

        m1 = (y1 - y2) / (x1 - x2) if x1 != x2 else None
        m2 = (y3 - y4) / (x3 - x4) if x3 != x4 else None

        if m1 is None and m2 is None:
            return None
        
        # one is parallel
        if m1 is None:
            x_int = x1
            y_int = m2 * (x_int - x3) + y3
        elif m2 is None:
            x_int = x3
            y_int = m1 * (x_int - x1) + y1
        else:
            if m1 == m2:
                return None

            x_int = x1 * m1 - y1 - x3 * m2 + y3
            x_int /= m1 - m2

            y_int = m1 * (x_int - x1) + y1

        lx1 = min(x1, x2)
        gx1 = max(x1, x2)
        lx2 = min(x3, x4)
        gx2 = max(x3, x4)

        if x1 == x2:  # line1 vertical
            ly1, gy1 = min(y1, y2), max(y1, y2)
            within_line1 = ly1 <= y_int <= gy1
        else:
            within_line1 = lx1 <= x_int <= gx1
        
        if x3 == x4:  # line2 vertical
            ly2, gy2 = min(y3, y4), max(y3, y4)
            within_line2 = ly2 <= y_int <= gy2
        else:
            within_line2 = lx2 <= x_int <= gx2

        # within segment
        if within_line1 and within_line2:
            return (round(x_int), round(y_int))
    
        close_to_line1 = self.dist((x_int, y_int), (x1, y1)) < POINT_TOLERANCE_PX or self.dist((x_int, y_int), (x2, y2)) < POINT_TOLERANCE_PX or within_line1
        close_to_line2 = self.dist((x_int, y_int), (x3, y3)) < POINT_TOLERANCE_PX or self.dist((x_int, y_int), (x4, y4)) < POINT_TOLERANCE_PX or within_line2

        if close_to_line1 and close_to_line2:
            return (round(x_int), round(y_int))

        return None

    # distance between two points
    def dist(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    
    # finds the court's courners
    def find_court_corners(self, points):
        if len(points) < 4:
            return None
    
        try:
            hull = ConvexHull(points)
            hull_points = [tuple(points[i]) for i in hull.vertices]
        except Exception as e:
            return None

        if len(hull_points) < 4:
            hull_points = points
        
        best_area = -1
        best_quad = None

        for combo in itertools.combinations(hull_points, 4):
            area = self.quad_area(combo)
            if area > best_area:
                best_area = area
                best_quad = combo
        
        return best_quad

    def quad_area(self, quad):
        center_x = sum(p[0] for p in quad) / 4
        center_y = sum(p[1] for p in quad) / 4
        sorted_pts = sorted(quad, key=lambda p: np.arctan2(p[1] - center_y, p[0] - center_x))

        # shoelace formula
        x1 = sorted_pts[0][0]
        y1 = sorted_pts[0][1]
        x2 = sorted_pts[1][0]
        y2 = sorted_pts[1][1]
        x3 = sorted_pts[2][0]
        y3 = sorted_pts[2][1]
        x4 = sorted_pts[3][0]
        y4 = sorted_pts[3][1]

        return 0.5 * abs((x1 * y2 + x2 * y3 + x3 * y4 + x4 * y1)
                         - (y1 * x2 + y2 * x3 + y3 * x4 + y4 * x1))