import cv2
import numpy as np

from .sam_loader import load_sam_model, run_sam_segmentation
from .geometry import get_convex_hull_mask
from .line_detection import get_corners, get_lines, split_by_orientation, get_corners

LOWER = np.array([110] * 3)
UPPER = np.array([255] * 3)

predictor = load_sam_model()

class CourtDetector:
    def process(self, frames, frame_id, context):
        if frame_id != 0 and context.get("court_points") is not None: return {}

        court = self.detect_court(frames[-1])
        return {"court_points": court}

    def detect_court(self, frame):
        image_np = np.array(frame)

        # SAM background removal
        raw_mask = run_sam_segmentation(predictor, frame)
        mask = get_convex_hull_mask(raw_mask)

        isolated_court = cv2.bitwise_and(image_np, image_np, mask=mask)
        raw_line_mask = cv2.inRange(isolated_court, LOWER, UPPER)

        line_mask = self.filter_mask(raw_line_mask)
        lines = get_lines(line_mask)

        width, height = image_np.shape[:2]
        raw_vert, raw_horiz = split_by_orientation(lines, width, height)

        corners = get_corners(raw_vert, raw_horiz)

        return corners

    # remove small or non-line noise
    def filter_mask(self, raw_mask):
        mask = np.zeros_like(raw_mask)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(raw_mask, connectivity=8)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]

            max_dim = max(w, h)
            min_dim = min(w, h) if min(w, h) > 0 else 1

            aspect_ratio = max_dim / min_dim

            if area > 200 or area >= 15 and aspect_ratio >= 2.5:
                mask = np.where(labels == i, 255, mask)