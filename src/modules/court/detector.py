import cv2
import numpy as np

from .sam_loader import load_sam_model, run_sam_segmentation
from .geometry import get_convex_hull_mask
from .line_detection import get_corners, get_lines, split_by_orientation
from .debug import show_detection_debug
from utils.consts import COURT_LENGTH, COURT_WIDTH
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
predictor = load_sam_model(DEVICE)

class CourtDetector:
    def process(self, frames, frame_id, context):
        if frame_id == 0 or (frame_id != 0 and context[frame_id - 1].get("court").get('H') is None):
            pts, H = self.detect_court(frames[0], True)
        else:
            last_court = context[frame_id - 1].get("court")

            if last_court is not None:
                pts = last_court["points"]
                H = last_court["H"]
            else:
                pts, H = self.detect_court(frames[-1], True)

        return {
            "court": {
                "points": pts,
                "H": H
            }
        }

    def get_line_mask(self, img, mask):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        inside = mask > 0

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k)

        t, _ = cv2.threshold(tophat[inside].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        t = max(t, 15)

        floor = np.median(gray[inside]) + 20
        out = (tophat > t) & (hsv[..., 1] < 70) & (gray > floor) & inside
        return out.astype(np.uint8) * 255

    def detect_court(self, frame, debug = False):
        image_np = np.array(frame)

        # SAM background removal
        raw_mask = run_sam_segmentation(predictor, frame)
        mask = get_convex_hull_mask(raw_mask)

        isolated_court = cv2.bitwise_and(image_np, image_np, mask=mask)
        raw_line_mask = self.get_line_mask(isolated_court, mask)

        line_mask = self.filter_mask(raw_line_mask)
        lines = get_lines(line_mask)

        height, width = image_np.shape[:2]
        raw_vert, raw_horiz = split_by_orientation(lines)

        corners = get_corners(raw_vert, raw_horiz, width, height)
        
        src_pts = np.array(corners, dtype=np.float32)

        dst = np.array(
            [
                [0,0],
                [COURT_WIDTH,0],
                [COURT_WIDTH,COURT_LENGTH],
                [0,COURT_LENGTH]
            ],
            dtype=np.float32
        )

        H = None
        if np.isfinite(src_pts).all() and src_pts.shape == (4, 2):
            H, _ = cv2.findHomography(src_pts, dst)

        if debug:
            show_detection_debug(
                image_np,
                raw_mask,
                mask,
                line_mask,
                lines,
                raw_horiz,
                raw_vert,
                corners
            )

        return corners, H

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

        return mask