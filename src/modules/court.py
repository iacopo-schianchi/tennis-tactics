import cv2
import numpy as np

LOWER = np.array([110] * 3)
UPPER = np.array([255] * 3)

MAX_AREA_THRESHOLD = 80000 # TODO: fine-tune based on resolution and distance from camera

class CourtDetector:
    def process(self, frame, frame_id):
        if frame_id != 0: return {}

        court = self.detect_lines(frame);
        return {"court_points": court}

    def detect_court(self, frame):
        mask = cv2.inRange(frame, LOWER, UPPER)

        pass