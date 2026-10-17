import cv2
import numpy as np
from ultralytics import YOLO
from utils.consts import COURT_WIDTH, COURT_LENGTH

yolo = YOLO("yolov8m.pt") # was yolo8n
tracker = "bytetrack.yaml"

class PlayerDetector:
    def __init__(self, player_map):
        self.player_map = player_map

    def process(self, frames, frame_id, context):
        detections = self._detect_players(frames[-1])

        court = context[frame_id].get("court")
        players = [self._build_player_dict(detection, court, frame_id) for detection in detections]

        return {"all_players": players}

    def _detect_players(self, frame, conf=0.25):
        results = yolo.track(
            frame,
            verbose=False,
            conf=conf,
            tracker=tracker,
            persist=True,
            classes=[0], # person
            imgsz=1280,
        )

        detections = []

        for result in results:
            if result.boxes is None:
                continue

            boxes = result.boxes

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(
                    int, box.xyxy[0].cpu().numpy()
                )

                track_id = None
                if boxes.id is not None:
                    track_id = int(boxes.id[i].item())

                detections.append(
                    ((x1, y1, x2, y2), track_id)
                )

        return detections

    def _pixel_to_court(self, x, y, H):
        src_pt = np.array([[[x, y]]], dtype=np.float32)
        x_bev, y_bev = cv2.perspectiveTransform(src_pt, H)[0, 0]

        x_pct = (x_bev / COURT_WIDTH) * 100.0
        y_pct = (y_bev / COURT_LENGTH) * 100.0

        return (x_pct, y_pct), (x_bev, y_bev)

    def _build_player_dict(self, detection, court, frame_id):
        bbox, track_id = detection
        x1, y1, x2, y2 = bbox
        feet_x_px = (x1 + x2) / 2.0
        feet_y_px = y2

        # pixel to percent
        pct = (None, None)
        H = court.get('H')
        bev = None
        if H is not None:
            pct, bev = self._pixel_to_court(feet_x_px, feet_y_px, H)

        _, y_pct = pct
        role = None
        if y_pct is not None:
            role = "near" if y_pct > 50 else "far"

        return {
            "id": track_id,
            "bbox": bbox,
            "feet_px": [feet_x_px, feet_y_px], # coords in image pixel coordinates
            "feet_m": bev, # coords in meters on court suface
            "feet_pct": pct, # coords as court %
            "role": role,
            "frame_id": frame_id
        }
