import cv2
import numpy as np
from ultralytics import YOLO
from ...consts.consts import COURT_WIDTH, COURT_LENGTH

yolo = YOLO("yolov8n.pt")

class PlayerDetector:
    def __init__(self, player_map):
        self.player_map = player_map

    def process(self, frames, frame_id, context):
        raw_bboxes = self._detect_players(frames[-1])

        court = context[frame_id].get("court")
        players = self._classify_players(raw_bboxes, court, frames[-1].shape[0])

        return {"players": players}

    def _detect_players(self, frame, conf=0.35):
        results = yolo.predict(
            frame,
            verbose=False,
            conf=conf
        )

        player_boxes = []

        for r in results:
            for box in r.boxes:
                if int(box.cls) == 0: # person
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    player_boxes.append((x1, y1, x2, y2))

        # TODO: keep only 2 closest players to the center of the court to filter background ppl.

        return player_boxes

    def _dist(self, p1, p2):
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def _pixel_to_court(self, x, y, H):
        src_pt = np.array([[[x, y]]], dtype=np.float32)
        x_bev, y_bev = cv2.perspectiveTransform(src_pt, H)

        x_pct = (x_bev / COURT_WIDTH) * 100.0
        y_pct = (y_bev / COURT_LENGTH) * 100.0

        return (int(x_pct), int(y_pct)), (int(x_bev), int(y_bev))

    def _build_player_dict(self, bbox, court, role):
        x1, y1, x2, y2 = bbox
        feet_x_px = (x1 + x2) / 2.0
        feet_y_px = y2

        # pixel to percent
        pct = (None, None)
        H = court['H']
        if H:
            pct, bev = self._pixel_to_court_pct(feet_x_px, feet_y_px, H)

        return {
            "id": self.player_map.get(role),
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "feet_px": [int(feet_x_px), int(feet_y_px)], # coords in image pixel coordinates
            "feet_m": bev, # coords in meters on court suface
            "feet_pct": pct, # coords as court %
        }
    
    def _classify_players(self, bboxes, court, frame_height):
        if not bboxes or len(bboxes) == 0:
            return {'near': None, 'far': None}

        tl, tr, br, bl = court['points']

        cx = (tl[0] + br[0] + tr[0] + bl[0]) / 4
        cy = (tl[1] + br[1] + tr[1] + bl[1]) / 4
        center = (int(cx), int(cy))

        # TODO: filter players that move < MOVE_THRESHOLD per frame throughout clip if len(bboxes) > 2

        # keep 2 nearest to center
        players = sorted(bboxes, key=lambda b: self._dist(((b[0] + b[2]) // 2, (b[1] + b[3]) // 2), center))[:min(2, len(bboxes))]

        sorted_players = sorted(players, key=lambda p: p[3])

        if len(players) == 2:
            far, near = sorted_players
            return {'near': self._build_player_dict(near, court, 'near'), 'far': self._build_player_dict(far, court, 'far')}
        else:
            feet_y = players[0][3]
            out = {}

            role = "near" if feet_y > (frame_height / 2) else "far"
            out[role] = self._build_player_dict(players[0], court, role)
            return out

