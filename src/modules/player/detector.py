from ultralytics import YOLO

yolo = YOLO("yolov8n.pt")

class PlayerDetector:
    def process(self, frames, _frame_id, _context):
        return {"players": self.detect_players(frames[-1])}

    def detect_players(frame, conf=0.35):
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

        return player_boxes
