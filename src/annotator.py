import cv2
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from modules.court.annotator import CourtAnnotator
from modules.ball.annotator import BallAnnotator
from modules.metrics.annotator import MetricAnnotator
from modules.player.annotator import PlayerAnnotator

def json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.bool_):
        return bool(obj)

    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

class VideoAnnotator:
    def __init__(self, context, fps=30):
        self.context = context
        self.annotators = [
            CourtAnnotator(),
            BallAnnotator(),
            MetricAnnotator(),
            PlayerAnnotator(),
        ]
        self.fps = fps

        OUTPUT_DIR = Path("/content/tennis-tactics/assets/Demos")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d-%H-%M-%S")

        self.OUTPUT_PATH = str(OUTPUT_DIR / f"{date_str}.mp4")

        context_path = str(OUTPUT_DIR / f"{date_str}-context.json")

        try:
            with open(context_path, "w") as f:
                json.dump(context, f, indent=2, default=json_default)
        except Exception as e:
            print("Failed uploading context: ", e)

        print(f"Saving annotated video to: {self.OUTPUT_PATH} and context to: {context_path}")

    def render(self, video_path):
        cap = cv2.VideoCapture(video_path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(self.OUTPUT_PATH, fourcc, self.fps, (w, h))

        frame_id = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            for annotator in self.annotators:
                frame = annotator.draw(frame, frame_id, self.context)

            writer.write(frame)
            frame_id += 1

        cap.release()
        writer.release()