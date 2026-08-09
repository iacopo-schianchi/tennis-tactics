import cv2
from datetime import datetime
from modules.court.annotator import CourtAnnotator

class VideoAnnotator:
    def __init__(self, context):
        self.context = context
        self.annotators = [
            CourtAnnotator()
        ]

        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d-%H-%M-%S")

        self.OUTPUT_PATH = f'../assets/Demos/{date_str}.mp4'

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