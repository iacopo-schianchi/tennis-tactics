import cv2
from modules.court.detector import CourtDetector
from modules.player.detector import PlayerDetector
from modules.ball.detector import BallDetector
from modules.events.detector import EventDetector
from modules.metrics.estimator import ShotMetricEstimator

class VideoProcessor:
    def __init__(self, fps):
        self.context = []
        self.fps = 30

        # each pass processes frames -> updates context
        self.passes = [
            (
                CourtDetector(),
                PlayerDetector(),
                BallDetector(self, fps),
            ),
            (
                EventDetector()
            ),
            (
                # shot classification & peak/speed estimation
                ShotMetricEstimator(self)
            ),
        ]
    
    def process(self, video_path):
        cap = cv2.VideoCapture(video_path)
        
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for i, modules in enumerate(self.passes):
            frames = []
            frame_id = 0

            window_size = max(
                module.window_size if hasattr(module, "window_size") else 1 for module in modules
            )

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
                if len(frames) > window_size:
                    frames.pop(0)
                
                frame_data = self._run_perception(frames, frame_id, modules)
                if i == 0:
                    self.context.append(frame_data)
                else:
                    self.context[frame_id].update(frame_data)

                if frame_id % 100 == 0:
                    print(f"Pass {i} | Processed {frame_id}/{self.total_frames} frames")
                
                frame_id += 1
        
        cap.release()

    def _run_perception(self, frames, frame_id, modules):
        frame_data = {}

        for module in modules:
            frame_data.update(module.process(frames, frame_id, self.context))
        
        return frame_data
    
    def set_context(self, frame_id, context):
        if self.get(frame_id): self.context[frame_id] = context