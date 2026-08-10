import cv2
from modules.court.detector import CourtDetector
from modules.player.detector import PlayerDetector
from modules.ball.detector import BallDetector
from modules.events.detector import EventDetector
from modules.metrics.estimator import ShotMetricEstimator
from annotator import VideoAnnotator

class VideoProcessor:
    def __init__(self, far_player_id, near_player_id, fps = 30):
        self.context = []
        self.fps = fps

        self.player_map = {
            'near': near_player_id,
            'far': far_player_id
        }

        # each pass processes frames -> updates context
        self.passes = [
            (
                CourtDetector(),
                PlayerDetector(self.player_map),
                BallDetector(self, fps),
            ),
            (
                EventDetector(),
            ),
            (
                # shot classification & peak/speed estimation
                ShotMetricEstimator(self),
            ),
        ]
    
    def process(self, video_path):
        print("Processing video...")
        
        cap = cv2.VideoCapture(video_path)
        
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for i, modules in enumerate(self.passes):
            print(f"Pass {i+1}")
            frames = []
            frame_id = 0

            window_size = max(
                module.window_size if hasattr(module, "window_size") else 1 for module in modules
            )

            # TODO: only read frames for passes that need it

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

                if frame_id % 10 == 0:
                    print(f"Processed {frame_id}/{self.total_frames} frames")
                
                frame_id += 1
        
        cap.release()

    def _run_annotation(self, video_path):
        annotator = VideoAnnotator(self.context)
        annotator.render(video_path)

    def _run_perception(self, frames, frame_id, modules):
        if frame_id >= len(self.context):
            self.context.append({})

        for module in modules:
            result = module.process(frames, frame_id, self.context)
            self.context[frame_id].update(result)

        return self.context[frame_id]
    
    def set_context(self, frame_id, context):
        if 0 <= frame_id < len(self.context): self.context[frame_id] = context