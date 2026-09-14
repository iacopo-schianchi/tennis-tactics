import cv2
from modules.court.detector import CourtDetector
from modules.player.detector import PlayerDetector
from modules.ball.detector import BallDetector
from modules.ball.kinematics import BallKinematics
from modules.events.detector import EventDetector
from modules.metrics.estimator import ShotMetricEstimator
from annotator import VideoAnnotator
import json

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
                BallKinematics(self, fps),
            ),
            (
                EventDetector(),
            ),
            (
                # shot classification & peak/speed estimation
                ShotMetricEstimator(self),
            ),
        ]
    
    def process(self, video_path, context_path=None, start_pass=0):
        print("Processing video...")
        
        cap = cv2.VideoCapture(video_path)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        if context_path:
            with open(context_path) as f:
                self.context = json.load(f)
            assert len(self.context) == self.total_frames, f"Loaded context length doesn't match video frame count ({self.total_frames} frames, {len(self.context)} context)"

        pass_start_idx = len(self.passes) if start_pass == 'full' else start_pass

        for i, modules in enumerate(self.passes):
            if i < pass_start_idx: continue
            print(f"Pass {i+1}")

            cap = cv2.VideoCapture(video_path)
            print("Video path:", video_path)
            print("Capture opened:", cap.isOpened())
            print("Frame count:", cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print("FPS:", cap.get(cv2.CAP_PROP_FPS))

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
                
                self._run_perception(frames, frame_id, modules)

                if frame_id % 10 == 0:
                    print(f"Processed {frame_id}/{self.total_frames} frames")
                
                frame_id += 1
        
            cap.release()

        self._run_annotation(video_path)

    def _run_annotation(self, video_path):
        annotator = VideoAnnotator(self.context, self.fps)
        annotator.render(video_path)

    def _run_perception(self, frames, frame_id, modules):
        if frame_id >= len(self.context):
            self.context.append({})

        for module in modules:
            result = module.process(frames, frame_id, self.context)
            self.context[frame_id].update(result)

        # return self.context[frame_id]
    
    def set_context(self, frame_id, context):
        if 0 <= frame_id < len(self.context): self.context[frame_id] = context