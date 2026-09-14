import cv2
from modules.court.detector import CourtDetector
from modules.player.detector import PlayerDetector
from modules.ball.detector import BallDetector
from modules.ball.kinematics import BallKinematics
from modules.events.detector import EventDetector
from modules.metrics.estimator import ShotMetricEstimator
from persister import Persister
from db.repository import MatchRepository
from annotator import VideoAnnotator
import json
import time

class VideoProcessor:
    def __init__(self, far_player, near_player, fps=30):
        self.context = []
        self.fps = fps

        self.player_map = {
            'near': near_player,
            'far': far_player
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
    
    def process(self, video_path, context_path=None, start_pass=0, persist=True):
        print("Processing video...")
        self.module_timing = {}
        
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
        if persist: self._persist_stats()

    def _persist_stats(self):
        persisted = Persister(self.player_map).persist(self.context)
        MatchRepository().save(persisted)

    def _run_annotation(self, video_path):
        annotator = VideoAnnotator(self.context, self.fps)
        annotator.render(video_path)

    def _run_perception(self, frames, frame_id, modules):
        if frame_id >= len(self.context):
            self.context.append({})

        for module in modules:
            module_name = module.__class__.__name__
            started_at = time.perf_counter()
            result = module.process(frames, frame_id, self.context)
            elapsed = time.perf_counter() - started_at
            self.context[frame_id].update(result)

            timing = self.module_timing.setdefault(
                module_name,
                {'total_seconds': 0.0, 'frames': 0},
            )
            timing['total_seconds'] += elapsed
            timing['frames'] += 1

        if (frame_id + 1) % 10 == 0:
            averages = ', '.join(
                f"{module_name}: {timing['total_seconds'] / timing['frames']:.4f}s"
                for module_name, timing in self.module_timing.items()
            )
            print(f"Running module averages after {frame_id + 1} frames: {averages}")

        # return self.context[frame_id]
    
    def set_context(self, frame_id, context):
        if 0 <= frame_id < len(self.context): self.context[frame_id] = context