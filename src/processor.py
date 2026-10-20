import cv2
import numpy as np
from modules.court.detector import CourtDetector
from modules.player.detector import PlayerDetector
from modules.player.filter import PlayerFilter
from modules.ball.detector import BallDetector
from modules.ball.kinematics import BallKinematics
from modules.events.detector import EventDetector
from modules.metrics.estimator import ShotMetricEstimator
from persister import Persister
from db.repository import ClipRepository
from annotator import VideoAnnotator
import json
import time
import math

from collections import deque
try:
    from IPython.display import display
except ImportError:
    display = None

class RollingLog:
    def __init__(self, max_lines=4):
        self.lines = deque(maxlen=max_lines)
        self.handle = None

    def log(self, text):
        if display is None:
            print(text)
            return
        self.lines.append(text)
        body = "\n".join(self.lines)
        if self.handle is None:
            self.handle = display(body, display_id=True)
        else:
            self.handle.update(body)

class VideoProcessor:
    TRACKNET_ASPECT_RATIO = 640 / 360
    COURT_PADDING_RATIO = 0.15

    # ball_only: temporarily only run ball detection algorithm
    def __init__(self, far_player, near_player, target_fps=30, broadcast_scoring=False, ball_only=False):
        self.context = []
        self.target_fps = target_fps
        self.broadcast_scoring = broadcast_scoring

        self.player_map = {
            'near': near_player,
            'far': far_player
        }
        self.crop_bounds = None
        self.ball_only = ball_only

    def _build_passes(self):
        # each pass processes frames -> updates context
        if self.ball_only:
            self.passes = [
                (BallDetector(self, self.fps),),
                (BallKinematics(self, self.fps),),
                (EventDetector(),)
            ]
            return

        self.passes = [
            (
                PlayerDetector(self.player_map),
                BallDetector(self, self.fps),
            ),
            (
                PlayerFilter(self.player_map, self.fps),
            ),
            (
                BallKinematics(self, self.fps),
            ),
            (
                EventDetector(),
            ),
            (
                # shot classification & peak/speed estimation
                ShotMetricEstimator(self),
            ),
        ]

    def _configure_sampling(self, video_path):
        cap = cv2.VideoCapture(video_path)
        self.source_fps = cap.get(cv2.CAP_PROP_FPS)
        self.source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        exact_ratio = self.source_fps / self.target_fps
        stride_options = {max(1, math.floor(exact_ratio)), max(1, math.ceil(exact_ratio))}

        best_stride = min(
            stride_options,
            key=lambda stride: abs(self.source_fps / stride - self.target_fps),
        )

        self.stride = best_stride
        self.fps = self.source_fps / best_stride
        self.total_frames = math.ceil(self.source_frames / self.stride)

    def _detect_court(self, video_path):
        detector = CourtDetector()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Could not open video for court detection: {video_path}")

        raw_idx = 0
        court = None
        try:
            while cap.grab():
                if raw_idx % self.stride == 0:
                    ret, frame = cap.retrieve()
                    if not ret:
                        break
                    points, homography = detector.detect_court(frame, debug=True)
                    if points is not None and homography is not None:
                        court = {"points": points, "H": homography}
                        break
                raw_idx += 1
        finally:
            cap.release()

        if court is None:
            raise RuntimeError("Court detection failed on every sampled frame")
        return court

    def _find_existing_court(self):
        for frame_context in self.context:
            court = frame_context.get("court")
            if court and court.get("H") is not None:
                return court
        return None

    def _get_crop_bounds(self, court, frame_shape):
        frame_h, frame_w = frame_shape[:2]
        points = np.asarray(court["points"], dtype=float)
        if points.shape != (4, 2) or not np.isfinite(points).all():
            raise ValueError("Court points must contain four finite image coordinates")
        left, top = points.min(axis=0)
        right, bottom = points.max(axis=0)
        court_width = right - left
        court_height = bottom - top

        padding_x = court_width * self.COURT_PADDING_RATIO
        padding_y = court_height * self.COURT_PADDING_RATIO

        required_width = max(
            court_width + 2 * padding_x,
            (court_height + 2 * padding_y) * self.TRACKNET_ASPECT_RATIO,
        )
        max_crop_units = min(frame_w // 16, frame_h // 9)
        
        required_crop_units = math.ceil(required_width / 16)
        crop_units = min(required_crop_units, max_crop_units)
        crop_width = crop_units * 16
        crop_height = crop_units * 9

        center_x = (left + right) / 2
        x0 = int(round(center_x - crop_width / 2))
        x0 = min(max(x0, 0), frame_w - crop_width)

        desired_bottom = min(frame_h, bottom + padding_y)
        y0 = int(round(desired_bottom - crop_height))
        y0 = min(max(y0, 0), frame_h - crop_height)

        return x0, y0, x0 + crop_width, y0 + crop_height

    def _set_court_homography_for_crop(self, court, x_offset, y_offset):
        translation = np.array(
            [
                [1.0, 0.0, x_offset],
                [0.0, 1.0, y_offset],
                [0.0, 0.0, 1.0],
            ]
        )
        return {**court, "H": np.asarray(court["H"]) @ translation}

    def _restore_original_coordinates(self):
        x_offset, y_offset = self.crop_bounds[:2]
        for frame_context in self.context:
            for player in frame_context.get("all_players", []):
                player["bbox"] = [
                    player["bbox"][0] + x_offset,
                    player["bbox"][1] + y_offset,
                    player["bbox"][2] + x_offset,
                    player["bbox"][3] + y_offset,
                ]
                player["feet_px"] = [
                    player["feet_px"][0] + x_offset,
                    player["feet_px"][1] + y_offset,
                ]

            ball = frame_context.get("ball")
            if ball:
                if ball.get("x_px") is not None:
                    ball["x_px"] += x_offset
                if ball.get("y_px") is not None:
                    ball["y_px"] += y_offset

            if "ball_candidates" in frame_context:
                frame_context["ball_candidates"] = [
                    (x + x_offset, y + y_offset, confidence)
                    for x, y, confidence in frame_context["ball_candidates"]
                ]

    def process(self, video_path, context_path=None, start_pass=0, persist=True):
        print(persist)
        print("Processing video...")
        self.module_timing = {}
        
        self._configure_sampling(video_path)
        self._build_passes()
        
        if context_path:
            with open(context_path) as f:
                self.context = json.load(f)
            assert len(self.context) == self.total_frames, f"Loaded context length doesn't match video frame count ({self.total_frames} frames, {len(self.context)} context)"

        if not self.context:
            self.context = [{} for _ in range(self.total_frames)]

        court = self._find_existing_court() or self._detect_court(video_path)
        for frame_context in self.context:
            frame_context["court"] = court

        first_frame = cv2.VideoCapture(video_path)
        ret, sample_frame = first_frame.read()
        first_frame.release()
        if not ret:
            raise RuntimeError(f"Could not read a frame from video: {video_path}")
        self.crop_bounds = self._get_crop_bounds(court, sample_frame.shape)

        pass_start_idx = len(self.passes) if start_pass == 'full' else start_pass
        if pass_start_idx == 0:
            cropped_court = self._set_court_homography_for_crop(
                court, self.crop_bounds[0], self.crop_bounds[1]
            )
            for frame_context in self.context:
                frame_context["court"] = cropped_court

        for i, modules in enumerate(self.passes):
            if i < pass_start_idx: continue
            print(f"Pass {i+1}")

            cap = cv2.VideoCapture(video_path)
            print("Video path:", video_path)
            print("Capture opened:", cap.isOpened())
            print("Frame count:", cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print("Source FPS:", cap.get(cv2.CAP_PROP_FPS))
            print("Target FPS:", self.target_fps)

            self.rolling = RollingLog(max_lines=4)

            frames = []

            window_size = max(
                module.window_size if hasattr(module, "window_size") else 1 for module in modules
            )

            # TODO: only read frames for passes that need it

            raw_idx = 0
            frame_id = 0
            while cap.isOpened():
                if not cap.grab():
                    break
                if raw_idx % self.stride == 0:
                    ret, frame = cap.retrieve()
                    if not ret:
                        break
                    x0, y0, x1, y1 = self.crop_bounds
                    frame = frame[y0:y1, x0:x1]
                    frames.append(frame)
                    if len(frames) > window_size:
                        frames.pop(0)
                    self._run_perception(frames, frame_id, modules)
                    
                    frame_id += 1
                raw_idx += 1
        
            cap.release()
            if i == 0:
                self._restore_original_coordinates()
                for frame_context in self.context:
                    frame_context["court"] = court

        self._run_annotation(video_path)
        if persist: self._persist_stats()

    def _persist_stats(self):
        persisted = Persister(self.player_map, broadcast_scoring=self.broadcast_scoring).persist(self.context)
        ClipRepository().save(persisted)

    def _run_annotation(self, video_path):
        annotator = VideoAnnotator(self.context, self.fps, self.stride, ball_only=self.ball_only)
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
            self.rolling.log(f"Processed {frame_id}/{self.total_frames} frames. Averages: {averages}")

        # return self.context[frame_id]
    
    def set_context(self, frame_id, context):
        if 0 <= frame_id < len(self.context): self.context[frame_id] = context