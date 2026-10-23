import numpy as np
import torch
import cv2
from models.tracknet.model import BallTrackerNet
from models.tracknet.postprocess import postprocess
from pathlib import Path

MODEL_WEIGHTS_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "models"
    / "tracknet"
    / "tracknet.pt"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trackernet_model = BallTrackerNet()
trackernet_model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE))
trackernet_model.to(DEVICE)
trackernet_model.eval()

class BallDetector:
    max_candidates_per_frame = 5
    missing_detection_cost = 1.07 # TODO: tune
    confidence_cost_weight = 3.0
    distance_reference_pixels = 40.0

    max_displacement_per_frame_pct = 0.09
    max_missing_gap_frames = 10

    def __init__(self, processor, fps):
        self.processor = processor
        self.fps = fps
        self.frame_width = None

    def image_to_court(self, x, y, H):
        if x is None or y is None or H is None:
            return None, None

        pt = np.array([[[x, y]]], dtype=np.float32)

        court_pt = cv2.perspectiveTransform(pt, H)[0, 0]

        return float(court_pt[0]), float(court_pt[1])

    def _clean_detections(self, context):
        if not context: return

        candidates_by_frame = []

        for frame_context in context:
            frame_candidates = frame_context.get("ball_candidates", [])

            frame_candidates = sorted(frame_candidates, key=lambda candidate: candidate[2], reverse=True)[:self.max_candidates_per_frame]
            candidates_by_frame.append(frame_candidates + [None])

        frame_count = len(candidates_by_frame)

        def detection_cost(candidate):
            if candidate is None:
                return self.missing_detection_cost

            confidence = np.clip(candidate[2], 1e-6, 1.0)

            return self.confidence_cost_weight * -np.log(confidence)

        def distance_cost(previous_candidate, current_candidate, dt=1):
            if previous_candidate is None or current_candidate is None:
                return 0.0

            previous_position = np.asarray(previous_candidate[:2], dtype=np.float64)
            current_position = np.asarray(current_candidate[:2], dtype=np.float64)

            distance_pixels = np.linalg.norm(current_position - previous_position)

            reference_width = getattr(self, "frame_width", None) or 640
            distance_scale = self.distance_reference_pixels * reference_width / 640.0

            return float(distance_pixels / (distance_scale * dt))

        def transition_is_feasible(previous_candidate, current_candidate, dt=1):
            if previous_candidate is None or current_candidate is None:
                return True

            previous_position = np.asarray(previous_candidate[:2], dtype=np.float64)
            current_position = np.asarray(current_candidate[:2], dtype=np.float64)

            distance_pixels = np.linalg.norm(current_position - previous_position)

            reference_width = getattr(self, "frame_width", None) or 640
            max_distance = self.max_displacement_per_frame_pct * reference_width * dt

            return distance_pixels <= max_distance

        observation_costs = [
            [detection_cost(candidate) for candidate in frame_candidates]
            for frame_candidates in candidates_by_frame
        ]

        if frame_count == 1:
            selected_indices = [int(np.argmin(observation_costs[0]))]

        else:
            path_costs = {}
            backpointers_by_frame = [{} for _ in range(frame_count)]

            for candidate_index, candidate in enumerate(candidates_by_frame[0]):
                last_observed = (0, candidate_index) if candidate is not None else None

                state = (candidate_index, last_observed)
                path_costs[state] = observation_costs[0][candidate_index]

            for frame_index in range(1, frame_count):
                next_path_costs = {}
                backpointers = {}

                current_candidates = candidates_by_frame[frame_index]

                for previous_state, previous_path_cost in path_costs.items():
                    previous_index, last_observed = previous_state

                    if last_observed is not None:
                        last_frame, _ = last_observed

                        # max frame gap
                        if frame_index - last_frame > self.max_missing_gap_frames: continue

                    for current_index, current_candidate in enumerate(current_candidates):
                        transition_cost = 0.0

                        if current_candidate is not None and last_observed is not None:
                            last_frame, last_candidate_index = last_observed

                            last_candidate = candidates_by_frame[last_frame][last_candidate_index]

                            dt = frame_index - last_frame

                            if not transition_is_feasible(last_candidate, current_candidate, dt): continue

                            transition_cost = distance_cost(last_candidate, current_candidate, dt)

                        if current_candidate is not None:
                            new_last_observed = (frame_index, current_index)
                        else:
                            new_last_observed = last_observed

                        new_state = (current_index, new_last_observed)

                        total_cost = previous_path_cost + observation_costs[frame_index][current_index] + transition_cost

                        if (new_state not in next_path_costs or total_cost < next_path_costs[new_state]):
                            next_path_costs[new_state] = total_cost
                            backpointers[new_state] = previous_state

                path_costs = next_path_costs
                backpointers_by_frame[frame_index] = backpointers

            selected_indices = [None] * frame_count

            state = min(path_costs, key=path_costs.get)

            for frame_index in range(frame_count - 1, -1, -1):
                selected_indices[frame_index] = state[0]

                if frame_index > 0:
                    state = backpointers_by_frame[frame_index][state]

        for frame_id, (frame_context, frame_candidates, selected_index) in enumerate(zip(context, candidates_by_frame, selected_indices)):
            selected_candidate = frame_candidates[selected_index]

            if selected_candidate is None:
                frame_context["ball"] = {
                    "is_missing": True,
                    "x": None,
                    "y": None,
                    "x_px": None,
                    "y_px": None,
                }
            else:
                x_px, y_px, confidence = selected_candidate

                homography = (frame_context.get("court") or {}).get("H")

                x, y = self.image_to_court(x_px, y_px, homography)

                frame_context["ball"] = {
                    "is_missing": x is None or y is None,
                    "x": x,
                    "y": y,
                    "x_px": x_px,
                    "y_px": y_px,
                    "confidence": confidence,
                }

            self.processor.set_context(frame_id, frame_context)

    def process(self, frames, frame_id, context):
        x_px, y_px, all_candidates = None, None, []
        if frame_id >= 2:
            x_px, y_px, all_candidates = self.detect_ball(frames)

        x, y = None, None
        if x_px is not None and y_px is not None:
            H = context[frame_id].get("court").get('H')
            x, y = self.image_to_court(x_px, y_px, H)

        new_context = {
            "ball": {
                'is_missing': x is None or y is None,
                'x': x,
                'y': y,
                'x_px': x_px,
                'y_px': y_px
            },
            "ball_candidates": all_candidates
        }

        if frame_id == self.processor.total_frames - 1:
            all_context = list(context)

            if frame_id < len(all_context):
                all_context[frame_id] = {**all_context[frame_id], **new_context}
            else:
                all_context.append(new_context)

            self._clean_detections(all_context)

            new_context["ball"] = all_context[frame_id]["ball"]

        return new_context

    def detect_ball(self, frames):
        orig_h, orig_w, _ = frames[0].shape
        self.frame_width = orig_w
        # H, W = 360, 640
        H, W = 544, 960

        img = cv2.resize(frames[2], (W, H), interpolation=cv2.INTER_AREA)
        img_prev = cv2.resize(frames[1], (W, H), interpolation=cv2.INTER_AREA)
        img_preprev = cv2.resize(frames[0], (W, H), interpolation=cv2.INTER_AREA)

        # x = np.concatenate((img_preprev, img_prev, img), axis=2)
        x = np.concatenate((img, img_prev, img_preprev), axis=2)
        x = x.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))
        x = np.expand_dims(x, axis=0)

        with torch.inference_mode():
            probs = trackernet_model(torch.from_numpy(x).to(DEVICE), testing=True)[0]

        ball_prob = probs[128:].sum(dim=0).cpu().numpy()
        candidate_balls = postprocess(ball_prob, W, H)

        all_candidates = [
            (
                float(candidate_x) * orig_w / W,
                float(candidate_y) * orig_h / H,
                float(confidence),
            )
            for candidate_x, candidate_y, confidence in candidate_balls
        ]

        if all_candidates:
            x_pred, y_pred = all_candidates[0][:2]
        else:
            x_pred, y_pred = None, None

        return x_pred, y_pred, all_candidates