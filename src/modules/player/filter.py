import math
import numpy as np
from utils.consts import COURT_LENGTH, COURT_WIDTH

W_PROXIMITY = 0.45
W_PERSIST = 0.3
W_MOVE = 0.25

PLAY_MARGIN_X_M = 1.5
PLAY_MARGIN_Y_M = 3.0
COURT_DECAY_M = 1.5 # proximity = exp(-d / COURT_DECAY_M), where d is the distance from play region

MOVE_REF_MPS = 1.0
SMOOTH_WINDOW = 5

MIN_PERSISTENCE = 0.1
MAX_INTERP_GAP = 30 # interpolate <= 30 frame gaps

class PlayerFilter:
    window_size = 1

    def __init__(self, player_map=None, fps=30):
        self.player_map = player_map or {}
        self.fps = fps
        self.results = None

    def process(self, _frames, frame_id, context):
        if self.results is None:
            self.results = self._compute(context)

        return {"players": self.results.get(frame_id, {"near": None, "far": None})}

    def _compute(self, context):
        n_frames = len(context)

        tracks = self._collect_tracks(context)
        track_scores = self._score_tracks(tracks, n_frames)
        eligible = self._eligible_tracks(tracks, track_scores)

        return self._select_per_frame(eligible, tracks, track_scores, n_frames)

    def _eligible_tracks(self, tracks, track_scores):
        return [
            track_id for track_id, detections in tracks.items()
            if len(detections) >= 2 and track_scores[track_id]["persist"] >= MIN_PERSISTENCE
        ]

    def _collect_tracks(self, context):
        tracks = {}

        for frame_context in context:
            for player in frame_context.get("all_players", []):
                track_id = player.get('id')
                if track_id is None or player.get('feet_m') is None: continue

                tracks[track_id] = tracks.get(track_id, []) + [player]

        return tracks

    def _get_court_proximity(self, feet_m):
        x, y = float(feet_m[0]), float(feet_m[1])

        dist_x = max(0, -PLAY_MARGIN_X_M - x, x - (COURT_WIDTH + PLAY_MARGIN_X_M))
        dist_y = max(0, -PLAY_MARGIN_Y_M - y, y - (COURT_LENGTH + PLAY_MARGIN_Y_M))

        return math.exp(-math.hypot(dist_x, dist_y) / COURT_DECAY_M)

    def _smooth(self, values):
        if len(values) < SMOOTH_WINDOW:
            return values

        padded = np.pad(values, SMOOTH_WINDOW // 2, mode="edge")
        return np.median(np.lib.stride_tricks.sliding_window_view(padded, SMOOTH_WINDOW), axis=1)

    def _get_movement(self, detections):
        if len(detections) < 2:
            return 0

        frames = np.array([d["frame_id"] for d in detections], dtype=float)
        xy = np.array([d["feet_m"] for d in detections], dtype=float)
        x, y = self._smooth(xy[:, 0]), self._smooth(xy[:, 1])

        path_m = float(np.hypot(np.diff(x), np.diff(y)).sum())
        seconds = (frames[-1] - frames[0]) / self.fps
        if seconds <= 0:
            return 0

        return 1 - math.exp(-(path_m / seconds) / MOVE_REF_MPS)

    def _score_track(self, detections, n_frames):
        proximity = float(np.mean([self._get_court_proximity(d["feet_m"]) for d in detections]))
        persist = min(1.0, len(detections) / n_frames) if n_frames > 0 else 0.0
        move = self._get_movement(detections)

        return {
            "side": self._get_side(detections),
            "proximity": proximity,
            "persist": persist,
            "move": move,
            "score": W_PROXIMITY * proximity + W_PERSIST * persist + W_MOVE * move,
        }

    def _get_side(self, detections):
        median_y = float(np.median([d["feet_m"][1] for d in detections]))
        return "near" if median_y > COURT_LENGTH / 2 else "far"

    def _score_tracks(self, tracks, n_frames):
        return {track_id: self._score_track(detections, n_frames) for track_id, detections in tracks.items()}

    def _select_per_frame(self, eligible, tracks, track_scores, n_frames):
        results = {frame_id: {"near": None, "far": None} for frame_id in range(n_frames)}
        best_score = {frame_id: {"near": -1.0, "far": -1.0} for frame_id in range(n_frames)}

        for track_id in eligible:
            side = track_scores[track_id]["side"]
            score = track_scores[track_id]["score"]

            for frame_id, player in self._fill_gaps(tracks[track_id]).items():
                if frame_id not in results or score <= best_score[frame_id][side]:
                    continue

                best_score[frame_id][side] = score
                results[frame_id][side] = self._build_player(side, track_id, player)

        return results

    @staticmethod
    def _lerp(a, b, t):
        return [(1 - t) * float(u) + t * float(v) for u, v in zip(a, b)]

    def _fill_gaps(self, detections):
        by_frame = {d["frame_id"]: d for d in detections}
        filled = {
            frame_id: {
                "bbox": d["bbox"],
                "feet_px": d["feet_px"],
                "feet_m": d["feet_m"],
                "interpolated": False,
            }
            for frame_id, d in by_frame.items()
        }

        frames = sorted(by_frame)
        for start, end in zip(frames, frames[1:]):
            gap = end - start - 1
            if gap < 1 or gap > MAX_INTERP_GAP:
                continue

            a, b = by_frame[start], by_frame[end]
            for frame_id in range(start + 1, end):
                t = (frame_id - start) / (end - start)
                filled[frame_id] = {
                    "bbox": [int(round(v)) for v in self._lerp(a["bbox"], b["bbox"], t)],
                    "feet_px": [int(round(v)) for v in self._lerp(a["feet_px"], b["feet_px"], t)],
                    "feet_m": tuple(self._lerp(a["feet_m"], b["feet_m"], t)),
                    "interpolated": True,
                }

        return filled

    def _build_player(self, role, track_id, player):
        x_m, y_m = (float(v) for v in player["feet_m"])

        return {
            "id": self.player_map.get(role, {}).get("id"),
            "track_id": track_id,
            "bbox": [int(v) for v in player["bbox"]],
            "feet_px": [int(v) for v in player["feet_px"]],
            "feet_m": (x_m, y_m),
            "feet_pct": (x_m / COURT_WIDTH * 100, y_m / COURT_LENGTH * 100),
            "interpolated": player["interpolated"],
        }