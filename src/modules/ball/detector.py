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

BATCH_SIZE = 64

class BallDetector:
    window_size = 3

    def __init__(self, processor, fps):
        self.processor = processor
        self.fps = fps
        self.batch_inputs = []
        self.batch_frame_ids = []

    def image_to_court(self, x, y, H):
        if x is None or y is None or H is None:
            return None, None

        pt = np.array([[[x, y]]], dtype=np.float32)

        court_pt = cv2.perspectiveTransform(pt, H)[0, 0]

        return float(court_pt[0]), float(court_pt[1])

    def process(self, frames, frame_id, context):
        if frame_id >= 2:
            self._queue_window(frames, frame_id)

            is_last = frame_id + 1 == self.processor.total_frames
            if len(self.batch_inputs) >= BATCH_SIZE or is_last:
                self._detect_ball_batched(context)

        return {
            "ball": {
                'is_missing': True,
                'x': None,
                'y': None,
                'x_px': None,
                'y_px': None
            }
        }

    def _queue_window(self, frames, frame_id):
        orig_h, orig_w, _ = frames[0].shape
        W, H = 640, 360

        img = cv2.resize(frames[2], (W, H))
        img_prev = cv2.resize(frames[1], (W, H))
        img_preprev = cv2.resize(frames[0], (W, H))

        x = np.concatenate((img_preprev, img_prev, img), axis=2)
        x = x.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))

        self.batch_inputs.append(x)
        self.batch_frame_ids.append((frame_id, orig_w, orig_h))

    def _detect_ball_batched(self, context):
        if not self.batch_inputs:
            return

        W, H = 640, 360
        batch = np.stack(self.batch_inputs, axis=0)

        with torch.inference_mode():
            output = trackernet_model(torch.from_numpy(batch).to(DEVICE))

        heatmaps = output.argmax(dim=1).cpu().numpy()

        for heatmap, (frame_id, orig_w, orig_h) in zip(heatmaps, self.batch_frame_ids):
            x_pred, y_pred = postprocess(heatmap)

            if x_pred is not None and y_pred is not None:
                x_pred = x_pred * (orig_w / W)
                y_pred = y_pred * (orig_h / H)

            court_x, court_y = None, None
            court_H = context[frame_id].get("court", {}).get('H')
            if x_pred is not None and y_pred is not None:
                court_x, court_y = self.image_to_court(x_pred, y_pred, court_H)

            self.processor.set_context(frame_id, {
                **context[frame_id],
                "ball": {
                    'is_missing': court_x is None or court_y is None,
                    'x': court_x,
                    'y': court_y,
                    'x_px': x_pred,
                    'y_px': y_pred
                }
            })

        self.batch_inputs.clear()
        self.batch_frame_ids.clear()