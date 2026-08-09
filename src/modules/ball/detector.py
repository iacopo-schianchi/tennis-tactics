import numpy as np
import os
import torch
import cv2
from src.models.tracknet.model import BallTrackerNet
from src.models.tracknet.postprocess import postprocess

MODEL_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'tracknet.pt')

trackernet_model = BallTrackerNet()
trackernet_model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH))
trackernet_model.eval()

class BallDetector:
    window_size = 3

    def __init__(self, processor, fps):
        self.processor = processor
        self.fps = fps

    def wrap_angle(self, a):
        return ((a + np.pi) % (2 * np.pi)) - np.pi

    def image_to_court(self, x, y, H):
        if x is None or y is None or H is None:
            return None, None

        pt = np.array([[x, y]], dtype=np.float32)

        court_pt = cv2.perspectiveTransform(pt, H)[0, 0]

        return float(court_pt[0]), float(court_pt[1])

    def set_last_accel(self, frame_id, context, vx, vy):
        # last frame
        if frame_id + 1 == self.processor.total_frames:
            prev = context[frame_id - 1]['ball']
            if prev['vx'] is not None and prev['vy'] is not None:
                ax = (vx - prev['vx']) * self.fps
                ay = (vy - prev['vy']) * self.fps
            else:
                ax, ay = 0.0, 0.0
        else:
            prev2 = context[frame_id - 2]['ball']
            if prev2['vx'] is not None and prev2['vy'] is not None:
                ax = (vx - prev2['vx']) * self.fps / 2
                ay = (vy - prev2['vy']) * self.fps / 2
            else:
                ax, ay = 0.0, 0.0

        accel = np.sqrt(ax ** 2 + ay ** 2)
        self.processor.set_context(frame_id - 1, {
            **context[frame_id - 1],
            "ball": {
                **context[frame_id - 1]['ball'],
                "ax": ax,
                "ay": ay,
                "accel": accel
            }
        })

    def process(self, frames, frame_id, context):
        x_px, y_px = None, None
        if frame_id >= 2:
            x_px, y_px = self.detect_ball(frames)

        x, y = None, None
        if x_px is not None and y_px is not None:
            H = context[frame_id].get("court").get('H')
            x, y = self.image_to_court(x_px, y_px, H)

        vx, vy = 0, 0 # TODO: change to centered derivative instead
        if frame_id > 2:
            prev = context[frame_id - 1]['ball']
            if x is not None and prev['x'] is not None:
                vx = (x - prev['x']) * self.fps
                vy = (y - prev['y']) * self.fps

        if frame_id >= 4:
            self.set_last_accel(frame_id, context, vx, vy)

        speed = np.sqrt(vx ** 2 + vy ** 2)

        if frame_id <= 2:
            angle = 0.0
        elif speed < 0.1:
            angle = context[frame_id-1]["angle"]
        else:
            angle = np.arctan2(vy, vx)

        if frame_id <= 2:
            delta_angle = 0.0
        else:
            raw_delta = angle - context[frame_id - 1]['angle']
            delta_angle = self.wrap_angle(raw_delta) * self.fps

        return {
            "ball": {
                'is_missing': x is None or y is None,
                'x': x,
                'y': y,
                'x_px': x_px,
                'y_px': y_px,
                'vx': vx,
                'vy': vy,
                'angle': angle,
                'delta_angle': delta_angle,
                'speed': speed,

                # set in next frame
                'ax': 0.0,
                'ay': 0.0,
                'accel': 0.0
            }
        }

    def detect_ball(self, frames):
        orig_h, orig_w, _ = frames[0].shape
        H, W = 360, 640

        img = cv2.resize(frames[2], (W, H))
        img_prev = cv2.resize(frames[1], (W, H))
        img_preprev = cv2.resize(frames[0], (W, H))

        x = np.concatenate((img_preprev, img_prev, img), axis=2)
        x = x.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))
        x = np.expand_dims(x, axis=0)

        with torch.no_grad():
            output = trackernet_model(torch.from_numpy(x))

        heatmap = output.argmax(dim=1).squeeze().cpu().numpy()
        x_pred, y_pred = postprocess(heatmap)

        if x_pred is not None and y_pred is not None:
            x_pred = x_pred * (orig_w / W)
            y_pred = y_pred * (orig_h / H)

        return x_pred, y_pred