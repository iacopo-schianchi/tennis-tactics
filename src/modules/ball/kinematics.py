import numpy as np
import pandas as pd

class BallKinematics:
    window_size = 1

    def __init__(self, processor, fps):
        self.processor = processor
        self.fps = fps
        self.results = None

    def wrap_angle(self, a):
        return ((a + np.pi) % (2 * np.pi)) - np.pi

    def process(self, _frames, frame_id, context):
        if self.results is None:
            self.results = self._compute(context)
        return {"ball": self.results[frame_id]}

    def _compute(self, context):
        n = len(context)
        raw = [c['ball'] for c in context]

        x_raw = np.array([b['x'] if b['x'] is not None else np.nan for b in raw])
        y_raw = np.array([b['y'] if b['y'] is not None else np.nan for b in raw])
        is_missing = np.isnan(x_raw)

        # TODO: improve from linear interpolation
        x = pd.Series(x_raw).interpolate(method='linear').bfill().ffill().to_numpy()
        y = pd.Series(y_raw).interpolate(method='linear').bfill().ffill().to_numpy()

        vx = np.zeros(n)
        vy = np.zeros(n)
        angle = np.zeros(n)
        delta_angle = np.zeros(n)
        ax = np.zeros(n)
        ay = np.zeros(n)

        for i in range(n):
            if i <= 2:
                vx[i], vy[i] = 0.0, 0.0
            elif i == n - 1:
                vx[i] = (x[i] - x[i - 1]) * self.fps
                vy[i] = (y[i] - y[i - 1]) * self.fps
            else:
                vx[i] = (x[i + 1] - x[i - 1]) * self.fps / 2
                vy[i] = (y[i + 1] - y[i - 1]) * self.fps / 2

            speed_i = np.hypot(vx[i], vy[i])

            if i <= 2:
                angle[i] = 0.0
            elif speed_i < 0.1:
                angle[i] = angle[i - 1]
            else:
                angle[i] = np.arctan2(vy[i], vx[i])

            if i <= 2:
                delta_angle[i] = 0.0
            else:
                raw_delta = angle[i] - angle[i - 1]
                delta_angle[i] = self.wrap_angle(raw_delta) * self.fps

            if i >= 4:
                ax[i - 1] = (vx[i] - vx[i - 2]) * self.fps / 2
                ay[i - 1] = (vy[i] - vy[i - 2]) * self.fps / 2

        if n > 1:
            ax[-1] = (vx[-1] - vx[-2]) * self.fps
            ay[-1] = (vy[-1] - vy[-2]) * self.fps

        speed = np.hypot(vx, vy)
        accel = np.hypot(ax, ay)

        results = []
        for i in range(n):
            b = raw[i]
            results.append({
                'is_missing': bool(is_missing[i]),
                'x': float(x[i]),
                'y': float(y[i]),
                'x_px': b['x_px'],
                'y_px': b['y_px'],
                'vx': float(vx[i]),
                'vy': float(vy[i]),
                'angle': float(angle[i]),
                'delta_angle': float(delta_angle[i]),
                'speed': float(speed[i]),
                'ax': float(ax[i]),
                'ay': float(ay[i]),
                'accel': float(accel[i])
            })

        return results