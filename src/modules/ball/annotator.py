import cv2
import numpy as np
from utils.helpers import interpolate_colors
from utils.consts import COLORS_BGR

class BallAnnotator:
    BALL_TRAIL_FRAMES = 8
    TRAIL_THICKNESS = 2
    SPLINE_STEPS = 6

    @staticmethod
    def catmull_rom(p0, p1, p2, p3, t):
        t2 = t * t
        t3 = t2 * t

        return 0.5 * (
            2 * p1
            + (-p0 + p2) * t
            + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
            + (-p0 + 3 * p1 - 3 * p2 + p3) * t3
        )

    @staticmethod
    def draw_faded_line(frame, p1, p2, color, alpha, thickness):
        if alpha <= 0:
            return

        x1, y1 = p1
        x2, y2 = p2

        h, w = frame.shape[:2]
        pad = thickness + 2

        x_min = max(0, min(x1, x2) - pad)
        y_min = max(0, min(y1, y2) - pad)
        x_max = min(w, max(x1, x2) + pad + 1)
        y_max = min(h, max(y1, y2) + pad + 1)

        if x_min >= x_max or y_min >= y_max:
            return

        roi = frame[y_min:y_max, x_min:x_max]
        overlay = roi.copy()

        cv2.line(overlay, (x1 - x_min, y1 - y_min), (x2 - x_min, y2 - y_min), color,thickness, cv2.LINE_AA,)

        cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, dst=roi,)

    def draw(self, frame, frame_id, context):
        start_i = max(0, frame_id - self.BALL_TRAIL_FRAMES)
        trail_context = context[start_i:frame_id + 1]

        points = []
        hit_frames = []

        for i, frame_context in enumerate(trail_context):
            absolute_frame = start_i + i

            if frame_context.get("event", {}).get("is_hit", False):
                hit_frames.append(absolute_frame)

            ball = frame_context.get("ball", {})
            bx, by = ball.get("x_px"), ball.get("y_px")

            if bx is None or by is None:
                continue

            points.append((absolute_frame, np.array([float(bx), float(by)], dtype=np.float64)))

        if len(points) < 2:
            return frame

        for i in range(len(points) - 1):
            frame1, p1 = points[i]
            frame2, p2 = points[i + 1]

            p0 = points[max(0, i - 1)][1]
            p3 = points[min(len(points) - 1, i + 2)][1]

            samples = []

            for step in range(self.SPLINE_STEPS + 1):
                t = step / self.SPLINE_STEPS
                position = self.catmull_rom(p0, p1, p2, p3, t)

                sample_frame = frame1 + (frame2 - frame1) * t
                samples.append((sample_frame, position))

            for j in range(len(samples) - 1):
                sample_frame1, pos1 = samples[j]
                sample_frame2, pos2 = samples[j + 1]

                midpoint_frame = (sample_frame1 + sample_frame2) / 2

                # fade out
                age = frame_id - midpoint_frame
                alpha = max(0, 1 - age / (self.BALL_TRAIL_FRAMES + 1))

                previous_hits = [hit for hit in hit_frames if hit <= midpoint_frame ]

                if previous_hits:
                    last_hit = previous_hits[-1]
                    since_hit = midpoint_frame - last_hit

                    t_hit = min(since_hit / self.BALL_TRAIL_FRAMES, 1)

                    color = interpolate_colors(COLORS_BGR.HIT_BALL.value, COLORS_BGR.BALL.value, t_hit,)
                else:
                    color = COLORS_BGR.BALL.value

                pt1 = tuple(np.rint(pos1).astype(int))
                pt2 = tuple(np.rint(pos2).astype(int))

                self.draw_faded_line(frame, pt1, pt2, color, alpha, self.TRAIL_THICKNESS,)

        return frame