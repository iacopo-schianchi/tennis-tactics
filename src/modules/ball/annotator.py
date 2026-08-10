import cv2
import numpy as np
from ...utils.helpers import interpolate_colors
from ...utils.consts import COLORS_BGR

class BallAnnotator():
    BALL_TRAIL_FRAMES = 8

    def draw(self, frame, frame_id, context):
        start_i = max(0, frame_id - self.BALL_TRAIL_FRAMES)
        trail_context = context[start_i:frame_id]

        last_hit_frame = None
        prev_point = None

        for i, frame_context in enumerate(trail_context):
            bx = frame_context.get('ball', {}).get('x_px')
            by = frame_context.get('ball', {}).get('y_px')

            is_hit = frame_context.get('event', {}).get('is_hit', False)
            if is_hit:
                last_hit_frame = start_i + i

            if not bx or not by:
                continue

            point = (int(bx), int(by))

            if is_hit:
                color = COLORS_BGR.RED.value
            elif last_hit_frame is not None:
                frames_since_hit = (start_i + i) - last_hit_frame
                t = min(frames_since_hit / self.BALL_TRAIL_FRAMES, 1.0)
                color = interpolate_colors(COLORS_BGR.RED.value, COLORS_BGR.YELLOW.value, t)
            else:
                color = COLORS_BGR.YELLOW.value

            if prev_point is not None:
                cv2.line(frame, prev_point, point, color, 2)

            cv2.circle(frame, point, 4, color, -1)

            prev_point = point

        return frame