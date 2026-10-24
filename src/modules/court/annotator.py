import cv2
import numpy as np
from utils.consts import COLORS_BGR

# court config
COURT_LENGTH_M = 23.77
COURT_WIDTH_M = 10.97

SINGLES_SIDELINE_INSET_M = 1.37
NET_Y_M = COURT_LENGTH_M / 2
SERVICE_LINE_DIST_FROM_NET_M = 6.4
SERVICE_LINE_NEAR_Y_M = NET_Y_M + SERVICE_LINE_DIST_FROM_NET_M
SERVICE_LINE_FAR_Y_M = NET_Y_M - SERVICE_LINE_DIST_FROM_NET_M
CENTER_X_M = COURT_WIDTH_M / 2

MINIMAP_HEIGHT_RATIO = 0.4
MINIMAP_PADDING_RATIO = 0.1 # space between court and minimap box edge, relative to court height
MINIMAP_MARGIN_RATIO = 0.1 # space between minimap box and frame corner, relative to court height
MINIMAP_BOUNCE_WINDOW = 30
MINIMAP_BOUNCE_MAX_RADIUS = 8

MINIMAP_BG_COLOR = (32, 37, 24)
MINIMAP_LINE_COLOR = COLORS_BGR.WHITE.value

MINIMAP_LINES_M = [
    # outer boundary
    ((0, 0), (COURT_WIDTH_M, 0)),
    ((COURT_WIDTH_M, 0), (COURT_WIDTH_M, COURT_LENGTH_M)),
    ((COURT_WIDTH_M, COURT_LENGTH_M), (0, COURT_LENGTH_M)),
    ((0, COURT_LENGTH_M), (0, 0)),
    # singles sidelines
    ((SINGLES_SIDELINE_INSET_M, 0), (SINGLES_SIDELINE_INSET_M, COURT_LENGTH_M)),
    ((COURT_WIDTH_M - SINGLES_SIDELINE_INSET_M, 0), (COURT_WIDTH_M - SINGLES_SIDELINE_INSET_M, COURT_LENGTH_M)),
    # net
    ((0, NET_Y_M), (COURT_WIDTH_M, NET_Y_M)),
    # service lines
    ((SINGLES_SIDELINE_INSET_M, SERVICE_LINE_NEAR_Y_M), (COURT_WIDTH_M - SINGLES_SIDELINE_INSET_M, SERVICE_LINE_NEAR_Y_M)),
    ((SINGLES_SIDELINE_INSET_M, SERVICE_LINE_FAR_Y_M), (COURT_WIDTH_M - SINGLES_SIDELINE_INSET_M, SERVICE_LINE_FAR_Y_M)),
    # center service line
    ((CENTER_X_M, SERVICE_LINE_FAR_Y_M), (CENTER_X_M, SERVICE_LINE_NEAR_Y_M)),
]

def draw_rounded_rect(image, top_left, bottom_right, color, radius):
    x1, y1 = top_left
    x2, y2 = bottom_right

    radius = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)

    cv2.rectangle(image, (x1 + radius, y1), (x2 - radius, y2), color, -1, cv2.LINE_AA)
    cv2.rectangle(image, (x1, y1 + radius), (x2, y2 - radius), color, -1, cv2.LINE_AA)

    for cx, cy in [
        (x1 + radius, y1 + radius),
        (x2 - radius, y1 + radius),
        (x1 + radius, y2 - radius),
        (x2 - radius, y2 - radius),
    ]:
        cv2.circle(image, (cx, cy), radius, color, -1, cv2.LINE_AA)

class CourtAnnotator():
    def draw(self, frame, frame_id, context):
        frame = self.draw_court(frame, frame_id, context)
        frame = self.draw_minimap(frame, frame_id, context)

        return frame

    def draw_minimap(self, frame, frame_id, context):
        frame_h, frame_w = frame.shape[:2]

        court_h = int(frame_h * MINIMAP_HEIGHT_RATIO)
        court_w = int(court_h * (COURT_WIDTH_M / COURT_LENGTH_M))

        padding_px = int(court_h * MINIMAP_PADDING_RATIO)
        margin_px = int(court_h * MINIMAP_MARGIN_RATIO)

        minimap_w = court_w + 2 * padding_px
        minimap_h = court_h + 2 * padding_px

        box_origin_x = frame_w - minimap_w - margin_px
        box_origin_y = margin_px

        court_origin_x = box_origin_x + padding_px
        court_origin_y = box_origin_y + padding_px

        x1 = box_origin_x + 1
        y1 = box_origin_y + 1
        x2 = box_origin_x + minimap_w - 2
        y2 = box_origin_y + minimap_h - 2

        overlay = frame.copy()

        draw_rounded_rect(
            overlay,
            (x1, y1),
            (x2, y2),
            MINIMAP_BG_COLOR,
            radius=12,
        )

        alpha = 0.8
        frame[y1:y2 + 1, x1:x2 + 1] = cv2.addWeighted(
            overlay[y1:y2 + 1, x1:x2 + 1],
            alpha,
            frame[y1:y2 + 1, x1:x2 + 1],
            1 - alpha,
            0,
        )

        def to_minimap_px(x_m, y_m):
            px = court_origin_x + int((x_m / COURT_WIDTH_M) * court_w)
            py = court_origin_y + int((y_m / COURT_LENGTH_M) * court_h)
            return px, py

        for i, (start_m, end_m) in enumerate(MINIMAP_LINES_M):
            p1 = to_minimap_px(*start_m)
            p2 = to_minimap_px(*end_m)
            thickness = 1
            
            # net
            if i == 6:
                thickness = 3
                dist = p1[0] - p2[0]
                cv2.line(frame, (p1[0] - 0.05*dist, p1[1]), (p2[0] + 0.05*dist, p2[1]), MINIMAP_LINE_COLOR, thickness)
            else:
                cv2.line(frame, p1, p2, MINIMAP_LINE_COLOR, thickness)
        
        # players
        players = context[frame_id].get('players') if isinstance(context[frame_id], dict) else None
        if players:
            for key in ('near', 'far'):
                player = players.get(key)
                if not player: continue
                feet = player.get('feet_m')
                if not feet: continue
                px, py = to_minimap_px(*feet)
                cv2.circle(frame, (px, py), 5, COLORS_BGR.NEAR_PLAYER.value if key == 'near' else COLORS_BGR.FAR_PLAYER.value, -1)

        # bounce
        start_i = max(0, frame_id - MINIMAP_BOUNCE_WINDOW)
        bounce_frame = None
        for i in range(frame_id, start_i - 1, -1):
            event = context[i].get('event', {}) if isinstance(context[i], dict) else {}
            if event.get('is_bounce'):
                bounce_frame = i
                break

        if bounce_frame is not None:
            ball = context[bounce_frame].get('ball', {})
            bx_m, by_m = ball.get('x'), ball.get('y')
            if bx_m is not None and by_m is not None:
                frames_since = frame_id - bounce_frame
                t = 1 - (frames_since / MINIMAP_BOUNCE_WINDOW)
                radius = max(1, int(MINIMAP_BOUNCE_MAX_RADIUS * t))
                px, py = to_minimap_px(bx_m, by_m)
                cv2.circle(frame, (px, py), radius, COLORS_BGR.BALL.value, -1)

        return frame

    def draw_court(self, frame, frame_id, context):
        court_points = context[frame_id].get('court').get('points')
        if court_points is None: return

        pts = np.array([[int(x), int(y)] for x, y in court_points], dtype=np.int32)

        overlay = frame.copy()

        cv2.polylines(overlay, [pts], isClosed=True, color=(42, 182, 232), thickness=3, lineType=cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, dst=frame)

        return frame
