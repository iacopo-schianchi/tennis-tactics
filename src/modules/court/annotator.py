import cv2
import numpy as np
from ...utils.consts import COLORS_BGR

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
MINIMAP_BOUNCE_WINDOW = 15
MINIMAP_BOUNCE_MAX_RADIUS = 8

MINIMAP_BG_COLOR = (40, 90, 40)
MINIMAP_BORDER_COLOR = (60, 140, 60)
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

        cv2.rectangle(
            frame,
            (box_origin_x, box_origin_y),
            (box_origin_x + minimap_w, box_origin_y + minimap_h),
            MINIMAP_BG_COLOR,
            -1,
        )
        cv2.rectangle(
            frame,
            (box_origin_x, box_origin_y),
            (box_origin_x + minimap_w, box_origin_y + minimap_h),
            MINIMAP_BORDER_COLOR,
            1,
        )

        def to_minimap_px(x_m, y_m):
            px = court_origin_x + int((x_m / COURT_WIDTH_M) * court_w)
            py = court_origin_y + int((y_m / COURT_LENGTH_M) * court_h)
            return px, py

        for start_m, end_m in MINIMAP_LINES_M:
            p1 = to_minimap_px(*start_m)
            p2 = to_minimap_px(*end_m)
            cv2.line(frame, p1, p2, MINIMAP_LINE_COLOR, 1)
        
        # players
        players = context[frame_id].get('players') if isinstance(context[frame_id], dict) else None
        if players:
            for key in ('near', 'far'):
                player = players.get(key)
                if not player: continue
                feet = player.get('feet_m')
                if not feet: continue
                px, py = to_minimap_px(*feet)
                cv2.circle(frame, (px, py), 5, COLORS_BGR.RED.value, -1)

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
                cv2.circle(frame, (px, py), radius, COLORS_BGR.BLUE.value, -1)

        return frame

    def draw_court(self, frame, frame_id, context):
        court_points = context[frame_id].get('court').get('points')
        if court_points is None: return

        pts = np.array([[int(x), int(y)] for x, y in court_points], dtype=np.int32)
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

        return frame
