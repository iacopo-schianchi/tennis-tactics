import cv2
from utils.consts import COLORS_BGR


class MetricAnnotator:
    HIT_EVENT_QUEUE = 2
    SEARCH_FRAME_PAD = 5

    PANEL_BG = (24, 32, 27)
    TEXT_COLOR = (240, 244, 248)
    MUTED_COLOR = (155, 170, 180)
    DIVIDER_COLOR = (65, 76, 73)

    PLAYER_COLORS = {
        "near": COLORS_BGR.NEAR_PLAYER.value,
        "far": COLORS_BGR.FAR_PLAYER.value,
    }

    SPEED_RANGES = (
        (80, (245, 190, 85)),
        (120, (0, 210, 255)),
        (float("inf"), (90, 100, 255)),
    )

    FONT = cv2.FONT_HERSHEY_SIMPLEX

    def __init__(self, player_map):
        self.player_map = player_map

    @staticmethod
    def _rounded_rect(image, x1, y1, x2, y2, color, radius):
        radius = max(0, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))

        cv2.rectangle(image, (x1 + radius, y1), (x2 - radius, y2), color, -1, cv2.LINE_AA)
        cv2.rectangle(image, (x1, y1 + radius), (x2, y2 - radius), color, -1, cv2.LINE_AA)

        for cx, cy in (
            (x1 + radius, y1 + radius),
            (x2 - radius, y1 + radius),
            (x1 + radius, y2 - radius),
            (x2 - radius, y2 - radius),
        ):
            cv2.circle(image, (cx, cy), radius, color, -1, cv2.LINE_AA)

    @classmethod
    def _speed_color(cls, speed):
        if speed is None:
            return cls.MUTED_COLOR

        for upper_bound, color in cls.SPEED_RANGES:
            if speed < upper_bound:
                return color

        return cls.SPEED_RANGES[-1][1]

    @staticmethod
    def _shot_name(shot_type):
        if shot_type is None:
            return "Unknown"

        name = getattr(shot_type, "name", str(shot_type))
        return name.replace("_", " ").replace("-", " ").title()

    def _get_near_ball(self, frame_id, context):
        ball = context[frame_id].get('ball') or {}
        bx = ball.get('x_px')
        by = ball.get('y_px')

        if bx is not None and by is not None:
            return ball

        for offset in range(1, self.SEARCH_FRAME_PAD + 1):
            for idx in (frame_id - offset, frame_id + offset):
                if 0 <= idx < len(context):
                    candidate = context[idx].get('ball') or {}
                    if (candidate.get('x_px') is not None and candidate.get('y_px') is not None):
                        return candidate

        return {'x_px': None, 'y_px': None}

    def _infer_hitting_player(self, frame_id, context):
        players = context[frame_id].get('players', {})
        ball = self._get_near_ball(frame_id, context)
        bx, by = ball.get('x_px'), ball.get('y_px')
        if bx is None or by is None:
            return None

        nearest = None
        nearest_distance = None
        for role in ('near', 'far'):
            player = players.get(role)
            if not player or not player.get('bbox'):
                continue
            x1, y1, x2, y2 = player['bbox']
            distance = ((bx - (x1 + x2) / 2) ** 2 + (by - y1) ** 2) ** 0.5
            if nearest_distance is None or distance < nearest_distance:
                nearest, nearest_distance = role, distance
        return nearest
    
    def _get_role(self, frame_id, context):
        return self._infer_hitting_player(frame_id, context)

    def _draw_text(self, frame, text, position, color, scale=0.48, thickness=1):
        cv2.putText(frame, str(text), position, self.FONT, scale, color, thickness, cv2.LINE_AA)

    def draw(self, frame, frame_id, context):
        recent_shots = []

        for i in range(min(frame_id, len(context) - 1), -1, -1):
            hit_context = context[i]

            if hit_context.get("event", {}).get("is_hit", False):
                recent_shots.append((i, hit_context))

                if len(recent_shots) >= self.HIT_EVENT_QUEUE:
                    break

        if not recent_shots:
            return frame

        frame_h, frame_w = frame.shape[:2]

        margin = max(12, int(frame_w * 0.02))
        panel_x = margin
        panel_y = margin
        panel_w = min(400, frame_w - 2 * margin)

        padding = 12
        header_h = 30
        row_h = 72
        row_gap = 5

        panel_h = 2 * padding + header_h + 8 + len(recent_shots) * row_h + (len(recent_shots) - 1) * row_gap

        panel_h = min(panel_h, frame_h - 2 * margin)

        x1, y1 = panel_x, panel_y
        x2 = panel_x + panel_w - 1
        y2 = panel_y + panel_h - 1

        overlay = frame[y1:y2 + 1, x1:x2 + 1].copy()

        self._rounded_rect(overlay, 0, 0, panel_w - 1, panel_h - 1, self.PANEL_BG, radius=12,)

        roi = frame[y1:y2 + 1, x1:x2 + 1]
        cv2.addWeighted(overlay, 0.88, roi, 0.12, 0, dst=roi)

        # header
        header_baseline = panel_y + padding + 18

        self._draw_text(frame, "RECENT SHOTS", (panel_x + padding, header_baseline), self.TEXT_COLOR, scale=0.55, thickness=2)

        # header label
        badge_text = "LAST 2"
        badge_size = cv2.getTextSize(badge_text, self.FONT, 0.38, 1)[0]

        self._draw_text( frame, badge_text, (panel_x + panel_w - padding - badge_size[0], header_baseline), self.MUTED_COLOR, scale=0.38)

        divider_y = panel_y + padding + header_h
        cv2.line(frame, (panel_x + padding, divider_y), (panel_x + panel_w - padding, divider_y), self.DIVIDER_COLOR, 1, cv2.LINE_AA)

        # shot rows
        row_start_y = divider_y + 8

        for index, (shot_frame_id, hit_context) in enumerate(recent_shots):
            row_y = row_start_y + index * (row_h + row_gap)

            role = self._get_role(shot_frame_id, context)
            player_info = self.player_map.get(role, {}) if role else {}

            player_name = player_info.get("name", role.title() + " Player" if role else "Unknown Player")

            player_color = self.PLAYER_COLORS.get(role, self.MUTED_COLOR)

            shot_type = hit_context.get("shot_type")
            shot_name = self._shot_name(shot_type)

            peak = hit_context.get("peak")
            speed = hit_context.get("speed")

            try: speed = None if speed is None else float(speed)
            except: speed = None

            try: peak = None if peak is None else float(peak)
            except: peak = None

            speed_text = f"{round(speed)} km/h" if speed is not None else "-- km/h"

            height_text = f"{peak:.1f} m" if peak is not None else "--"

            left_x = panel_x + padding
            right_x = panel_x + panel_w - padding
            speed_color = self._speed_color(speed)

            # player identity marker and name
            cv2.circle(frame, (left_x + 4, row_y + 12), 4, player_color, -1, cv2.LINE_AA)

            name_text = f"{player_name}  |  {role.upper() if role else 'PLAYER'}"

            self._draw_text(frame, name_text, (left_x + 14, row_y + 17), player_color, scale=0.45, thickness=2)

            # stroke label and value
            self._draw_text(frame, "STROKE", (left_x, row_y + 39), self.MUTED_COLOR, scale=0.38)

            self._draw_text(frame, shot_name, (left_x + 63, row_y + 39), self.TEXT_COLOR, scale=0.46, thickness=1)

            # speed colored value
            speed_size = cv2.getTextSize(speed_text, self.FONT, 0.47, 2)[0]

            self._draw_text(frame, speed_text, (right_x - speed_size[0], row_y + 39), speed_color, scale=0.47, thickness=2)

            # height label and value
            self._draw_text(frame, "SHOT HEIGHT", (left_x, row_y + 61), self.MUTED_COLOR, scale=0.38)

            self._draw_text(frame, height_text, (left_x + 91, row_y + 61), self.TEXT_COLOR, scale=0.43)

            # separator
            if index < len(recent_shots) - 1:
                separator_y = row_y + row_h + row_gap // 2

                cv2.line(frame, (left_x, separator_y), (right_x, separator_y), self.DIVIDER_COLOR, 1, cv2.LINE_AA)

        return frame