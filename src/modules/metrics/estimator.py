from enum import Enum
import numpy as np
import math

SHOT_TYPE_COORD_FRAME_PAD = 5
g = 9.81

# TODO: add slices
class ShotTypes(Enum):
    BACKHAND = 0
    BACKHAND_VOLLEY = 1
    FOREHAND = 2
    FOREHAND_VOLLEY = 3
    OVERHEAD = 4
    SERVE = 5

STROKE_HEIGHTS = {
    ShotTypes.BACKHAND: 1.1,
    ShotTypes.BACKHAND_VOLLEY: 1.1,
    ShotTypes.FOREHAND: 1.1,
    ShotTypes.FOREHAND_VOLLEY: 1.1,
    ShotTypes.SERVE: 2.6,
    ShotTypes.OVERHEAD: 2.6
}

class ShotMetricEstimator:
    def __init__(self, processor):
        self.processor = processor

    def process(self, _frames, frame_id, context):
        # shot type, peak, speed

        shot_type = None

        frame_event = context[frame_id].get('event', {}) if isinstance(context[frame_id], dict) else {}
        if frame_event.get('is_hit'):
            shot_type, stroke_height = self._estimate_shot_type(frame_id, context)
            if shot_type is not None:
                self._update_last_metrics(frame_id, context, stroke_height)
        elif frame_event.get('is_bounce'):
            self._update_last_metrics(frame_id, context, 0)

        return {
            'shot_type': shot_type,
            
            # estimate retroactively
            'peak': None,
            'speed': None,
        }

    def _get_surrounding_players(self, frame_id, context):
        players = context[frame_id].get('players') if isinstance(context[frame_id], dict) else None
        near = players.get('near') if players else None
        far = players.get('far') if players else None

        checked_offset = 1
        while (near is None or far is None) and checked_offset <= SHOT_TYPE_COORD_FRAME_PAD and frame_id - checked_offset >= 0 and frame_id + checked_offset < self.processor.total_frames:
            id1 = frame_id - checked_offset
            id2 = frame_id + checked_offset

            players1 = context[id1].get('players') if isinstance(context[id1], dict) else None
            players2 = context[id2].get('players') if isinstance(context[id2], dict) else None

            if near is None:
                if players1 and players1.get('near') is not None:
                    near = players1['near']
                elif players2 and players2.get('near') is not None:
                    near = players2['near']

            if far is None:
                if players1 and players1.get('far') is not None:
                    far = players1['far']
                elif players2 and players2.get('far') is not None:
                    far = players2['far']

            checked_offset += 1

        return {'near': near, 'far': far}

    def _estimate_shot_type(self, frame_id, context):
        players = self._get_surrounding_players(frame_id, context)

        ball = self._get_near_ball(frame_id, context) # TODO: guard against None ball
        bx, by = ball['x_px'], ball['y_px']

        if bx is None or by is None:
            return None, None

        nearest_player, is_far = self._get_nearest_player(players, bx, by)
        if nearest_player is None:
            return None, None
        
        last_event, _ = self._get_last_event(frame_id, context)

        x1, y1, x2, _ = nearest_player['bbox']

        mid_x = int((x1 + x2) / 2)

        shot_type = None

        if (bx > x2 and by < y1):
            # top-right
            if abs(bx - x2) > abs(by - y1):
                shot_type = ShotTypes.BACKHAND if is_far else ShotTypes.FOREHAND
            else:
                shot_type = ShotTypes.OVERHEAD if last_event is not None else ShotTypes.SERVE
        elif (bx < x1 and by < y1):
            # top-left
            if abs(bx - x1) > abs(by - y1):
                shot_type = ShotTypes.FOREHAND if is_far else ShotTypes.BACKHAND
            else:
                shot_type = ShotTypes.OVERHEAD if last_event is not None else ShotTypes.SERVE
        elif (bx > x1 and bx < x2 and by < y1):
            shot_type = ShotTypes.OVERHEAD if last_event is not None else ShotTypes.SERVE
        elif bx < mid_x:
            shot_type = ShotTypes.FOREHAND if is_far else ShotTypes.BACKHAND
        else:
            shot_type = ShotTypes.BACKHAND if is_far else ShotTypes.FOREHAND

        stroke_height = STROKE_HEIGHTS[shot_type]

        if last_event == 'hit' and shot_type in [ShotTypes.FOREHAND, ShotTypes.BACKHAND]:
            shot_type = ShotTypes.FOREHAND_VOLLEY if shot_type == ShotTypes.FOREHAND else ShotTypes.BACKHAND_VOLLEY

        return shot_type, stroke_height
    
    def _get_last_event(self, frame_id, context):
        if frame_id == 0: return None, None

        i = frame_id - 1
        while i >= 0:
            event = context[i]['event']
            if event['is_hit']: return 'hit', i
            elif event['is_bounce']: return 'bounce', i
            i -= 1

        return None, None

    def _get_near_ball(self, frame_id, context):
        bx, by = context[frame_id]['ball']['x_px'], context[frame_id]['ball']['y_px']

        if bx is None or by is None:
            for offset in range(1, SHOT_TYPE_COORD_FRAME_PAD + 1):
                candidates = []

                if frame_id - offset >= 0:
                    candidates.append(frame_id - offset)

                if frame_id + offset < self.processor.total_frames:
                    candidates.append(frame_id + offset)

                for idx in candidates:
                    ball = context[idx].get('ball', {})
                    candidate_x = ball.get('x_px')
                    candidate_y = ball.get('y_px')

                    if candidate_x is not None and candidate_y is not None:
                        return ball

        print(context[max(0, frame_id - SHOT_TYPE_COORD_FRAME_PAD):min(self.processor.total_frames-1, frame_id + SHOT_TYPE_COORD_FRAME_PAD)])

        return {'x_px': None, 'y_px': None}

    # returns nearest_player, is_far
    def _get_nearest_player(self, player_boxes, bx, by):
        if not player_boxes or (player_boxes.get('near') is None and player_boxes.get('far') is None):
            return None, None

        near = player_boxes.get('near')
        far = player_boxes.get('far')

        def center(box):
            x1, y1, x2, y2 = box
            return ((x1 + x2) / 2, (y1 + y2) / 2)

        def dist(p1, p2):
            return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

        if near is None:
            return far, True
        if far is None:
            return near, False

        near_dist = dist(center(near['bbox']), (bx, by))
        far_dist = dist(center(far['bbox']), (bx, by))

        is_near = near_dist < far_dist

        return (near, False) if is_near else (far, True)

    def _update_last_metrics(self, frame_id, context, h1):
        event_type, i = self._get_last_event(frame_id, context)
        if i is None: return

        last_context = context[i]
        frame_separation = frame_id - i
        prev_shot_type = last_context.get('shot_type') if isinstance(last_context, dict) else None
        if event_type == 'bounce':
            h0 = 0
        else:
            if prev_shot_type in STROKE_HEIGHTS:
                h0 = STROKE_HEIGHTS[prev_shot_type]
            else:
                h0 = 1.1 # fallback if not available

        T = frame_separation / self.processor.fps

        peak = self._estimate_peak(T, h0, h1)
        speed_kmh = None
        if event_type == 'hit':
            curr_is_hit = context[frame_id]['event']['is_hit']

            ball = self._get_near_ball(frame_id, context) # TODO: guard against None ball

            curr_nearest_player, _ = self._get_nearest_player(context[frame_id]['players'], ball['x_px'], ball['y_px'])
            curr_court_coords = curr_nearest_player['feet_m'] if curr_is_hit else (ball['x'], ball['y'])

            speed = self._estimate_speed(last_context, curr_court_coords, T, h0, h1, ball)
            speed_kmh = speed * 3.6

        self.processor.set_context(i, {**context[i], 'peak': peak, 'speed': speed_kmh})

    def _estimate_peak(self, T, h0, h1):
        v_y0 = (h1 - h0 + 0.5 * g * T ** 2) / T
        t_up = v_y0 / g
        return h0 + 0.5 * g * t_up ** 2

    def _estimate_v_x0_linear_drag(self, d, T, k=0.25):
        if k == 0:
            return d / T
        return (d * k) / (1 - math.exp(-k * T))

    def _estimate_speed(self, last_context, curr_court_coords, T, h0, h1, ball):
        last_nearest_player, _ = self._get_nearest_player(last_context['players'], ball['x_px'], ball['y_px'])
        last_x, last_y = last_nearest_player['feet_m']
        curr_x, curr_y = curr_court_coords

        d = np.sqrt((last_x - curr_x) ** 2 + (last_y - curr_y) ** 2)

        v_y0 = (h1 - h0 + 0.5 * g * T ** 2) / T
        v_x0 = self._estimate_v_x0_linear_drag(d, T)

        v = math.sqrt(v_x0**2 + v_y0**2)

        return v