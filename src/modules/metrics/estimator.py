from enum import Enum

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

    def process(self, frames, frame_id, context):
        # shot type, peak, speed

        shot_type = None

        frame_event = context[frame_id]['events']
        if frame_event['is_hit']:
            shot_type, stroke_height = self._estimate_shot_type(frame_id, context)
            self.update_last_event(frame_id, context, stroke_height)
        elif frame_event['is_bounce']:
            self.update_last_event(frame_id, context, 0)

        return {
            'shot_type': shot_type,
            
            # estimate retroactively
            'peak': None,
            'speed': None,
        }
        

    def _estimate_shot_type(self, frame_id, context):
        players = context[frame_id]['players']
        # TODO: get player coords from surrounding frames if missing

        bx, by = context[frame_id]['ball']['x'], context[frame_id]['ball']['y']

        checked_offset = 1
        while bx is None or by is None and checked_offset <= SHOT_TYPE_COORD_FRAME_PAD and frame_id - checked_offset >= 0 and frame_id + checked_offset < self.processor.total_frames:
            id1 = frame_id - checked_offset
            id2 = frame_id + checked_offset

            bx1, by1 = context[id1]['ball']['x'], context[id1]['ball']['y']
            bx2, by2 = context[id2]['ball']['x'], context[id2]['ball']['y']

            if bx1 is not None and by1 is not None:
                bx, by = bx1, by1
            elif by1 is not None and by2 is not None:
                bx, by = bx2, by2
            
            checked_offset += 1
        
        if bx is None or by is None:
            return None

        nearest_player, is_far = self._get_nearest_player(players, bx, by)
        last_event, _ = self._get_last_event(frame_id, context)

        x1, y1, x2, _ = nearest_player

        mid_x = int((x1 + x2) / 2)

        shot_type = None

        if (bx > x2 and by < y1):
            # top-right
            if abs(bx - x2) > abs(y - y1):
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
        if frame_id == 0: return None

        i = frame_id - 1
        while i >= 0:
            event = context[i]['events']
            if event['is_hit']: return 'hit', i
            elif event['is_bounce']: return 'bounce', i
            i -= 1

    def _get_nearest_player(self, player_boxes, bx, by):
        def center(box):
            x1, y1, x2, y2 = box
            return ((x1 + x2) / 2, (y1 + y2) / 2)

        return min(
            player_boxes,
            key=lambda box: (
                (center(box)[0] - bx) ** 2 +
                (center(box)[1] - by) ** 2
            )
        )

    def _update_last_metrics(self, frame_id, context, h1):
        event_type, i = self._get_last_event(frame_id, context)
        last_event = context[i]
        frame_separation = frame_id - i
        h0 = 0 if event_type == 'bounce' else STROKE_HEIGHTS[last_event['shot_type']]

        T = frame_separation / self.processor.fps

        peak = self._estimate_peak(T, h0, h1)
        speed = self._estimate_speed() # TODO

        self.processor.set_context(i, {**context[i], 'peak': peak, 'speed': speed})

    def _estimate_peak(self, T, h0, h1):
        v_y0 = (h1 - h0 + 0.5 * g * T ** 2) / T
        t_up = v_y0 / g
        return h0 + 0.5 * g * t_up ** 2

    def _estimate_speed(self):
        pass