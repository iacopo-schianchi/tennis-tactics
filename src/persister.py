from uuid import uuid4

from modules.court.annotator import SINGLES_SIDELINE_INSET_M
from modules.metrics.estimator import ShotTypes
from utils.consts import COURT_WIDTH

from datetime import datetime

SEARCH_FRAME_PAD = 5

MIN_X_PERCENT = SINGLES_SIDELINE_INSET_M / COURT_WIDTH * 100.0
MAX_X_PERCENT = (1 - (SINGLES_SIDELINE_INSET_M / COURT_WIDTH)) * 100.0

NET_Y_PERCENT = 50.0
SEARCH_FRAME_PAD = 5

MAX_GAP_FRAMES = 120


def gen_id():
    return str(uuid4())


def other(role):
    return 'far' if role == 'near' else 'near'

class Persister:
    def __init__(self, players, recorded_on=datetime.now().strftime("%Y-%m-%d")):
        self.players = dict(players)
        self.recorded_on = recorded_on

    def _find_near_property(self, frame_id, context, fetch_child, role=None):
        for offset in range(SEARCH_FRAME_PAD + 1):
            candidates = [frame_id - offset, frame_id + offset]
            for candidate in candidates:
                if not 0 <= candidate < len(context):
                    continue
                players = context[candidate].get('players', {})
                roles = (role,) if role is not None else ('near', 'far')
                for player_role in roles:
                    player = players.get(player_role)
                    if player is None:
                        continue
                    value = fetch_child(player)
                    if value is not None:
                        return value
        return None

    def _form_event(self, frame_id, context):
        event_context = context[frame_id].get('event') or {}

        def player_coord(role, axis):
            return self._find_near_property(frame_id, context, lambda player: (player.get('feet_pct') or [None, None])[axis], role)

        hitting_player = None
        if event_context.get('is_hit'):
            hitting_player = event_context.get('hitting_player') or self._infer_hitting_player(frame_id, context)

        bounce_x, bounce_y, is_in = None, None, None
        if event_context.get('is_bounce'):
            bounce_x, bounce_y = event_context.get('bounce_coord_pct') or (None, None)
            if bounce_x is not None and bounce_y is not None:
                is_in_y = 0 <= bounce_y <= 100
                is_in_x = MIN_X_PERCENT <= bounce_x <= MAX_X_PERCENT
                is_in = is_in_x and is_in_y

        shot_type = context[frame_id].get('shot_type')
        if shot_type is not None:
            shot_type = getattr(shot_type, 'name', shot_type).lower()

        return {
            'id': gen_id(),
            '_frame': frame_id, # for gap checks

            'is_hit': bool(event_context.get('is_hit')),
            'hitting_player': hitting_player, # "near" | "far", converted to player.id
            'shot_type': shot_type,
            'shot_speed': None if context[frame_id].get('speed') is None else float(context[frame_id].get('speed')),
            'shot_height': None if context[frame_id].get('peak') is None else float(context[frame_id].get('peak')),

            'is_bounce': bool(event_context.get('is_bounce')),
            'bounce_x': bounce_x,
            'bounce_y': bounce_y,
            'is_in': is_in,

            'near_player_x': player_coord('near', 0),
            'near_player_y': player_coord('near', 1),
            'far_player_x': player_coord('far', 0),
            'far_player_y': player_coord('far', 1),
        }

    def _get_near_ball(self, frame_id, context):
            ball = context[frame_id].get('ball') or {}
            bx = ball.get('x_px')
            by = ball.get('y_px')
    
            if bx is not None and by is not None:
                return ball
    
            for offset in range(1, SEARCH_FRAME_PAD + 1):
                for idx in (frame_id - offset, frame_id + offset):
                    if 0 <= idx < self.processor.total_frames:
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

    def _side(self, y):
        if y is None or y == NET_Y_PERCENT: return None
        return 'near' if y > NET_Y_PERCENT else 'far'

    # only if hit bounces on other side
    def _starts_point(self, hit, bounce):
        hitter = hit['hitting_player']
        side = self._side(bounce['bounce_y'])
        return hitter is not None and side is not None and side != hitter

    def _mark_serve(self, hit):
        if hit['shot_type'] in ('overhead', 'serve'):
            hit['shot_type'] = 'serve'
            return hit['hitting_player']
        return None

    def _is_ace(self, active):
        hits = [event for event in active if event['is_hit']]
        return len(hits) == 1 and hits[0]['shot_type'] == 'serve'

    # return point winnner, reason
    def _judge(self, active):
        bounce = active[-1]
        previous = active[-2]

        hits = [e for e in active[:-1] if e['is_hit']]
        last_hit = hits[-1] if hits else None
        hitter = last_hit['hitting_player'] if last_hit else None
        if hitter is None:
            return None, None
        side = self._side(bounce['bounce_y'])

        if previous['is_hit']:
            # 1st bounce after hit
            if bounce['is_in'] is False or side == hitter:
                return other(hitter), 'unforced_error'
            return None, None

        # 2nd bounce on same side
        if side is not None and side == self._side(previous['bounce_y']):
            return other(side), ('ace' if self._is_ace(active) else 'winner')
        return None, None

    def _build_point(self, active, winner_role, end_reason, server_role, clip_id):
        point_id = gen_id()
        rows = []
        for order, event in enumerate(active, start=1):
            row = dict(event)
            row.pop('_frame')
            role = row.pop('hitting_player')
            row['hitting_player_id'] = self._player_id(role)
            row['point_id'] = point_id
            row['event_order'] = order
            rows.append(row)

        point = {
            'id': point_id,
            'near_player_id': self._player_id('near'),
            'far_player_id': self._player_id('far'),
            'server_id': self._player_id(server_role),
            'winner_id': self._player_id(winner_role),
            'end_reason': end_reason,
            'clip_id': clip_id
        }
        return point, rows

    def _gen_clip(self):
        clip_id = gen_id()
        return {
            'id': clip_id,
            'recorded_on': self.recorded_on
        }

    def persist(self, context):
        clip = self._gen_clip()

        points, events = [], []
        pending_hit = None
        active = []
        server_role = None
        fault_server = None

        for frame_id, frame_context in enumerate(context):
            event_context = frame_context.get('event') or {}
            is_hit = bool(event_context.get('is_hit'))
            if not is_hit and not event_context.get('is_bounce'):
                continue
            event = self._form_event(frame_id, context)

            # ball lost for too long
            anchor = active[-1] if active else pending_hit
            if anchor is not None and frame_id - anchor['_frame'] > MAX_GAP_FRAMES:
                active, pending_hit, server_role = [], None, None

            if not active:
                if is_hit:
                    pending_hit = event
                    continue
                hit, pending_hit = pending_hit, None
                if hit is None or not self._starts_point(hit, event):
                    continue

                # start point
                active = [hit, event]
                server_role = self._mark_serve(hit)
            elif is_hit:
                previous = active[-1]
                hitter = event['hitting_player']
                if previous['is_hit'] and hitter is not None and previous['hitting_player'] == hitter:
                    continue # duplicate hit by same player
                active.append(event)
                continue
            else:
                active.append(event)

            winner, reason = self._judge(active)
            if winner is None or reason is None:
                continue

            # serve fault
            if server_role is not None and len(active) == 2:
                if fault_server == server_role:
                    winner, reason = other(server_role), 'double_fault'
                else:
                    fault_server = server_role
                    active, server_role = [], None
                    continue

            point, rows = self._build_point(active, winner, reason, server_role, clip['id'])
            points.append(point)
            events.extend(rows)
            active, server_role, fault_server = [], None, None

        return {
            'players': self._player_rows(),
            'points': points,
            'events': events,
            'clips': [clip] if len(points) > 0 else []
        }

    def _player_rows(self):
        rows = []
        for role in ('near', 'far'):
            player = self.players.get(role)
            if isinstance(player, dict):
                rows.append({
                    'id': player.get('id'),
                    'handedness': player.get('handedness'),
                    'name': player.get('name'),
                })
            else:
                rows.append({'id': self._player_id(role), 'handedness': None, 'name': None})
        return rows

    def _player_id(self, role):
        if role is None:
            return None
        player = self.players.get(role, role)
        return player.get('id') if isinstance(player, dict) else player