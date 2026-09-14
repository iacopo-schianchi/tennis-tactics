from modules.court.annotator import SINGLES_SIDELINE_INSET_M
from utils.consts import COURT_WIDTH
from uuid import uuid4

# singles in range
MIN_X_PERCENT = SINGLES_SIDELINE_INSET_M / COURT_WIDTH * 100.0
MAX_X_PERCENT = (1 - (SINGLES_SIDELINE_INSET_M / COURT_WIDTH)) * 100.0

def gen_id():
    return str(uuid4())

SEARCH_FRAME_PAD = 5

class Persister:
    def __init__(self, players, games_per_set=6, sets_per_match=3):
        if not isinstance(games_per_set, int) or games_per_set < 1:
            raise ValueError('games_per_set must be a positive integer')
        if not isinstance(sets_per_match, int) or sets_per_match < 1:
            raise ValueError('sets_per_match must be a positive integer')

        self.players = players
        self.games_per_set = games_per_set
        self.sets_per_match = sets_per_match

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

    def _form_event(self, frame_id, context, point_id):
        event_context = context[frame_id].get('event', {})

        def player_coord(role, axis):
            value = self._find_near_property(
                frame_id,
                context,
                lambda player: (player.get('feet_pct') or [None, None])[axis],
                role
            )
            return value

        near_player_x = player_coord('near', 0)
        near_player_y = player_coord('near', 1)
        far_player_x = player_coord('far', 0)
        far_player_y = player_coord('far', 1)

        hitting_player = event_context.get('hitting_player')
        shot_type = context[frame_id].get('shot_type')
        shot_speed = context[frame_id].get('speed')
        shot_height = context[frame_id].get('peak')

        if event_context.get('is_hit') and hitting_player is None:
            hitting_player = self._infer_hitting_player(frame_id, context)

        bounce_x, bounce_y = None, None
        is_in = False
        if event_context.get('is_bounce'):
            bounce_x, bounce_y = event_context.get('bounce_coord_pct') or (None, None)

            if bounce_x is not None and bounce_y is not None:
                is_in_y = 0 <= bounce_y <= 100
                is_in_x = MIN_X_PERCENT <= bounce_x <= MAX_X_PERCENT
                is_in = is_in_x and is_in_y


        event = {
            'id': gen_id(),
            'point_id': point_id,
            'next_event_id': None,

            # hit
            'frame_id': frame_id,
            'is_hit': bool(event_context.get('is_hit')),
            'hitting_player': hitting_player,
            'shot_type': shot_type,
            'shot_speed': shot_speed,
            'shot_height': shot_height,

            # bounce
            'is_bounce': bool(event_context.get('is_bounce')),
            'bounce_x': bounce_x,
            'bounce_y': bounce_y,
            'is_in': is_in,

            # shared
            'near_player_x': near_player_x,
            'near_player_y': near_player_y,
            'far_player_x': far_player_x,
            'far_player_y': far_player_y
        }
        return event

    def _infer_hitting_player(self, frame_id, context):
        players = context[frame_id].get('players', {})
        ball = context[frame_id].get('ball', {})
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

    def persist(self, context):
        events, points = [], []
        current_events = []

        def close_point(winner=None, is_winner=False):
            nonlocal current_events
            if not current_events:
                return
            point_id = gen_id()
            for event_order, item in enumerate(current_events, start=1):
                item['point_id'] = point_id
                item['event_order'] = event_order
            point = {
                'id': point_id,
                'game_id': None,
                'point_order': None,
                'winner_id': self._player_id(winner),
                'is_winner': is_winner,
                'last_hit_id': next((item['id'] for item in reversed(current_events) if item['is_hit']), None)
            }
            points.append(point)
            current_events = []

        for frame_id, frame_context in enumerate(context):
            event_context = frame_context.get('event') or {}
            if not event_context.get('is_hit') and not event_context.get('is_bounce'):
                continue
            point_id = current_events[0]['point_id'] if current_events else None
            event_data = self._form_event(frame_id, context, point_id)
            if current_events:
                current_events[-1]['next_event_id'] = event_data['id']
            current_events.append(event_data)
            events.append(event_data)

            if event_data['is_bounce'] and not event_data['is_in']:
                close_point(self._opponent_of_last_hit(current_events), False)
            elif len(current_events) >= 2 and current_events[-1]['is_bounce'] and current_events[-2]['is_bounce']:
                close_point(self._opponent_of_last_hit(current_events), True)

        close_point()
        games, sets, matches = self._build_match(points, events)
        return {
            'players': self._player_rows(),
            'events': events,
            'points': points,
            'games': games,
            'sets': sets,
            'matches': matches,
        }

    def _build_match(self, points, events):
        games = []
        sets = []
        current_game_points = []
        current_game_score = {'near': 0, 'far': 0}
        current_set_games = []
        current_set_score = {'near': 0, 'far': 0}
        match_set_ids = []
        event_by_id = {event['id']: event for event in events}

        def close_game():
            nonlocal current_game_points, current_game_score, current_set_games
            if not current_game_points:
                return
            game_id = gen_id()
            for point in current_game_points:
                point['game_id'] = game_id
            game = {
                'id': game_id,
                'set_id': None,
                'game_order': None,
                'near_player_score': current_game_score['near'],
                'far_player_score': current_game_score['far'],
                'server_id': self._server_id(current_game_points, event_by_id),
                'winner': 'near' if current_game_score['near'] > current_game_score['far'] else 'far',
            }
            games.append(game)
            for point_order, point in enumerate(current_game_points, start=1):
                point['point_order'] = point_order
            game['game_order'] = len(current_set_games) + 1
            current_set_games.append(game)
            current_game_points = []
            current_game_score = {'near': 0, 'far': 0}

        def close_set():
            nonlocal current_set_games, current_set_score
            if not current_set_games:
                return
            set_id = gen_id()
            for game in current_set_games:
                game['set_id'] = set_id
            set_data = {
                'id': set_id,
                'match_id': None,
                'set_order': len(match_set_ids) + 1,
                'near_player_score': current_set_score['near'],
                'far_player_score': current_set_score['far'],
            }
            sets.append(set_data)
            match_set_ids.append(set_id)
            current_set_games = []
            current_set_score = {'near': 0, 'far': 0}

        for point in points:
            winner_role = self._role_for_id(point['winner_id'])
            if winner_role is None:
                continue
            current_game_points.append(point)
            current_game_score[winner_role] += 1
            near_points = current_game_score['near']
            far_points = current_game_score['far']
            game_won = max(near_points, far_points) >= 4 and abs(near_points - far_points) >= 2
            if game_won:
                winner = 'near' if near_points > far_points else 'far'
                close_game()
                current_set_score[winner] += 1
                set_won = current_set_score[winner] >= self.games_per_set
                if set_won:
                    close_set()

        close_game()
        if current_set_games:
            close_set()

        match_id = gen_id()
        for set_data in sets:
            set_data['match_id'] = match_id
        matches = [{
            'id': match_id,
            'near_player_id': self._player_id('near'),
            'far_player_id': self._player_id('far'),
            'near_player_score': sum(set_data['near_player_score'] > set_data['far_player_score'] for set_data in sets),
            'far_player_score': sum(set_data['far_player_score'] > set_data['near_player_score'] for set_data in sets),
        }]
        return games, sets, matches

    def _server_id(self, game_points, event_by_id):
        if not game_points:
            return None
        first_hit_id = game_points[0].get('last_hit_id')
        first_hit = event_by_id.get(first_hit_id, {})
        return self._player_id(first_hit.get('hitting_player'))

    def _player_rows(self):
        rows = []
        for role in ('near', 'far'):
            player = self.players.get(role) if isinstance(self.players, dict) else None
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
        if isinstance(self.players, dict):
            player = self.players.get(role, role)
            return player.get('id') if isinstance(player, dict) else player
        for player in self.players or []:
            if isinstance(player, dict) and player.get('role') == role:
                return player.get('id')
        return role

    def _role_for_id(self, player_id):
        for role in ('near', 'far'):
            if self._player_id(role) == player_id:
                return role
        return None

    def _opponent_of_last_hit(self, events):
        last_hit = next((event for event in reversed(events) if event['is_hit']), None)
        if last_hit is None:
            return None
        return 'far' if last_hit.get('hitting_player') == 'near' else 'near'

"""
---
players
1. id
2. handedness
3. name

---
matches
1. id
2. near_player_id -> players.id
3. far_player_id -> players.id
4. near_player_score
5. far_player_score
6. set_order (['near' | 'far', ...])

---
sets
1. id 
2. match_id -> match.id
3. near_player_score
4. far_player_score
5. game_order (['near' | 'far', ...])

---
games
1. id
2. set_id -> sets.id
3. near_player_score
4. far_player_score
5. point_order (['near' | 'far', ...])
6. server_id -> players.id

---
points
1. game_id -> game.id
2. winner_id -> players.id
3. is_winner (bounces in then out w/o hit)
4. last_hit_id -> event.id

---
event
1. id
2. is_hit
3. near_player_x / near-player_y
4. far_player_x / far_player_y
5. hitting_player ('near' | 'far')
6. shot_type
7. shot_speed
8. shot_height
9. point_id -> points.id
10. next_event_id -> event.id

1. id
2. is_bounce
3. bounce_x / bounce_y
4. near_player_x / near-player_y
5. far_player_x / far_player_y
6. is_in
7. point_id -> point.id
8. next_event_id -> event.id

"""
