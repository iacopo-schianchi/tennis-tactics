class MatchRepository:
    INSERT_ORDER = ('players', 'matches', 'sets', 'games', 'points', 'events')

    def __init__(self, client=None):
        if client is None:
            from db.client import supabase

            client = supabase
        self.client = client

    def _insert_rows(self, table_name, rows):
        if not rows:
            return []
        return self.client.table(table_name).insert(rows).execute().data

    def _update_row(self, table_name, row_id, values):
        return (
            self.client.table(table_name)
            .update(values)
            .eq('id', row_id)
            .execute()
            .data
        )

    def save(self, persisted):
        rows = {table: list(persisted.get(table, [])) for table in self.INSERT_ORDER}

        for table_name in self.INSERT_ORDER[:4]:
            self._insert_rows(table_name, rows[table_name])

        points = [dict(row, last_hit_id=None) for row in rows['points']]
        events = [dict(row, next_event_id=None) for row in rows['events']]
        self._insert_rows('points', points)
        self._insert_rows('events', events)

        for event in rows['events']:
            if event.get('next_event_id') is not None:
                self._update_row(
                    'events',
                    event['id'],
                    {'next_event_id': event['next_event_id']},
                )

        for point in rows['points']:
            if point.get('last_hit_id') is not None:
                self._update_row(
                    'points',
                    point['id'],
                    {'last_hit_id': point['last_hit_id']},
                )

        return persisted