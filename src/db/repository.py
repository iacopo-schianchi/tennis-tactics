class ClipRepository:
    INSERT_ORDER = ('players', 'clips', 'points', 'events')

    def __init__(self, client=None):
        if client is None:
            from db.client import supabase

            client = supabase
        self.client = client

    def _upsert_rows(self, table_name, rows):
        if not rows:
            return []
        return self.client.table(table_name).upsert(rows).execute().data
    
    def save(self, persisted):
        rows = {table: list(persisted.get(table, [])) for table in self.INSERT_ORDER}

        for table_name in self.INSERT_ORDER[:2]:
            self._upsert_rows(table_name, rows[table_name])

        points = [dict(row, last_hit_id=None) for row in rows['points']]
        events = [dict(row, next_event_id=None) for row in rows['events']]
        self._upsert_rows('points', points)
        self._upsert_rows('events', events)

        return persisted