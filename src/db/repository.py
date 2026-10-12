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
        print(persisted)
        rows = {table: list(persisted.get(table, [])) for table in self.INSERT_ORDER}

        for table_name in self.INSERT_ORDER[:2]:
            self._upsert_rows(table_name, rows[table_name])

        self._upsert_rows('points', rows['points'])
        self._upsert_rows('events', rows['events'])

        return persisted