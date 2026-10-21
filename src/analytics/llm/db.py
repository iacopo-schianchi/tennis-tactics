class PostgresRunner:
    def __init__(self, dsn, timeout_ms=15000):
        self.dsn = dsn
        self.timeout_ms = int(timeout_ms)

    def __call__(self, sql, params, max_rows=None):
        import psycopg
        from psycopg.rows import dict_row

        with psycopg.connect(self.dsn, row_factory=dict_row) as conn:
            conn.read_only = True
            with conn.cursor() as cur:
                cur.execute(f"SET LOCAL statement_timeout = {self.timeout_ms}")
                cur.execute(sql, params)
                return cur.fetchmany(max_rows) if max_rows else cur.fetchall()