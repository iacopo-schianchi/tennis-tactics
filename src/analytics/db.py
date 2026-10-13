import os
import psycopg
from psycopg.rows import dict_row
from dotenv import load_dotenv

load_dotenv()
DB_URL = os.environ["SUPABASE_DB_URL"]

def run_sql(sql, params=None):
    with psycopg.connect(DB_URL, row_factory=dict_row) as conn:
        conn.execute("set statement_timeout = '10s'")
        return conn.execute(sql, params or {}).fetchall()