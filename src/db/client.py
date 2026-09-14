import os

from dotenv import load_dotenv
from supabase import Client, create_client

load_dotenv()

url = os.environ["SUPABASE_URL"]
key = os.environ["SUPABASE_PUBLISHABLE_KEY"]

supabase: Client = create_client(url, key)