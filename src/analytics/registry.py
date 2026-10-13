
from dataclasses import dataclass
from typing import Callable

@dataclass
class Query:
    name: str
    description: str
    sql: str
    generate_insight: Callable | None = None

REGISTRY: dict[str, Query] = {}

def register(query: Query) -> Query:
    REGISTRY[query.name] = query
    return query