from dataclasses import dataclass, field
from typing import Any

from analytics.db import run_sql
from analytics.registry import REGISTRY

@dataclass
class QueryResult:
    name: str
    description: str
    rows: list[dict[str, Any]] = field(default_factory=list)
    insight: str | None = None
    error: str | None = None

class AnalyticsReport:
    def __init__(self, player_id):
        self.player_id = player_id

    def generate_report(self):
        report = {}
        for query in REGISTRY.values():
            report[query.name] = self._run_query(query)
        return report

    def _run_query(self, query):
        result = QueryResult(query.name, query.description)

        try:
            result.rows = run_sql(query.sql, {"player_id": self.player_id})
            if query.generate_insight is not None:
                result.insight = query.generate_insight(result.rows)
        except Exception as error:
            result.error = str(error)

        return result

    def print_report(self, report, print_rows=False):
        for result in report.values():
            print(f"\n--- {result.name} ---")
            print(result.description)
            if result.error:
                print(f"Error: {result.error}")
                continue

            if print_rows:
                for row in result.rows:
                    print(row)

            if result.insight:
                print(f"\nInsight: {result.insight}")