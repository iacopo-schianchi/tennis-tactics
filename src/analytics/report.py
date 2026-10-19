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

    def get_report_string(self, report, print_rows=False):
        out = ""
        for result in report.values():
            out += f"\n--- {result.name} ---\n"
            out += f"{result.description}\n"
            if result.error:
                out += f"Error: {result.error}\n"
                continue

            if print_rows:
                for row in result.rows:
                    out += f"{row}\n"

            if result.insight:
                out += f"\nInsight: {result.insight}\n"

        return out