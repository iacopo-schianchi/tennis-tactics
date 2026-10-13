import argparse
from uuid import UUID

from analytics.report import AnalyticsReport

import analytics.queries.serve

def uuid_type(value):
    try:
        return str(UUID(value))
    except ValueError as error:
        raise argparse.ArgumentTypeError(f'invalid UUID: {value}') from error

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--player-id", type=uuid_type, default=None, help="UUID of far player")
    parser.add_argument("--query", default=None, help="Tactical query for multi-agent LLM system")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    ar = AnalyticsReport(args.player_id)
    report = ar.generate_report()

    # TODO: handle natural language query

    ar.print_report(report)