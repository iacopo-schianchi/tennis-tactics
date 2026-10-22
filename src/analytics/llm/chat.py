import argparse
import logging
import os

from analytics.llm.agent import DEFAULT_HISTORY_PATH
from analytics.llm.db import PostgresRunner
from analytics.llm.model import Model
from analytics.llm.multi_agent import MultiAgent

# registry queries whose insight is called with condensed=True
CONDENSED_QUERIES = ['reply_direction_by_incoming_shot']


def load_queries():
    from analytics import registry

    for attr in ("QUERIES", "REGISTRY", "queries"):
        found = getattr(registry, attr, None)
        if found:
            return dict(found) if isinstance(found, dict) else {q.name: q for q in found}
    raise SystemExit("Could not find the query registry. Edit load_queries() in chat.py.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--player-id", required=True)
    parser.add_argument("--query", default=None, help="Question. Omit for an interactive session")
    parser.add_argument("--report-file", default="./src/analytics/llm/synthesized_report.txt", help="Synthesized report text, stored as the synthesizer's context")
    parser.add_argument("--schema-file", default="./src/analytics/llm/schema.sql")
    parser.add_argument("--dsn", default=os.environ.get("SUPABASE_DB_URL"))
    parser.add_argument("--history", default=DEFAULT_HISTORY_PATH)
    parser.add_argument("--reset", action="store_true", help="Clear the chat history and report first")
    parser.add_argument("--use-local-llm", action=argparse.BooleanOptionalAction, default=False, help="Load local LLM for chat functionality.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Log plans, SQL attempts and verdicts")
    parser.add_argument("--model-name" default=None, help="Model used for chat functionality.")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(name)s: %(message)s")

    with open(args.schema_file) as f:
        schema = f.read()
    report = None
    if args.report_file:
        with open(args.report_file) as f:
            report = f.read()

    system = MultiAgent(
        Model(name=args.model_name, local=args.use_local_llm),
        PostgresRunner(args.dsn),
        load_queries(),
        schema,
        args.player_id,
        report=report,
        history_path=args.history,
        condensed=CONDENSED_QUERIES,
    )
    if args.reset:
        system.reset()

    if args.query:
        print(system.query(args.query))
        return

    print("Ask a question (blank line or Ctrl-D to quit).")
    while True:
        try:
            question = input("> ").strip()
            print("Reasoning...")
        except EOFError:
            break
        if not question:
            break
        print(system.query(question), "\n")


if __name__ == "__main__":
    main()