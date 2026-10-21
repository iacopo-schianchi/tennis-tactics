import argparse
from uuid import UUID

from analytics.report import AnalyticsReport

import analytics.queries.serve.serve_return
import analytics.queries.serve.ace_rate
import analytics.queries.serve.serve_placement
import analytics.queries.serve.serve_speed

import analytics.queries.strokes.end_stroke
import analytics.queries.strokes.stroke_distribution
import analytics.queries.strokes.distance_for_winner

import analytics.queries.sequences.shot_return_type_distribution
import analytics.queries.sequences.rally_length_stats
import analytics.queries.sequences.shot_sequence_effectiveness

import analytics.queries.positioning.position_depth_stats

from analytics.llm.prompts import REPORT_SYNTHESIZER_SYS
from analytics.llm.model import Model
from analytics.llm.agent import Agent

def uuid_type(value):
    try:
        return str(UUID(value))
    except ValueError as error:
        raise argparse.ArgumentTypeError(f'invalid UUID: {value}') from error

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--player-id", type=uuid_type, default=None, help="UUID of far player")
    parser.add_argument("--query", default=None, help="Tactical query for multi-agent LLM system")
    parser.add_argument("--use-local-llm", action=argparse.BooleanOptionalAction, default=False, help="Load local LLM for chat functionality.")
    return parser.parse_args()

# condense full insight output
CONDENSED_QUERIES = ['reply_direction_by_incoming_shot']

if __name__ == "__main__":
    args = parse_args()

    ar = AnalyticsReport(args.player_id, condensed_queries=CONDENSED_QUERIES)
    report = ar.generate_report()

    full_report, _ = ar.get_report_string(report)
    print(full_report)

    print("Loading model...")

    model = Model(local=args.use_local_llm)

    report_synthesizer = Agent(model, "report_synthesizer", REPORT_SYNTHESIZER_SYS, stateless=True, max_new_tokens=2048, thinking=True)

    print("Synthesizing report...")
    
    synthesized_report = report_synthesizer.ask(f"Generate a synthesized report for this player:\n\n{full_report}")

    print("\n--- Synthesized Report ---")
    print(synthesized_report)

    with open("./src/analytics/llm/synthesized_report.txt", "w") as file:
        file.write(synthesized_report)