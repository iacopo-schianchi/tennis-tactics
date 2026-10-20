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

# for report synthesizer context
FILTERED_QUERIES = ['reply_direction_by_incoming_shot', 'winner_rate_by_trajectory_distance']

if __name__ == "__main__":
    args = parse_args()

    ar = AnalyticsReport(args.player_id)
    report = ar.generate_report()

    # TODO: handle natural language query

    full_report, filtered_report = ar.get_report_string(report, filtered_queries=FILTERED_QUERIES)
    print(full_report)

    print("Loading model...")

    report_synthesizer = Model(system_prompt=REPORT_SYNTHESIZER_SYS)

    print("Synthesizing report...")
    
    synthesized_report = report_synthesizer.get_response(f"Generate a synthesized report for this player:\n\n{filtered_report}", max_new_tokens=1024, thinking=True)

    print("\n--- Synthesized Report ---")
    print(synthesized_report)