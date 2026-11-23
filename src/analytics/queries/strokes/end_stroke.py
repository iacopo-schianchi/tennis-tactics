from analytics.registry import Query, register
from collections import defaultdict

LOW_POINT_COUNT = 20

def generate_insight(rows):
    if not rows: return "Not enough end-stroke data to analyze."

    stats = defaultdict(lambda: {
            "total": 0,
            "winners": 0,
            "unforced_errors": 0,
            "other": 0,
    })

    for row in rows:
        shot_type = row["shot_type"]

        if not shot_type: continue

        stroke = shot_type.strip().lower()
        outcome = row["outcome"]

        stats[stroke]["total"] += 1

        if outcome == "winner":
            stats[stroke]["winners"] += 1
        elif outcome == "unforced_error":
            stats[stroke]["unforced_errors"] += 1
        else:
            stats[stroke]["other"] += 1

    if not stats: return "Not enough end-stroke data to analyze."

    out = "End-stroke analysis: "

    for stroke, data in sorted(stats.items(), key=lambda item: item[1]["total"], reverse=True):
        total = data["total"]

        winner_rate = data["winners"] / total
        error_rate = data["unforced_errors"] / total
        other_rate = data["other"] / total

        out += f"\n{stroke.capitalize()} (n={total}): {winner_rate:.1%} winners ({data['winners']}), {error_rate:.1%} unforced errors ({data['unforced_errors']}), {other_rate:.1%} other ({data['other']})."

        if total < LOW_POINT_COUNT: out += " (Warning: low sample size.)"

    return out


register(Query(
    name="end_stroke_stats",
    description="Winners and errors by stroke type for end strokes in a rally",
    sql="""
SELECT
    points.id AS point_id,
    events.shot_type,
    CASE
        WHEN points.end_reason = 'winner'
            AND points.winner_id = events.hitting_player_id
            THEN 'winner'
        WHEN points.end_reason = 'unforced_error'
            AND points.winner_id <> events.hitting_player_id
            THEN 'unforced_error'
        ELSE 'other'
    END AS outcome
FROM points
JOIN events
    ON events.point_id = points.id
WHERE events.is_hit = TRUE
    AND events.hitting_player_id = %(player_id)s
    AND %(player_id)s IN (
        points.near_player_id,
        points.far_player_id
    )
    AND points.end_reason IN ('winner', 'unforced_error')
    AND events.shot_type IS NOT NULL
    AND events.event_order = (
        SELECT MAX(events.event_order)
        FROM events
        WHERE events.point_id = points.id
        AND events.is_hit = TRUE
    );
""",
    generate_insight=generate_insight
))