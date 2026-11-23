from analytics.registry import Query, register
from modules.court.annotator import SINGLES_SIDELINE_INSET_M
from utils.consts import COURT_WIDTH

LOW_POINT_COUNT = 10

MIN_X_PERCENT = SINGLES_SIDELINE_INSET_M / COURT_WIDTH * 100.0
MAX_X_PERCENT = (1 - (SINGLES_SIDELINE_INSET_M / COURT_WIDTH)) * 100.0

SERVICE_THIRD_PERCENT = (MAX_X_PERCENT - MIN_X_PERCENT) / 6.0

# x-coordinate right percentage boundary for deuce and ad serves
DEUCE_SERVE_BUCKET_X_ENDS = {
    'wide': MIN_X_PERCENT + SERVICE_THIRD_PERCENT,
    'body': MIN_X_PERCENT + 2 * SERVICE_THIRD_PERCENT,
    'T': MIN_X_PERCENT + 3 * SERVICE_THIRD_PERCENT
}

AD_SERVE_BUCKET_X_ENDS = {
    'T': MAX_X_PERCENT - 2 * SERVICE_THIRD_PERCENT,
    'body': MAX_X_PERCENT - SERVICE_THIRD_PERCENT,
    'wide': MAX_X_PERCENT
}

def generate_insight(rows):
    counts = {
        "deuce": {"wide": 0, "body": 0, "T": 0},
        "ad": {"T": 0, "body": 0, "wide": 0},
    }

    center_x_percent = 50

    for row in rows:
        x = row["bounce_x"]

        if x is None: continue

        if x <= center_x_percent:
            side = "deuce"
            bucket_ends = DEUCE_SERVE_BUCKET_X_ENDS
            bucket_order = ("wide", "body", "T")
        else:
            side = "ad"
            bucket_ends = AD_SERVE_BUCKET_X_ENDS
            bucket_order = ("T", "body", "wide")

        for bucket in bucket_order:
            if x <= bucket_ends[bucket]:
                counts[side][bucket] += 1
                break

    side_totals = {
        side: sum(bucket_counts.values())
        for side, bucket_counts in counts.items()
    }

    total = sum(side_totals.values())

    if total == 0: return "Not enough valid serve-bounce data to analyze placement."

    out = "Serve placement among included in-bounds serves: "

    for side in ("deuce", "ad"):
        side_total = side_totals[side]

        if side_total == 0:
            out += f"No valid {side}-side serves to analyze. "
            continue

        bucket_order = (
            ("wide", "body", "T")
            if side == "deuce"
            else ("T", "body", "wide")
        )

        percentages = ", ".join(
            f"{bucket}: {counts[side][bucket] / side_total:.0%} ({counts[side][bucket]})"
            for bucket in bucket_order
        )

        out += f"{side.capitalize()} side (n={side_total}): {percentages}."

    low_data_sides = [side for side, side_total in side_totals.items() if side_total < LOW_POINT_COUNT]

    if low_data_sides: out += f"Warning: low sample size for {' and '.join(low_data_sides)}-side placement estimates."

    return out


register(Query(
    name="serve_placement",
    description="Placement of serve on ad and deuce side.",
    sql="""
SELECT points.id, events.bounce_x
FROM points
JOIN events
    ON events.point_id = points.id
WHERE points.server_id = %(player_id)s
    AND events.is_bounce = TRUE
    AND events.is_in = TRUE
    AND events.bounce_x IS NOT NULL
    AND events.event_order = (
        SELECT MIN(events.event_order)
        FROM events
        WHERE events.point_id = points.id
        AND events.is_bounce = TRUE
        AND events.event_order > (
            SELECT MIN(events.event_order)
            FROM events
            WHERE events.point_id = points.id
                AND events.is_hit = TRUE
                AND events.hitting_player_id = points.server_id
                AND events.shot_type = 'serve'
        )
    );
""",
    generate_insight=generate_insight
))