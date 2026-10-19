from analytics.registry import Query, register
import math

LOW_POINT_COUNT = 8

RALLY_BUCKETS = [
    ("1-3", 1, 3),
    ("4-6", 4, 6),
    ("7-9", 7, 9),
    ("10+", 10, float("inf")),
]

def generate_insight(rows):
    if not rows: return "Not enough rally-length data to analyze."

    counts = {label: 0 for label, _, _ in RALLY_BUCKETS}
    wins = {label: 0 for label, _, _ in RALLY_BUCKETS}

    for row in rows:
        length = row["rally_length"]

        for label, low, high in RALLY_BUCKETS:
            if low <= length <= high:
                counts[label] += 1
                if row["won_point"]:
                    wins[label] += 1
                break

    total = sum(counts.values())

    if total == 0: return "No points with at least one recorded hit to analyze."

    out = "Point win rates by rally length:"

    for label, _, _ in RALLY_BUCKETS:
        count = counts[label]
        bucket_wins = wins[label]

        if count == 0:
            out += f"\n{label} shots: no data."
            continue

        rate = bucket_wins / count
        out += f"\n{label} shots: {rate:.1%} ({bucket_wins}/{count} points won)."

        if count < LOW_POINT_COUNT:
            out += " (Warning: low sample size.)"

    return out

register(Query(
    name="rally_lenght_stats",
    description="Win rate of poitns based on their length",
    sql="""
SELECT
    points.id AS point_id,
    COALESCE(points.winner_id = %(player_id)s, FALSE) AS won_point,
    COUNT(events.id)::INTEGER AS rally_length
FROM points
LEFT JOIN events
    ON events.point_id = points.id
    AND events.is_hit = TRUE
WHERE %(player_id)s IN (points.near_player_id, points.far_player_id)
GROUP BY points.id, points.winner_id
ORDER BY points.id;
""",
    generate_insight=generate_insight
))