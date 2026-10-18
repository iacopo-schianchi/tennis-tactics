from analytics.registry import Query, register

LOW_POINT_COUNT = 20

def generate_insight(rows):
    if not rows or rows[0]["avg_serve_speed"] is None: return "Not enough recorded serve-speed data to analyze."

    avg_speed = float(rows[0]["avg_serve_speed"])
    serves_counted = rows[0]["serves_counted"]

    out = f"Average serve speed is {avg_speed:.1f} km/h, based on {serves_counted} recorded serves."

    if serves_counted < LOW_POINT_COUNT:
        out += "(Warning: low sample size for a reliable estimate.)"

    return out


register(Query(
    name="serve_speed",
    description="Average serve speed",
    sql="""
SELECT
    events.hitting_player_id AS player_id,
    AVG(events.shot_speed) AS avg_serve_speed,
    COUNT(*) AS serves_counted
FROM events
WHERE events.hitting_player_id = :player_id
    AND events.is_hit = TRUE
    AND LOWER(events.shot_type) = 'serve'
    AND events.shot_speed IS NOT NULL
GROUP BY events.hitting_player_id;
""",
    generate_insight=generate_insight
))