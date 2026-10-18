from analytics.registry import Query, register

LOW_POINT_COUNT = 20

def generate_insight(rows):
    stats = {r["is_server"]: r for r in rows}
    served = stats.get(True)
    returned = stats.get(False)

    if not served or not returned: return "Not enough data to compare service aces and return aces."

    service_points = served["points"]
    return_points = returned["points"]

    if service_points == 0 or return_points == 0: return "Not enough data to calculate both ace percentages."

    service_ace_rate = served["aces"] / service_points
    return_ace_rate = returned["aces"] / return_points

    out = ""

    if (service_points < LOW_POINT_COUNT or return_points < LOW_POINT_COUNT): out += "(Warning: low data for accurate analysis)\n"

    out += f"The player wins {service_ace_rate:.1%} of service points with aces and loses {return_ace_rate:.1%} of return points to oppenent aces."

    if service_ace_rate >= 0.10:
        out += " Strength: the serve produces a high proportion of ace points."
    elif service_ace_rate < 0.01:
        out += " Potential weakness: the serve produces few ace points."

    if return_ace_rate >= 0.10:
        out += " Weakness: opponents frequently win return points with aces."
    elif return_ace_rate < 0.01:
        out += " Strength: few return points are lost to aces."

    return out


register(Query(
    name="ace_rate",
    description="Percentage of service points won and lost by aces",
    sql="""
SELECT
    (server_id = %(player_id)s) AS is_server,
    COUNT(*) AS points,
    COUNT(*) FILTER (WHERE end_reason = 'ace') AS aces
FROM points
WHERE %(player_id)s IN (near_player_id, far_player_id)
    AND server_id IS NOT NULL
    AND winner_id IS NOT NULL
GROUP BY (server_id = %(player_id)s)
""",
    generate_insight=generate_insight
))