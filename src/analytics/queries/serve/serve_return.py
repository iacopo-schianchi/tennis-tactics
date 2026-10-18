from analytics.registry import Query, register

LOW_POINT_COUNT = 20

def generate_insight(rows):
    stats = {r["is_server"]: r for r in rows}
    served, returned = stats.get(True), stats.get(False)
    if not served or not returned: return "Not enough data to compare serving and returning."

    out = ""

    if served["points"] < LOW_POINT_COUNT or returned["points"] < LOW_POINT_COUNT:
        out += "(Warning: low data for accurate analysis)\n"

    serve_rate = served["wins"] / served["points"]
    return_rate = returned["wins"] / returned["points"]
    gap = serve_rate - return_rate

    out += f"Wins {serve_rate:.0%} of service points and {return_rate:.0%} of return points."
    if gap < 0:
        return out + " Weakness: serve is less effective than returning."
    if gap < 0.10:
        return out + " Weakness: serve gives little edge over returning."
    if return_rate < 0.25:
        return out + " Weakness: returns."
    return out


register(Query(
    name="win_rate_serving_vs_returning",
    description="Point win rate when serving vs. returning",
    sql="""
SELECT
    (server_id = %(player_id)s) AS is_server,
    COUNT(*) AS points,
    COUNT(*) FILTER (WHERE winner_id = %(player_id)s) AS wins
FROM points
WHERE %(player_id)s IN (near_player_id, far_player_id) AND winner_id IS NOT NULL
GROUP BY (server_id = %(player_id)s)
""",
    generate_insight=generate_insight
))