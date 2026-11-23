from itertools import groupby

from analytics.registry import Query, register

LOW_POINT_COUNT = 20

SERVICE_LINE_DEPTH = 54.0
BASELINE_DEPTH = 100.0
DEEP_DEPTH = 117.0

RANGES = (
    ("net", "Net (inside the service line)"),
    ("attacking", "Attacking (service line to baseline)"),
    ("neutral", "Neutral (close to baseline)"),
    ("deep", "Deep (far behind baseline)"),
)


def _position_range(depth):
    if depth < SERVICE_LINE_DEPTH: return "net"
    if depth < BASELINE_DEPTH: return "attacking"
    if depth < DEEP_DEPTH: return "neutral"
    return "deep"


def generate_insight(rows):
    stats = {key: {"n": 0, "winners": 0, "won_within_1": 0, "followed_by_winner": 0} for key, _ in RANGES}

    for _, point_rows in groupby(rows, key=lambda row: row["point_id"]):
        hits = list(point_rows)
        last = len(hits) - 1
        ended_by_winner = hits[0]["end_reason"] == "winner"
        player_won = bool(hits[0]["player_won"])

        for i, hit in enumerate(hits):
            if not hit["by_player"] or (hit["shot_type"] or "").strip().lower() == "serve": continue

            near = hit["hitter_is_near"]
            y = hit["near_player_y"] if near else hit["far_player_y"]
            if y is None: continue
            depth = (y - 50) * 2 if near else (50 - y) * 2

            later = hits[i + 1:]
            opp_hits_after = sum(1 for other in later if not other["by_player"])
            next_opp = next((i + 1 + k for k, other in enumerate(later) if other["by_player"]), None)

            s = stats[_position_range(depth)]
            s["n"] += 1
            s["winners"] += i == last and ended_by_winner
            s["won_within_1"] += player_won and opp_hits_after <= 1
            s["followed_by_winner"] += next_opp == last and ended_by_winner and not player_won

    total = sum(s["n"] for s in stats.values())
    if total == 0: return "No recorded rally shots with a known court position for this player."

    out = f"Outcomes by where the player stood when hitting, across {total} rally shots (serves excluded)."

    if total < LOW_POINT_COUNT: out += "\n(Warning: low sample size for reliable estimates.)"

    for key, label in RANGES:
        s = stats[key]
        n = s["n"]
        if n == 0:
            out += f"\n{label}: no shots recorded."
            continue

        out += f"\n{label} ({n} shots): winners {s['winners'] / n:.1%}, won within 1 opponent hit {s['won_within_1'] / n:.1%}, followed by opponent's winner {s['followed_by_winner'] / n:.1%}."

    return out


register(Query(
    name="outcomes_by_court_position",
    description="For each court position range the player hit from (net, attacking, neutral, deep), the share of shots that were winners, points won within 1 opponent hit, and shots followed by the opponent's winner",
    sql="""
SELECT
    e.point_id,
    e.event_order,
    e.shot_type,
    e.near_player_y,
    e.far_player_y,
    e.hitting_player_id = %(player_id)s AS by_player,
    e.hitting_player_id = p.near_player_id AS hitter_is_near,
    p.winner_id = %(player_id)s AS player_won,
    p.end_reason
FROM events e
JOIN points p ON p.id = e.point_id
WHERE e.is_hit = TRUE
    AND p.end_reason <> 'incomplete'
    AND (p.near_player_id = %(player_id)s OR p.far_player_id = %(player_id)s)
ORDER BY e.point_id, e.event_order;
""",
    generate_insight=generate_insight
))