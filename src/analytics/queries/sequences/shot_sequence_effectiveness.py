from analytics.registry import Query, register

LOW_POINT_COUNT = 20
LOW_BUCKET_COUNT = 5

# % of half court from net
DEEP_CUTOFF = 70.0

LEFT_LANE_MAX = 37.5
RIGHT_LANE_MIN = 62.5

OUTCOME_WINDOW = 2

TOP_N = 2
BOTTOM_N = 2

LATERAL_NAMES = {
    "forehand": "forehand side",
    "middle": "middle",
    "backhand": "backhand side",
}


def _zone_name(zone):
    depth, lateral = zone.split("_")
    return f"{depth} {LATERAL_NAMES[lateral]}"


def _option(row):
    n = row["n"]
    low = ", low sample" if n < LOW_BUCKET_COUNT else ""
    return (
        f"{_zone_name(row['b_zone'])} (win {row['wins'] / n:.0%}, lose {row['losses'] / n:.0%}, "
        f"neither {row['neithers'] / n:.0%}, n={n}{low})"
    )


def sequence_insight(rows):
    totals = {}
    options = {}
    for row in rows:
        if row["b_zone"] is None:
            totals[row["a_zone"]] = row
        else:
            options.setdefault(row["a_zone"], []).append(row)

    if not totals: return "No two-shot sequences against this player were recorded."

    total_n = sum(row["n"] for row in totals.values())
    warning = " (Warning: low sample size for reliable estimates.)" if total_n < LOW_POINT_COUNT else ""

    out = f"Shot sequences that opponents hit against this player ({total_n} sequences).{warning}"

    if not all(row["handedness_known"] for row in rows):
        out += " Handedness missing; assuming right-handed."

    def best_key(row):
        return row["wins"] / row["n"], row["n"]

    def worst_key(row):
        return row["losses"] / row["n"], row["n"]

    ranked = []
    for a_zone, total in totals.items():
        follow_ups = options.get(a_zone, [])
        top = sorted(follow_ups, key=best_key, reverse=True)[:TOP_N]
        rest = [row for row in follow_ups if row not in top]
        bottom = sorted(rest, key=worst_key, reverse=True)[:BOTTOM_N]
        ranked.append((best_key(top[0]) if top else (-1, 0), a_zone, total, top, bottom))

    for _, a_zone, total, top, bottom in sorted(ranked, key=lambda item: item[0], reverse=True):
        n = total["n"]
        low = ", low sample" if n < LOW_BUCKET_COUNT else ""
        out += (
            f"\n\nOpening shot {_zone_name(a_zone)} (n={n}{low}): "
            f"win {total['wins'] / n:.0%}, lose {total['losses'] / n:.0%}, neither {total['neithers'] / n:.0%}."
        )
        out += "\n  Best follow-ups: " + ("; ".join(_option(row) for row in top) if top else "none.")
        out += "\n  Worst follow-ups: " + ("; ".join(_option(row) for row in bottom) if bottom else "none.")

    return out


def _zone_sql(prefix):
    return f"""
        CASE WHEN {prefix}_depth >= {DEEP_CUTOFF} THEN 'deep' ELSE 'short' END
        || '_' ||
        CASE
            WHEN {prefix}_x BETWEEN {LEFT_LANE_MAX} AND {RIGHT_LANE_MIN} THEN 'middle'
            WHEN ({prefix}_x > {RIGHT_LANE_MIN}) = (recv_is_near <> is_left) THEN 'forehand'
            ELSE 'backhand'
        END"""


register(Query(
    name="opponent_shot_sequence_outcomes",
    description="For the shots opponents hit against this player: for each landing zone of an opening shot, which landing zone of the next opponent shot (after this player's reply) wins or loses the most points. Zones are short or deep by forehand side, middle, or backhand side of this player",
    sql=f"""
WITH player_points AS (
    SELECT id, near_player_id, far_player_id, winner_id
    FROM points
    WHERE near_player_id = %(player_id)s OR far_player_id = %(player_id)s
),
receiver AS (
    SELECT
        COALESCE(LEFT(LOWER(TRIM(p.handedness)), 1) = 'l', FALSE) AS is_left,
        COALESCE(LEFT(LOWER(TRIM(p.handedness)), 1) IN ('l', 'r'), FALSE) AS handedness_known
    FROM (SELECT 1) AS x
    LEFT JOIN players p ON p.id = %(player_id)s
),
hits AS (
    SELECT
        e.*,
        LEAD(e.event_order, 1) OVER w AS next1_order,
        LEAD(e.event_order, 2) OVER w AS next2_order,
        COALESCE(SUM((e.hitting_player_id = %(player_id)s)::int) OVER (
            PARTITION BY e.point_id ORDER BY e.event_order
            ROWS BETWEEN 1 FOLLOWING AND UNBOUNDED FOLLOWING
        ), 0) AS recv_hits_after
    FROM events e
    JOIN player_points pp ON pp.id = e.point_id
    WHERE e.is_hit = TRUE
    WINDOW w AS (PARTITION BY e.point_id ORDER BY e.event_order)
),
sequences AS (
    -- opponent shot A, reply by player_id, opponent shot B
    SELECT
        (pp.near_player_id = %(player_id)s) AS recv_is_near,
        ab.bounce_x AS a_x,
        ab.bounce_y AS a_y,
        ab.is_in AS a_is_in,
        bb.bounce_x AS b_x,
        bb.bounce_y AS b_y,
        b.recv_hits_after,
        pp.winner_id
    FROM hits a
    JOIN player_points pp ON pp.id = a.point_id
    JOIN hits r
        ON r.point_id = a.point_id
        AND r.event_order = a.next1_order
        AND r.hitting_player_id = %(player_id)s
    JOIN hits b
        ON b.point_id = a.point_id
        AND b.event_order = a.next2_order
        AND b.hitting_player_id = a.hitting_player_id
    JOIN events ab
        ON ab.point_id = a.point_id
        AND ab.event_order = a.event_order + 1
        AND ab.is_bounce = TRUE
        AND ab.event_order < a.next1_order
    JOIN events bb
        ON bb.point_id = b.point_id
        AND bb.event_order = b.event_order + 1
        AND bb.is_bounce = TRUE
        AND (b.next1_order IS NULL OR bb.event_order < b.next1_order)
    WHERE a.hitting_player_id IS NOT NULL
        AND a.hitting_player_id <> %(player_id)s
        AND COALESCE(LOWER(TRIM(a.shot_type)), '') <> 'serve'
        AND COALESCE(LOWER(TRIM(b.shot_type)), '') <> 'serve'
),
zoned AS (
    SELECT
        s.*,
        rc.is_left,
        rc.handedness_known,
        -- landing depth on the player_id's side, as % of the half court from the net
        CASE WHEN recv_is_near THEN (a_y - 50) * 2 ELSE (50 - a_y) * 2 END AS a_depth,
        CASE WHEN recv_is_near THEN (b_y - 50) * 2 ELSE (50 - b_y) * 2 END AS b_depth
    FROM sequences s
    CROSS JOIN receiver rc
    WHERE a_x IS NOT NULL AND a_y IS NOT NULL
        AND b_x IS NOT NULL AND b_y IS NOT NULL
        AND a_is_in IS NOT FALSE
),
labelled AS (
    SELECT
        handedness_known,
        {_zone_sql('a')} AS a_zone,
        {_zone_sql('b')} AS b_zone,
        CASE
            WHEN winner_id IS NULL THEN 'neither'
            WHEN recv_hits_after > {OUTCOME_WINDOW} THEN 'neither'
            WHEN winner_id = %(player_id)s THEN 'lose'
            ELSE 'win'
        END AS outcome
    FROM zoned
    WHERE a_depth >= 0 AND b_depth >= 0
)
SELECT
    a_zone,
    b_zone,
    COUNT(*) AS n,
    COUNT(*) FILTER (WHERE outcome = 'win') AS wins,
    COUNT(*) FILTER (WHERE outcome = 'lose') AS losses,
    COUNT(*) FILTER (WHERE outcome = 'neither') AS neithers,
    BOOL_AND(handedness_known) AS handedness_known
FROM labelled
GROUP BY GROUPING SETS ((a_zone, b_zone), (a_zone))
ORDER BY a_zone, b_zone NULLS FIRST;
""",
    generate_insight=sequence_insight
))