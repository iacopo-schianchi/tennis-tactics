from analytics.registry import Query, register
from utils.consts import COURT_WIDTH

LOW_POINT_COUNT = 20
LOW_BUCKET_COUNT = 5

COURT_LENGTH_M = 23.77

BUCKET_EDGES = (1.0, 2.0, 3.0, 4.0)

MIN_LINE_M = 3.0
MAX_LINE_T = 3.0

PERSPECTIVES = {
    "player_hitting": "The player's own shots (distance is how far the opponent was from the ball's line)",
    "player_receiving": "The opponent's shots against the player (distance is how far the player was from the ball's line)",
}


def _pct(numerator, denominator):
    return "n/a" if not denominator else f"{numerator / denominator:.0%}"


def _bucket_label(bucket):
    if bucket is None: return "all distances"
    if bucket == 0: return "unknown distance"
    if bucket == 1: return f"under {BUCKET_EDGES[0]:.1f} m"
    if bucket == len(BUCKET_EDGES) + 1: return f"{BUCKET_EDGES[-1]:.1f} m or more"
    return f"{BUCKET_EDGES[bucket - 2]:.1f} to {BUCKET_EDGES[bucket - 1]:.1f} m"


def _bucket_rank(row):
    bucket = row["dist_bucket"]
    if bucket is None: return -1
    return 99 if bucket == 0 else bucket


def distance_insight(rows):
    if not rows or not any(row["n"] for row in rows):
        return "No rally shots that landed in play were recorded for this player."

    groups = {}
    for row in rows:
        groups.setdefault((row["perspective"], row["stroke"]), []).append(row)

    out = "Win rate by distance of receiver to ball trajectory"

    last_perspective = None
    for (perspective, stroke), cells in sorted(
        groups.items(), key=lambda item: (item[0][0], item[0][1] is not None, item[0][1] or "")
    ):
        if perspective != last_perspective:
            out += f"\n\n{PERSPECTIVES[perspective]}:"
            last_perspective = perspective

        total = next((cell["n"] for cell in cells if cell["dist_bucket"] is None), 0)
        warning = ", low sample size for reliable estimates" if total < LOW_POINT_COUNT else ""
        heading = "All strokes" if stroke is None else stroke.capitalize()
        out += f"\n{heading} ({total} shots{warning}):"

        for cell in sorted(cells, key=_bucket_rank):
            n = cell["n"]
            low = ", low sample" if n < LOW_BUCKET_COUNT else ""
            out += f"\n  {_bucket_label(cell['dist_bucket'])} (n={n}{low}): winners {_pct(cell['winners'], n)}, not returned {_pct(cell['unreturned'], n)}, receiver errors {_pct(cell['receiver_errors'], n)}, point won by hitter {_pct(cell['points_won'], cell['decided'])}."

    return out


register(Query(
    name="winner_rate_by_trajectory_distance",
    description="Winner rate, not-returned rate, receiver error rate and point win rate for in-play shots by how far the receiver was from the ball's trajectory, for the player's own shots and for the opponent's shots against the player, overall and by stroke",
    sql=f"""
WITH player_points AS (
    SELECT id, near_player_id, far_player_id, winner_id, end_reason
    FROM points
    WHERE near_player_id = %(player_id)s OR far_player_id = %(player_id)s
),
hits AS (
    SELECT
        e.*,
        LEAD(e.event_order) OVER (PARTITION BY e.point_id ORDER BY e.event_order) AS next_hit_order,
        MAX(e.event_order) OVER (PARTITION BY e.point_id) AS last_hit_order
    FROM events e
    JOIN player_points pp ON pp.id = e.point_id
    WHERE e.is_hit = TRUE
),
shots AS (
    SELECT
        CASE WHEN h.hitting_player_id = %(player_id)s THEN 'player_hitting' ELSE 'player_receiving' END AS perspective,
        COALESCE(NULLIF(LOWER(TRIM(h.shot_type)), ''), 'unknown') AS stroke,
        (h.hitting_player_id = pp.near_player_id) AS hitter_is_near,
        (r.id IS NOT NULL) AS returned,
        (hb.id IS NOT NULL) AS has_bounce,
        hb.bounce_x,
        hb.bounce_y,
        hb.is_in,
        CASE WHEN h.hitting_player_id = pp.near_player_id THEN h.near_player_x ELSE h.far_player_x END AS hit_x,
        CASE WHEN h.hitting_player_id = pp.near_player_id THEN h.near_player_y ELSE h.far_player_y END AS hit_y,
        CASE WHEN h.hitting_player_id = pp.near_player_id THEN h.far_player_x ELSE h.near_player_x END AS recv_x,
        CASE WHEN h.hitting_player_id = pp.near_player_id THEN h.far_player_y ELSE h.near_player_y END AS recv_y,
        CASE WHEN h.hitting_player_id = pp.near_player_id THEN r.far_player_x ELSE r.near_player_x END AS contact_x,
        CASE WHEN h.hitting_player_id = pp.near_player_id THEN r.far_player_y ELSE r.near_player_y END AS contact_y,
        (h.event_order = h.last_hit_order AND pp.end_reason = 'winner') AS is_winner,
        COALESCE(
            r.event_order = r.last_hit_order
            AND pp.end_reason = 'unforced_error'
            AND pp.winner_id = h.hitting_player_id,
            FALSE
        ) AS receiver_error,
        (pp.winner_id = h.hitting_player_id) AS hitter_won
    FROM hits h
    JOIN player_points pp ON pp.id = h.point_id
    LEFT JOIN hits r
        ON r.point_id = h.point_id
        AND r.event_order = h.next_hit_order
        AND r.hitting_player_id IS NOT NULL
        AND r.hitting_player_id <> h.hitting_player_id
    LEFT JOIN events hb
        ON hb.point_id = h.point_id
        AND hb.event_order = h.event_order + 1
        AND hb.is_bounce = TRUE
        AND (r.event_order IS NULL OR hb.event_order < r.event_order)
    WHERE h.hitting_player_id IS NOT NULL
        AND COALESCE(LOWER(TRIM(h.shot_type)), '') <> 'serve'
),
measured AS (
    SELECT
        s.*,
        -- landing depth on the receiver's side, as % of the half court from the net
        CASE
            WHEN bounce_y IS NULL THEN NULL
            WHEN hitter_is_near THEN (50 - bounce_y) * 2
            ELSE (bounce_y - 50) * 2
        END AS depth_raw,
        -- line endpoint: 1st bounce or receiver's contact position (volley)
        CASE WHEN has_bounce THEN bounce_x ELSE contact_x END AS end_x,
        CASE WHEN has_bounce THEN bounce_y ELSE contact_y END AS end_y
    FROM shots s
),
in_play AS (
    SELECT *
    FROM measured
    WHERE (has_bounce OR returned)
        AND (depth_raw IS NULL OR depth_raw >= 0)
        AND is_in IS NOT FALSE
),
lined AS (
    SELECT
        i.*,
        (end_x - hit_x) * {COURT_WIDTH} / 100.0 AS dx_m,
        (end_y - hit_y) * {COURT_LENGTH_M} / 100.0 AS dy_m,
        (recv_x - hit_x) * {COURT_WIDTH} / 100.0 AS rx_m,
        (recv_y - hit_y) * {COURT_LENGTH_M} / 100.0 AS ry_m
    FROM in_play i
),
geom AS (
    SELECT
        l.*,
        SQRT(dx_m * dx_m + dy_m * dy_m) AS line_len_m,
        (rx_m * dx_m + ry_m * dy_m) / NULLIF(dx_m * dx_m + dy_m * dy_m, 0) AS t
    FROM lined l
),
dist AS (
    SELECT
        g.*,
        CASE
            WHEN line_len_m IS NULL OR t IS NULL THEN NULL
            WHEN line_len_m < {MIN_LINE_M} THEN NULL
            WHEN t <= 0 OR t > {MAX_LINE_T} THEN NULL
            ELSE ABS(dx_m * ry_m - dy_m * rx_m) / line_len_m
        END AS dist_m
    FROM geom g
),
labelled AS (
    SELECT
        perspective,
        stroke,
        is_winner,
        returned,
        receiver_error,
        hitter_won,
        CASE
            WHEN dist_m IS NULL THEN 0
            WHEN dist_m < {BUCKET_EDGES[0]} THEN 1
            WHEN dist_m < {BUCKET_EDGES[1]} THEN 2
            WHEN dist_m < {BUCKET_EDGES[2]} THEN 3
            WHEN dist_m < {BUCKET_EDGES[3]} THEN 4
            ELSE 5
        END AS dist_bucket
    FROM dist
)
SELECT
    perspective,
    stroke,
    dist_bucket,
    COUNT(*) AS n,
    COUNT(*) FILTER (WHERE is_winner) AS winners,
    COUNT(*) FILTER (WHERE NOT returned) AS unreturned,
    COUNT(*) FILTER (WHERE receiver_error) AS receiver_errors,
    COUNT(hitter_won) AS decided,
    COUNT(*) FILTER (WHERE hitter_won) AS points_won
FROM labelled
GROUP BY GROUPING SETS (
    (perspective),
    (perspective, dist_bucket),
    (perspective, stroke),
    (perspective, stroke, dist_bucket)
)
ORDER BY perspective, stroke NULLS FIRST, dist_bucket NULLS FIRST;
""",
    generate_insight=distance_insight
))