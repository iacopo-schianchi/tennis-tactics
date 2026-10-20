from analytics.registry import Query, register
from utils.consts import COURT_WIDTH

LOW_POINT_COUNT = 20
LOW_BUCKET_COUNT = 5

COURT_LENGTH_M = 23.77

# % of half court from the net
SHORT_DEPTH = 54.0
DEEP_DEPTH = 80.0

BODY_DIST_M = 1.0
WIDE_DIST_M = 2.5

MIN_LINE_M = 3.0
MAX_LINE_T = 3.0

LEFT_LANE_MAX = 37.5
RIGHT_LANE_MIN = 62.5

RESPONSES = (
    ("crosscourt", "crosscourt"),
    ("down_the_line", "down the line"),
    ("short_crosscourt", "short crosscourt"),
    ("short_down_the_line", "short down the line"),
    ("middle", "middle"),
)

DEPTH_ORDER = ["short", "mid", "deep", "no_bounce", "unknown"]
DEPTH_NAMES = {
    "short": "short",
    "mid": "mid-depth",
    "deep": "deep",
    "no_bounce": "no bounce before contact",
    "unknown": "unknown depth",
}

LATERAL_ORDER = ["body", "moderate", "wide", "unknown"]
LATERAL_NAMES = {
    "body": "to the body",
    "moderate": "moderate reach",
    "wide": "wide",
    "unknown": "unknown position",
}


def _pct(value):
    return "n/a" if value is None else f"{float(value):.0%}"


def _label(row):
    parts = []
    if row["incoming_depth"] is not None:
        parts.append(DEPTH_NAMES[row["incoming_depth"]])
    if row["incoming_lateral"] is not None:
        parts.append(LATERAL_NAMES[row["incoming_lateral"]])
    return "incoming " + ", ".join(parts) if parts else "all incoming rally balls"


def _rank(row):
    has_depth = row["incoming_depth"] is not None
    has_lateral = row["incoming_lateral"] is not None
    kind = 3 if has_depth and has_lateral else 2 if has_lateral else 1 if has_depth else 0
    depth_pos = DEPTH_ORDER.index(row["incoming_depth"]) if has_depth else -1
    lateral_pos = LATERAL_ORDER.index(row["incoming_lateral"]) if has_lateral else -1
    return kind, depth_pos, lateral_pos


def response_insight(rows, condensed=False):
    if not rows or not any(row["n"] for row in rows): return "No rally balls received by this player were recorded."

    groups = {}
    for row in rows:
        if row["stroke"] == "not_returned": continue
        groups.setdefault(row["stroke"], []).append(row)

    out = (
        "How the player handles rally balls they receive."
    )

    for stroke, cells in groups.items():
        total = next((cell["n"] for cell in cells if cell["incoming_depth"] is None and cell["incoming_lateral"] is None), 0)
        warning = ", low sample size for reliable estimates" if total < LOW_POINT_COUNT else ""

        if stroke is None: out += f"\nAll balls received ({total} balls{warning}):"
        else: out += f"\n{stroke.capitalize()} replies ({total} replies{warning}):"

        for cell in sorted(cells, key=_rank):
            if condensed and (cell["incoming_depth"] is not None or cell["incoming_lateral"] is not None): continue
            n = cell["n"]
            low = ", low sample" if n < LOW_BUCKET_COUNT else ""
            returned = cell["returned_n"]
            located = cell["located"]

            line = f"\n  {_label(cell)} (n={n}{low}): "

            if stroke is None: line += f"returned {returned / n:.0%}, not returned {(n - returned) / n:.0%}; "

            if located: line += ", ".join(f"{label} {cell[key] / located:.0%}" for key, label in RESPONSES)
            else: line += "no replies with a tracked landing"

            line += f"; in {_pct(cell['in_rate'])}, point win {_pct(cell['point_win_rate'])}"

            if returned: line += f", reply errors {cell['reply_errors'] / returned:.0%}"

            unlocated = returned - located
            if unlocated: line += f" ({unlocated} replies with no tracked landing)"

            out += line + "."

    return out


register(Query(
    name="reply_direction_by_incoming_shot",
    description="How the player handles rally balls they receive, by the depth and position of the incoming ball.",
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
sequences AS (
    SELECT
        (pp.near_player_id = %(player_id)s) AS is_near,
        COALESCE(NULLIF(LOWER(TRIM(r.shot_type)), ''), 'unknown') AS reply_stroke,
        (r.id IS NOT NULL) AS returned,
        (ib.id IS NOT NULL) AS has_bounce,
        ib.bounce_x AS in_x,
        ib.bounce_y AS in_y,
        ib.is_in AS in_is_in,
        CASE WHEN pp.near_player_id = %(player_id)s THEN i.near_player_x ELSE i.far_player_x END AS recv_x,
        CASE WHEN pp.near_player_id = %(player_id)s THEN i.near_player_y ELSE i.far_player_y END AS recv_y,
        CASE WHEN pp.near_player_id = %(player_id)s THEN i.far_player_x ELSE i.near_player_x END AS hit_x,
        CASE WHEN pp.near_player_id = %(player_id)s THEN i.far_player_y ELSE i.near_player_y END AS hit_y,
        CASE WHEN pp.near_player_id = %(player_id)s THEN r.near_player_x ELSE r.far_player_x END AS contact_x,
        CASE WHEN pp.near_player_id = %(player_id)s THEN r.near_player_y ELSE r.far_player_y END AS contact_y,
        rb.bounce_x AS out_x,
        rb.bounce_y AS out_y,
        rb.is_in AS out_is_in,
        (pp.winner_id = %(player_id)s) AS won_point,
        COALESCE(
            r.id IS NOT NULL
            AND r.event_order = r.last_hit_order
            AND pp.end_reason = 'unforced_error'
            AND pp.winner_id <> %(player_id)s,
            FALSE
        ) AS reply_error
    FROM hits i
    JOIN player_points pp ON pp.id = i.point_id
    LEFT JOIN hits r
        ON r.point_id = i.point_id
        AND r.event_order = i.next_hit_order
        AND r.hitting_player_id = %(player_id)s
    LEFT JOIN events ib
        ON ib.point_id = i.point_id
        AND ib.event_order = i.event_order + 1
        AND ib.is_bounce = TRUE
        AND (r.event_order IS NULL OR ib.event_order < r.event_order)
    LEFT JOIN events rb
        ON rb.point_id = r.point_id
        AND rb.event_order = r.event_order + 1
        AND rb.is_bounce = TRUE
    WHERE i.hitting_player_id IS NOT NULL
        AND i.hitting_player_id <> %(player_id)s
        AND COALESCE(LOWER(TRIM(i.shot_type)), '') <> 'serve'
        AND (ib.id IS NOT NULL OR r.id IS NOT NULL)
),
measured AS (
    SELECT
        s.*,
        CASE
            WHEN in_y IS NULL THEN NULL
            WHEN is_near THEN (in_y - 50) * 2
            ELSE (50 - in_y) * 2
        END AS in_depth,
        CASE
            WHEN out_x IS NULL OR out_y IS NULL THEN NULL
            WHEN is_near AND out_y < 50 THEN (50 - out_y) * 2
            WHEN NOT is_near AND out_y > 50 THEN (out_y - 50) * 2
        END AS out_depth,
        -- line endpoint: first bounce if there was one, else the receiver's contact position (volley)
        CASE WHEN has_bounce THEN in_x ELSE contact_x END AS end_x,
        CASE WHEN has_bounce THEN in_y ELSE contact_y END AS end_y
    FROM sequences s
),
incoming AS (
    -- keep balls that landed in on the receiver's side, or were taken before the bounce
    SELECT *
    FROM measured
    WHERE (in_depth IS NULL OR in_depth >= 0)
        AND in_is_in IS NOT FALSE
),
lined AS (
    SELECT
        i.*,
        (end_x - hit_x) * {COURT_WIDTH} / 100.0 AS dx_m,
        (end_y - hit_y) * {COURT_LENGTH_M} / 100.0 AS dy_m,
        (recv_x - hit_x) * {COURT_WIDTH} / 100.0 AS rx_m,
        (recv_y - hit_y) * {COURT_LENGTH_M} / 100.0 AS ry_m
    FROM incoming i
),
distances AS (
    SELECT
        l.*,
        SQRT(dx_m * dx_m + dy_m * dy_m) AS line_len_m,
        (rx_m * dx_m + ry_m * dy_m) / NULLIF(dx_m * dx_m + dy_m * dy_m, 0) AS t
    FROM lined l
),
labelled AS (
    SELECT
        CASE WHEN returned THEN reply_stroke ELSE 'not_returned' END AS stroke,
        returned,
        won_point,
        reply_error,
        out_is_in,
        CASE
            WHEN NOT has_bounce THEN 'no_bounce'
            WHEN in_depth IS NULL THEN 'unknown'
            WHEN in_depth < {SHORT_DEPTH} THEN 'short'
            WHEN in_depth >= {DEEP_DEPTH} THEN 'deep'
            ELSE 'mid'
        END AS incoming_depth,
        CASE
            WHEN line_len_m IS NULL OR t IS NULL THEN 'unknown'
            WHEN line_len_m < {MIN_LINE_M} THEN 'unknown'
            WHEN t <= 0 OR t > {MAX_LINE_T} THEN 'unknown'
            WHEN ABS(dx_m * ry_m - dy_m * rx_m) / line_len_m <= {BODY_DIST_M} THEN 'body'
            WHEN ABS(dx_m * ry_m - dy_m * rx_m) / line_len_m <= {WIDE_DIST_M} THEN 'moderate'
            ELSE 'wide'
        END AS incoming_lateral,
        CASE
            WHEN NOT returned OR out_depth IS NULL OR contact_x IS NULL THEN 'unlocated'
            WHEN out_x BETWEEN {LEFT_LANE_MAX} AND {RIGHT_LANE_MIN} THEN 'middle'
            WHEN (out_x < {LEFT_LANE_MAX}) = (contact_x < 50) THEN
                CASE WHEN out_depth < {SHORT_DEPTH} THEN 'short_down_the_line' ELSE 'down_the_line' END
            ELSE
                CASE WHEN out_depth < {SHORT_DEPTH} THEN 'short_crosscourt' ELSE 'crosscourt' END
        END AS response
    FROM distances
)
SELECT
    stroke,
    incoming_depth,
    incoming_lateral,
    COUNT(*) AS n,
    COUNT(*) FILTER (WHERE returned) AS returned_n,
    COUNT(*) FILTER (WHERE response <> 'unlocated') AS located,
    COUNT(*) FILTER (WHERE response = 'crosscourt') AS crosscourt,
    COUNT(*) FILTER (WHERE response = 'down_the_line') AS down_the_line,
    COUNT(*) FILTER (WHERE response = 'short_crosscourt') AS short_crosscourt,
    COUNT(*) FILTER (WHERE response = 'short_down_the_line') AS short_down_the_line,
    COUNT(*) FILTER (WHERE response = 'middle') AS middle,
    COUNT(*) FILTER (WHERE reply_error) AS reply_errors,
    AVG(out_is_in::int) AS in_rate,
    AVG(won_point::int) AS point_win_rate
FROM labelled
GROUP BY GROUPING SETS (
    (),
    (incoming_depth),
    (incoming_lateral),
    (incoming_depth, incoming_lateral),
    (stroke),
    (stroke, incoming_depth),
    (stroke, incoming_lateral),
    (stroke, incoming_depth, incoming_lateral)
)
ORDER BY stroke NULLS FIRST, incoming_depth NULLS FIRST, incoming_lateral NULLS FIRST;
""",
    generate_insight=response_insight
))