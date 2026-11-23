from analytics.registry import Query, register

LOW_POINT_COUNT = 20
LOW_BUCKET_COUNT = 5

LOW_GRID_COUNT = 45

METRICS = (("speed", "Speed", "km/h"), ("height", "Height", "meters"), ("depth", "Depth", "\% from net to baseline"))

DEPTH_NOTE = "Depth is % of the half court from the net (0 = net, 100 = opponent baseline, above 100 = long)."

SPEED_LABELS = {1: "slow", 2: "medium", 3: "fast"}
DEPTH_LABELS = {1: "short", 2: "medium", 3: "deep"}


def _num(value):
    return "n/a" if value is None else f"{float(value):.1f}"


def _pct(value):
    return "n/a" if value is None else f"{float(value):.0%}"


SHOTS_CTE = """
WITH last_hits AS (
    SELECT point_id, MAX(event_order) AS last_hit_order
    FROM events
    WHERE is_hit = TRUE
    GROUP BY point_id
),
shots AS (
    SELECT
        COALESCE(NULLIF(LOWER(TRIM(e.shot_type)), ''), 'unknown') AS shot_type,
        e.shot_speed,
        e.shot_height,
        b.is_in,
        CASE
            WHEN b.bounce_y IS NULL THEN NULL
            WHEN e.hitting_player_id = p.near_player_id AND b.bounce_y < 50 THEN (50 - b.bounce_y) * 2
            WHEN e.hitting_player_id = p.far_player_id AND b.bounce_y > 50 THEN (b.bounce_y - 50) * 2
            ELSE NULL
        END AS depth,
        (p.winner_id = e.hitting_player_id) AS won_point,
        (e.event_order = lh.last_hit_order AND p.end_reason IN ('winner', 'ace')) AS is_winner,
        (e.event_order = lh.last_hit_order AND p.end_reason IN ('unforced_error', 'double_fault')) AS is_error
    FROM events e
    JOIN points p ON p.id = e.point_id
    JOIN last_hits lh ON lh.point_id = e.point_id
    LEFT JOIN events b
        ON b.point_id = e.point_id
        AND b.event_order = e.event_order + 1
        AND b.is_bounce = TRUE
    WHERE e.hitting_player_id = %(player_id)s AND e.is_hit = TRUE
)
"""


# distribution of speed, height and depth per stroke
def distribution_insight(rows):
    if not rows: return "No recorded hits found for this player."

    if sum(row["hits"] for row in rows) == 0: return "No recorded hits found for this player."

    out = f"Shot speed, height and depth by stroke (median and interquartile range). {DEPTH_NOTE}"

    for row in sorted(rows, key=lambda row: row["hits"], reverse=True):
        low = ", low sample" if row["hits"] < LOW_POINT_COUNT else ""
        out += f"\n{row['shot_type'].capitalize()} ({row['hits']} hits{low}):"

        for key, label, unit in METRICS:
            n = row[f"{key}_n"]
            if not n:
                out += f"\n  {label}: no data."
                continue
            out += (
                f"\n  {label} ({unit}): median {_num(row[f'{key}_median'])}, "
                f"IQR {_num(row[f'{key}_p25'])} to {_num(row[f'{key}_p75'])}, "
                f"avg {_num(row[f'{key}_avg'])} (n={n})."
            )

    return out


register(Query(
    name="shot_metric_distribution",
    description="Average, median and quartiles of the player's shot speed, shot height and shot depth for each stroke type",
    sql=SHOTS_CTE + """
SELECT
    shot_type,
    COUNT(*) AS hits,

    COUNT(shot_speed) AS speed_n,
    AVG(shot_speed) AS speed_avg,
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY shot_speed) AS speed_p25,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY shot_speed) AS speed_median,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY shot_speed) AS speed_p75,

    COUNT(shot_height) AS height_n,
    AVG(shot_height) AS height_avg,
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY shot_height) AS height_p25,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY shot_height) AS height_median,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY shot_height) AS height_p75,

    COUNT(depth) AS depth_n,
    AVG(depth) AS depth_avg,
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY depth) AS depth_p25,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY depth) AS depth_median,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY depth) AS depth_p75
FROM shots
GROUP BY shot_type
ORDER BY COUNT(*) DESC;
""",
    generate_insight=distribution_insight
))


# outcomes by quartile of each metric
def outcome_insight(rows):
    if not rows: return "Not enough recorded hits to compare shot speed, height and depth against outcomes."

    groups = {}
    for row in rows:
        groups.setdefault((row["shot_type"], row["metric"]), []).append(row)

    out = "Outcomes by quartile of shot speed, height and depth, within the player's own distribution for each stroke (Q1 = lowest values, Q4 = highest). 'in' is the share of shots that landed in, 'point win' is the share of points won, 'winners' and 'errors' are the share of shots that ended the point that way. " + DEPTH_NOTE

    for (stroke, metric), buckets in groups.items():
        total = sum(bucket["n"] for bucket in buckets)
        warning = ", low sample size for reliable estimates" if total < LOW_POINT_COUNT else ""
        out += f"\n{stroke.capitalize()} by {metric} ({total} shots{warning}):"
        for bucket in sorted(buckets, key=lambda row: row["bucket"]):
            low = ", low sample" if bucket["n"] < LOW_BUCKET_COUNT else ""
            out += f"\n  Q{bucket['bucket']} ({_num(bucket['bucket_min'])} to {_num(bucket['bucket_max'])}, n={bucket['n']}{low}): in {_pct(bucket['in_rate'])}, point win {_pct(bucket['point_win_rate'])}, winners {_pct(bucket['winner_rate'])}, errors {_pct(bucket['error_rate'])}."

    return out


register(Query(
    name="shot_metric_outcomes",
    description="In-rate, point win rate, winner rate and error rate of the player's shots by quartile of shot speed, shot height and shot depth, for each stroke type",
    sql=SHOTS_CTE + """,
metrics AS (
    SELECT s.shot_type, v.metric, v.value, s.is_in, s.won_point, s.is_winner, s.is_error
    FROM shots s
    CROSS JOIN LATERAL (
        VALUES ('speed', s.shot_speed), ('height', s.shot_height), ('depth', s.depth)
    ) AS v(metric, value)
    WHERE v.value IS NOT NULL
),
bucketed AS (
    SELECT
        *,
        NTILE(4) OVER (PARTITION BY shot_type, metric ORDER BY value) AS bucket
    FROM metrics
)
SELECT
    shot_type,
    metric,
    bucket,
    COUNT(*) AS n,
    MIN(value) AS bucket_min,
    MAX(value) AS bucket_max,
    AVG(is_in::int) AS in_rate,
    AVG(won_point::int) AS point_win_rate,
    AVG(is_winner::int) AS winner_rate,
    AVG(is_error::int) AS error_rate
FROM bucketed
GROUP BY shot_type, metric, bucket
ORDER BY shot_type, metric, bucket;
""",
    generate_insight=outcome_insight
))


# speed by depth interaction
def interaction_insight(rows):
    if not rows: return "Not enough recorded hits to compare shot speed and depth together."

    groups = {}
    for row in rows:
        groups.setdefault(row["shot_type"], []).append(row)

    out = "Outcomes by shot speed and depth tercile, within the player's own distribution for each stroke. " + DEPTH_NOTE

    for stroke, cells in groups.items():
        total = sum(cell["n"] for cell in cells)
        warning = ", low sample size for reliable estimates" if total < LOW_GRID_COUNT else ""
        out += f"\n{stroke.capitalize()} ({total} shots{warning}):"
        for cell in sorted(cells, key=lambda row: (row["speed_tile"], row["depth_tile"])):
            low = ", low sample" if cell["n"] < LOW_BUCKET_COUNT else ""
            out += f"\n  {SPEED_LABELS[cell['speed_tile']]} and {DEPTH_LABELS[cell['depth_tile']]} (speed {_num(cell['speed_min'])} to {_num(cell['speed_max'])}, depth {_num(cell['depth_min'])} to {_num(cell['depth_max'])}, n={cell['n']}{low}): in {_pct(cell['in_rate'])}, point win {_pct(cell['point_win_rate'])}."

        usable = [cell for cell in cells if cell["n"] >= LOW_BUCKET_COUNT and cell["in_rate"] is not None]
        if usable:
            worst = min(usable, key=lambda cell: cell["in_rate"])
            best = max(usable, key=lambda cell: cell["in_rate"])
            out += f"\n  Lowest in-rate: {SPEED_LABELS[worst['speed_tile']]} and {DEPTH_LABELS[worst['depth_tile']]} ({_pct(worst['in_rate'])}). Highest in-rate: {SPEED_LABELS[best['speed_tile']]} and {DEPTH_LABELS[best['depth_tile']]} ({_pct(best['in_rate'])})."

    return out


register(Query(
    name="shot_speed_depth_interaction",
    description="In-rate and point win rate of the player's shots in a 3x3 grid of shot speed and shot depth, for each stroke type",
    sql=SHOTS_CTE + """,
tiled AS (
    SELECT
        *,
        NTILE(3) OVER (PARTITION BY shot_type ORDER BY shot_speed) AS speed_tile,
        NTILE(3) OVER (PARTITION BY shot_type ORDER BY depth) AS depth_tile
    FROM shots
    WHERE shot_speed IS NOT NULL AND depth IS NOT NULL
)
SELECT
    shot_type,
    speed_tile,
    depth_tile,
    COUNT(*) AS n,
    MIN(shot_speed) AS speed_min,
    MAX(shot_speed) AS speed_max,
    MIN(depth) AS depth_min,
    MAX(depth) AS depth_max,
    AVG(is_in::int) AS in_rate,
    AVG(won_point::int) AS point_win_rate
FROM tiled
GROUP BY shot_type, speed_tile, depth_tile
ORDER BY shot_type, speed_tile, depth_tile;
""",
    generate_insight=interaction_insight
))