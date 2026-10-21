-- players: one row per player
CREATE TABLE players (
    id UUID PRIMARY KEY,
    name TEXT NOT NULL,
    handedness TEXT -- 'right' or 'left', may be NULL (treat NULL as right)
);

CREATE TABLE clips (
    id UUID PRIMARY KEY,
    recorded_on DATE NOT NULL
);

-- points: one row per point
CREATE TABLE points (
    id UUID PRIMARY KEY,
    clip_id UUID NOT NULL REFERENCES clips(id),
    near_player_id UUID NOT NULL REFERENCES players(id), -- player on the camera side (y > 50)
    far_player_id UUID NOT NULL REFERENCES players(id), -- player on the far side (y < 50)
    server_id UUID REFERENCES players(id),
    winner_id UUID REFERENCES players(id), -- NULL only when end_reason = 'incomplete'
    end_reason TEXT NOT NULL
        -- 'winner' (last hitter won with a winner), 'ace', 'unforced_error' (last hitter lost the point),
        -- 'double_fault' (server lost), 'incomplete'
);

-- events: one row per hit or bounce, ordered within a point by event_order (1, 2, 3, ...)
-- A hit is normally followed by bounce at event_order + 1 or another  hit (if volley).
CREATE TABLE events (
    id UUID PRIMARY KEY,
    point_id UUID NOT NULL REFERENCES points(id),
    event_order INTEGER NOT NULL,

    is_hit BOOLEAN NOT NULL,
    hitting_player_id UUID REFERENCES players(id), -- set on hits
    shot_type TEXT, -- forehand, backhand, forehand_volley, backhand_volley, overhead, serve
    shot_speed DOUBLE PRECISION, -- km/h
    shot_height DOUBLE PRECISION, -- peak height of the shot (meters)

    is_bounce BOOLEAN NOT NULL,
    bounce_x DOUBLE PRECISION, -- percent of doubles court width (0 to 100, singles lines at 12.5 and 87.5)
    bounce_y DOUBLE PRECISION, -- percent of court length (net at 50, near side > 50, far side < 50)
    is_in BOOLEAN, -- bounce landed in (serves are judged against the service box)

    -- player's feet position at the time of the event, same percentage coordinates as the bounce
    near_player_x DOUBLE PRECISION,
    near_player_y DOUBLE PRECISION,
    far_player_x DOUBLE PRECISION,
    far_player_y DOUBLE PRECISION,

    UNIQUE (point_id, event_order)
);