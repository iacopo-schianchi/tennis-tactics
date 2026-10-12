CREATE TABLE players (
    id UUID PRIMARY KEY,
    name TEXT NOT NULL,
    handedness TEXT
);

CREATE TABLE clips (
    id UUID PRIMARY KEY,
    recorded_on DATE NOT NULL DEFAULT CURRENT_DATE
);

CREATE TABLE points (
    id UUID PRIMARY KEY,
    clip_id UUID NOT NULL REFERENCES clips(id) ON DELETE CASCADE,

    near_player_id UUID NOT NULL REFERENCES players(id),
    far_player_id UUID NOT NULL REFERENCES players(id),
    server_id UUID REFERENCES players(id),

    winner_id UUID REFERENCES players(id),
    end_reason TEXT NOT NULL,

    CONSTRAINT points_distinct_players CHECK (near_player_id <> far_player_id),
    CONSTRAINT points_winner_is_player CHECK (winner_id IN (near_player_id, far_player_id)),
    CONSTRAINT points_server_is_player CHECK (server_id IN (near_player_id, far_player_id)),
    CONSTRAINT points_end_reason_valid CHECK (end_reason IN ('winner', 'unforced_error', 'ace', 'double_fault', 'incomplete')),
    CONSTRAINT points_incomplete_has_no_winner CHECK ((winner_id IS NULL) = (end_reason = 'incomplete'))
);

CREATE TABLE events (
    id UUID PRIMARY KEY,
    point_id UUID NOT NULL REFERENCES points(id) ON DELETE CASCADE,
    event_order INTEGER NOT NULL,

    is_hit BOOLEAN NOT NULL DEFAULT FALSE,

    near_player_x DOUBLE PRECISION,
    near_player_y DOUBLE PRECISION,
    far_player_x DOUBLE PRECISION,
    far_player_y DOUBLE PRECISION,

    hitting_player_id UUID REFERENCES players(id),

    shot_type TEXT,
    shot_speed DOUBLE PRECISION,
    shot_height DOUBLE PRECISION,

    is_bounce BOOLEAN NOT NULL DEFAULT FALSE,
    bounce_x DOUBLE PRECISION,
    bounce_y DOUBLE PRECISION,
    is_in BOOLEAN,

    CONSTRAINT events_point_order_unique UNIQUE (point_id, event_order),
    CONSTRAINT events_is_hit_or_bounce CHECK (is_hit OR is_bounce)
);

CREATE INDEX points_clip_idx ON points(clip_id);
CREATE INDEX points_near_idx ON points(near_player_id);
CREATE INDEX points_far_idx ON points(far_player_id);
CREATE INDEX points_winner_idx ON points(winner_id);
CREATE INDEX events_hitting_player_idx ON events(hitting_player_id);