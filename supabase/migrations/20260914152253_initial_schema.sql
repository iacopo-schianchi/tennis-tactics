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
    winner_id UUID REFERENCES players(id) CHECK (winner_id IN (near_player_id, far_player_id)),
    last_hit_id UUID,
    
    near_player_id UUID REFERENCES players(id),
    far_player_id UUID REFERENCES players(id),
    server_id UUID REFERENCES players(id),
    end_reason TEXT, -- winner, unforced_error, ace, double_fault
    clip_id UUID NOT NULL REFERENCES clips(id) ON DELETE CASCADE,

    CHECK (near_player_id <> far_player_id)
);

CREATE TABLE events (
    id UUID PRIMARY KEY,
    point_id UUID NOT NULL REFERENCES points(id),
    frame_id INTEGER NOT NULL,
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

    UNIQUE (point_id, event_order)
);

ALTER TABLE points
    ADD CONSTRAINT points_last_hit_fk
    FOREIGN KEY (last_hit_id) REFERENCES events(id)
    DEFERRABLE INITIALLY DEFERRED;

CREATE INDEX events_point_idx ON events(point_id);
CREATE INDEX points_near_idx ON points(near_player_id);
CREATE INDEX points_clip_idx ON points(clip_id);
CREATE INDEX points_far_idx ON points(far_player_id);