CREATE TABLE players (
    id UUID PRIMARY KEY,
    name TEXT NOT NULL,
    handedness TEXT
);

CREATE TABLE matches (
    id UUID PRIMARY KEY,
    near_player_id UUID NOT NULL REFERENCES players(id),
    far_player_id UUID NOT NULL REFERENCES players(id),
    near_player_score INTEGER NOT NULL DEFAULT 0,
    far_player_score INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE sets (
    id UUID PRIMARY KEY,
    match_id UUID NOT NULL REFERENCES matches(id),
    near_player_score INTEGER NOT NULL DEFAULT 0,
    far_player_score INTEGER NOT NULL DEFAULT 0,
    set_order INTEGER NOT NULL
);

CREATE TABLE games (
    id UUID PRIMARY KEY,
    set_id UUID NOT NULL REFERENCES sets(id),
    near_player_score INTEGER NOT NULL DEFAULT 0,
    far_player_score INTEGER NOT NULL DEFAULT 0,
    game_order INTEGER NOT NULL,
    server_id UUID REFERENCES players(id),
    winner TEXT NOT NULL
);

CREATE TABLE points (
    id UUID PRIMARY KEY,
    game_id UUID NOT NULL REFERENCES games(id),
    point_order INTEGER NOT NULL,
    winner_id UUID REFERENCES players(id),
    is_winner BOOLEAN NOT NULL DEFAULT FALSE,
    last_hit_id UUID
);

CREATE TABLE events (
    id UUID PRIMARY KEY,
    point_id UUID NOT NULL REFERENCES points(id),
    next_event_id UUID REFERENCES events(id),
    frame_id INTEGER NOT NULL,
    event_order INTEGER NOT NULL,

    is_hit BOOLEAN NOT NULL DEFAULT FALSE,

    near_player_x DOUBLE PRECISION,
    near_player_y DOUBLE PRECISION,
    far_player_x DOUBLE PRECISION,
    far_player_y DOUBLE PRECISION,

    hitting_player TEXT,

    shot_type TEXT,
    shot_speed DOUBLE PRECISION,
    shot_height DOUBLE PRECISION,

    is_bounce BOOLEAN NOT NULL DEFAULT FALSE,
    bounce_x DOUBLE PRECISION,
    bounce_y DOUBLE PRECISION,
    is_in BOOLEAN
);

ALTER TABLE points
    ADD CONSTRAINT points_last_hit_fk
    FOREIGN KEY (last_hit_id) REFERENCES events(id);

ALTER TABLE events
    ADD CONSTRAINT events_point_order_unique
    UNIQUE (point_id, event_order);

ALTER TABLE points
    ADD CONSTRAINT points_game_order_unique
    UNIQUE (game_id, point_order);

ALTER TABLE games
    ADD CONSTRAINT games_set_order_unique
    UNIQUE (set_id, game_order);

ALTER TABLE sets
    ADD CONSTRAINT sets_match_order_unique
    UNIQUE (match_id, set_order);