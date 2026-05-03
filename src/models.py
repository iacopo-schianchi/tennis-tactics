from dataclasses import dataclass, field
from typing import List, Optional
import time

@dataclass
class Shot:
    player_id: str
    shot_type: str
    start_frame: int
    peak_height_px: Optional[float] = None
    bounce_coords_bev: Optional[tuple] = None 
    is_winner: bool = False
    is_error: bool = False

@dataclass
class PointContext:
    point_id: str = field(default_factory=lambda: str(int(time.time())))
    shots: List[Shot] = field(default_factory=list)
    active_shot: Optional[Shot] = None

    def close_current_shot(self):
        if self.active_shot:
            self.shots.append(self.active_shot)
            self.active_shot = None