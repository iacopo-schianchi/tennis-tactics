from src.models import PointContext, Shot
from src.physics import TrajectoryProcessor

WAITING = 0
IN_POINT = 1
SAVING = 2

class MatchOrchestrator:
    def __init__(self):
        self.state = WAITING
        self.context = None
        self.physics = TrajectoryProcessor()
    
    def process_frame(self, frame_data):
        if self.state == WAITING:
            if self.detect_serve(frame_data):
                self.context = PointContext(server_id=frame_data.server_id)
                self.state = IN_POINT
        
        elif self.state == IN_POINT:
            self._manage_rally(frame_data)
        
    def _manage_rally(self, data):
        self.physics.update(data.ball_x, data.ball_y)

        # Detect new shot
        if data.is_contact:
            self.context.active_shot.is_error = False
            self.context.active_shot.is_winner = False

            if self.context.active_shot:
                self.context.close_current_shot()
            
            shot_type = self.classify_shot(data) # only forehand or backhand for now
            new_shot = Shot(
                player_id=data.hitter_id,
                shot_type=shot_type,
                start_frame=data.frame_number
            )
            self.context.active_shot = new_shot
        
        # Detect peak
        peak = self.physics.detect_peak()
        if peak != None:
            if self.context.active_shot:
                self.context.active_shot.peak_height_px = peak[1] # y-coord of peak in px
        
        # Detect bounce
        bounce = self.physics.detect_bounce()
        if bounce != None:
            self._process_bounce(bounce)
        
    def _process_bounce(self, bounce_coords):
        # TODO: Implement bounce processing logic (in/out-of-court, map to BEV, check for double bounce)
        pass

    def _end_point(self, is_winner):
        self.state = SAVING

        self.context.current_shot.is_winner = is_winner
        self.context.current_shot.is_error = not is_winner
        self.context.close_current_shot()
        
        # TODO: save self.context.shots to DB

        self.state = WAITING