class PointContext:
    def __init__(self):
        self.shots = []
        self.active_shot = None
    
    def add_shot(self, shot):
        self.shots.append(shot)