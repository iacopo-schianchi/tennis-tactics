from collections import deque

class Trajectory:
    def __init__(self, buffer_size=10):
        self.buffer = deque(maxlen=buffer_size) # (x, y)
    
    def update(self, x, y):
        self.buffer.append((x, y))

    def detect_peak(self):
        if len(self.buffer) < 3:
            return None
        
        # TODO: improve from basic midpoint (e.g., ML model by y and maybe x positions)
        _x1, y1 = self.buffer[-3]
        x2, y2 = self.buffer[-2]
        _x3, y3 = self.buffer[-1]

        if y2 > y1 and y2 > y3:
            return (x2, y2)
        return None

    def detect_bounce(self):
        if len(self.buffer) < 3:
            return None
        
        # TODO: improve from basic midpoint
        x1, y1 = self.buffer[-3]
        x2, y2 = self.buffer[-2]
        x3, y3 = self.buffer[-1]

        # TODO: convert to BEV (x, y) relative to court
        if y2 < y1 and y2 < y3:
            return (x2, y2)
        return None