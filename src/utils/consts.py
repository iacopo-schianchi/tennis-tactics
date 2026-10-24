from enum import Enum

# dimensions in meters
COURT_LENGTH = 23.77
COURT_WIDTH = 10.97


class COLORS_BGR(Enum):
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)
    YELLOW = (0, 255, 255)
    WHITE = (255, 255, 255)
    NEAR_PLAYER = (44, 230, 161)
    FAR_PLAYER = (127, 0, 255)
    BALL = (255, 217, 0)
    HIT_BALL = (247, 118, 72)
