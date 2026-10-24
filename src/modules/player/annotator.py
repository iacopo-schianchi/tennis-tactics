import cv2
from utils.consts import COLORS_BGR

class PlayerAnnotator:
    def draw(self, frame, frame_id, context):
        players = context[frame_id]['players']

        for role, player in players.items():
            if player is None: continue

            x, y = player['feet_px']
            cv2.circle(frame, (x, y), 12, COLORS_BGR.NEAR_PLAYER.value if role == "near" else COLORS_BGR.FAR_PLAYER.value, -1)

        return frame