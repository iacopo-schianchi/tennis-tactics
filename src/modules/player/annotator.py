import cv2
from ...utils.consts import COLORS_BGR

class PlayerAnnotator:
    def draw(self, frame, frame_id, context):
        players = context[frame_id]['players']

        for player in players.values():
            if player is None: continue

            x1, y1, x2, y2  = player['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS_BGR.RED.value, 2)

        return frame