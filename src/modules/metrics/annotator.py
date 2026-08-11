import cv2
from utils.consts import COLORS_BGR

class MetricAnnotator:
    HIT_EVENT_QUEUE = 3

    def draw(self, frame, frame_id, context):
        past_hit_context = []

        i = frame_id
        while i >= 0 and len(past_hit_context) < self.HIT_EVENT_QUEUE:
            is_hit = context[i].get('event', {}).get('is_hit', False)
            if is_hit:
                past_hit_context.append(context[i])

            i -= 1

        y = 30

        for hit_context in past_hit_context:
            shot_type = hit_context['shot_type']
            shot_name = 'Unnknown'
            if shot_type is None: shot_name = shot_type.name
            peak = hit_context['peak']
            speed = hit_context['speed']
            hit_text = f'{shot_name} (~{speed} km/h, ~{peak}m peak)'

            cv2.putText(
                frame,
                hit_text,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                COLORS_BGR.WHITE.value,
                2,
                cv2.LINE_AA
            )
            y += 28

        return frame