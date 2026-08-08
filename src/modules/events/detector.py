class EventDetector:
    def process(self, frames, _frame_id, _context):
        # TODO

        no_event_prob = 1
        hit_prob = 0
        bounce_prob = 0

        event = 'hit' if hit_prob > bounce_prob and hit_prob > no_event_prob else 'bounce'
        event = None if no_event_prob > hit_prob and no_event_prob > bounce_prob else event

        #  TODO: keep only 1 detection if multiple consecutive (max prob)

        return {"events": {
            "no_event_prob": no_event_prob,
            "hit_prob": hit_prob,
            "is_hit": event == 'hit',
            "bounce_prob": bounce_prob,
            "is_bounce": event == 'bounce'
        }}