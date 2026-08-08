import numpy as np
import joblib
from tensorflow.keras.models import load_model

class EventDetector:
    window_size = 21
    FEATURE_COLS = [
        "x", "y", "is_missing", "vx", "vy", "ax", "ay",
        "speed", "accel", "angle", "delta_angle",
    ]

    def __init__(self):
        self.model = load_model('src/models/bilstm_event_classifier_small.keras')
        self.scaler = joblib.load('src/models/bilstm_event_scaler_small.pkl')
        self.predictions_cache = None

    def process(self, _frames, frame_id, context):
        if self.predictions_cache is None:
            self._run_sequence_inference(context) # run once for clip

        return {"event": self.predictions_cache[frame_id]} # populate per frame

    def _run_sequence_inference(self, context):
        n_total = len(context)

        feature_list = []
        for frame_ctx in context:
            ball = frame_ctx.get("ball", {})

            row = [
                float(ball.get(feat)) if ball.get(feat) is not None else 0.0 for feat in self.FEATURE_COLS
            ]
            feature_list.append(row)

        X_raw = np.array(feature_list, dtype=np.float32)

        X_scaled = self.scaler.transform(X_raw)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0)

        prob_sum = np.zeros((n_total, 3), dtype=np.float64)
        overlap_count = np.zeros(n_total, dtype=np.float64)

        # clip too short
        if n_total < self.window_size:
            pad_len = self.window_size - n_total

            X_padded = np.pad(
                X_scaled, ((0, pad_len), (0, 0)), mode="constant", constant_values=0.0
            )

            batch_input = np.expand_dims(X_padded, axis=0)
            preds = self.model.predict(batch_input, verbose=0)[0][:n_total]

            prob_sum += preds
            overlap_count += 1
        else:
            starts = list(range(0, n_total - self.window_size + 1, 1))
            X_windows = np.stack([X_scaled[s : s + self.window_size] for s in starts])
            preds = self.model.predict(X_windows, verbose=0)

            for w_i, s in enumerate(starts):
                prob_sum[s : s + self.window_size] += preds[w_i]
                overlap_count[s : s + self.window_size] += 1

        overlap_count[overlap_count == 0] = 1
        avg_probs = prob_sum / overlap_count[:, None]
        predicted_status = np.argmax(avg_probs, axis=1)

        # TODO: keep only peak for continuous hit/bounce events (max prob)

        self.predictions_cache = []
        for frame in range(n_total):
            self.predictions_cache.append({
                "status": int(predicted_status[frame]), # 0=none, 1=hit, 2=bounce
                "no_event_prob": float(avg_probs[frame][0]),
                "hit_prob": float(avg_probs[frame][1]),
                "is_hit": predicted_status[frame] == 1,
                "bounce_prob": float(avg_probs[frame][2]),
                "is_bounce": predicted_status[frame] == 2
            })