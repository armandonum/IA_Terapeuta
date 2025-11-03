from emotion_processor.emotions_recognition.features.weights_emotion_score import WeightedEmotionScore


class SadScore(WeightedEmotionScore):
    def __init__(self):
        super().__init__(eyebrows_weight=0.30, eyes_weight=0.30, nose_weight=0.1, mouth_weight=0.3)

    def calculate_eyebrows_score(self, eyebrows_result: dict) -> float:
        score = 0.0
        score += 60 * eyebrows_result.get('together', 0.0)  # Cejas juntas (AU4)
        score += 20 * (1.0 - eyebrows_result.get('right_raised', 0.0))  # Cejas bajas (inverso de raised)
        score += 20 * (1.0 - eyebrows_result.get('left_raised', 0.0))
        return min(100.0, score)

    def calculate_eyes_score(self, eyes_result: dict) -> float:
        openness = eyes_result.get('openness', 0.0)
        if openness < 0.4:  # Ojos cerrados o bajos
            return 100 * min(1.0, (0.4 - openness) / 0.4)  # Gradual
        return 0.0

    def calculate_nose_score(self, nose_result: dict) -> float:
        flared = nose_result.get('flared', 0.0)
        return 100 * (1.0 - flared) if flared < 0.1 else 0.0  # Nariz neutral

    def calculate_mouth_score(self, mouth_result: dict) -> float:
        score = 0.0
        tension = mouth_result.get('tension', 0.0)
        no_smile = mouth_result.get('no_smile', 0.0)
        score += 30 * tension  # Boca cerrada/tensa (AU17)
        score += 70 * no_smile  # Sin sonrisa (AU15)
        return min(100.0, score)