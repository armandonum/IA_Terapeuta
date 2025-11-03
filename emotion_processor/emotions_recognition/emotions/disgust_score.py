from emotion_processor.emotions_recognition.features.weights_emotion_score import WeightedEmotionScore


class DisgustScore(WeightedEmotionScore):
    def __init__(self):
        super().__init__(eyebrows_weight=0.25, eyes_weight=0.25, nose_weight=0.35, mouth_weight=0.15)

    def calculate_eyebrows_score(self, eyebrows_result: dict) -> float:
        score = 0.0
        together = eyebrows_result.get('together', 0.0)
        right_raised = eyebrows_result.get('right_raised', 0.0)
        left_raised = eyebrows_result.get('left_raised', 0.0)
        score += 33.33 * together if together > 0.5 else 0.0  # Cejas juntas (AU4), umbral estricto
        score += 33.33 * (1.0 - right_raised) if right_raised < 0.3 else 0.0  # Cejas bajas
        score += 33.33 * (1.0 - left_raised) if left_raised < 0.3 else 0.0
        return min(100.0, score)

    def calculate_eyes_score(self, eyes_result: dict) -> float:
        openness = eyes_result.get('openness', 0.0)
        tension = eyes_result.get('tension', 0.0)
        score = 0.0
        if openness < 0.2:  # Ojos entrecerrados (AU6/AU7), umbral más bajo
            score += 50 + 50 * (0.2 - openness) / 0.2  # Gradual
        score += 50 * tension if tension > 0.5 else 0.0  # Tensión alta
        return min(100.0, score)

    def calculate_nose_score(self, nose_result: dict) -> float:
        flared = nose_result.get('flared', 0.0)
        if flared > 0.3:  # Nariz arrugada (AU9), umbral más alto
            return 100 * min(1.0, (flared - 0.3) / 0.2)  # Gradual, rango 0.3-0.5
        return 0.0

    def calculate_mouth_score(self, mouth_result: dict) -> float:
        score = 0.0
        tension = mouth_result.get('tension', 0.0)
        no_smile = mouth_result.get('no_smile', 0.0)
        score += 50 * tension if tension > 0.6 else 0.0  # Boca fruncida (AU10/AU17)
        score += 50 * no_smile if no_smile > 0.7 else 0.0  # Sin sonrisa (AU15)
        return min(100.0, score)