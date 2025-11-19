from emotion_processor.emotions_recognition.features.weights_emotion_score import WeightedEmotionScore

class AnxietyScore(WeightedEmotionScore):
    def __init__(self):
        super().__init__(eyebrows_weight=0.35, eyes_weight=0.35, nose_weight=0.1, mouth_weight=0.2)

    def calculate_eyebrows_score(self, eyebrows_result: dict) -> float:
        score = 0.0
        score += 50 * eyebrows_result.get('together', 0.0)  # 0-50 por cercanía
        score += 25 * eyebrows_result.get('right_raised', 0.0)  # Grado de raise derecho
        score += 25 * eyebrows_result.get('left_raised', 0.0)
        return min(100.0, score)  # Cap a 100

    def calculate_eyes_score(self, eyes_result: dict) -> float:
        openness = eyes_result.get('openness', 0.0)
        if openness > 0.6:  # Umbral para 'abiertos' (calibra)
            return 70 + 30 * (openness - 0.6) / 0.4  # Gradual de 70-100
        return 0.0

    def calculate_nose_score(self, nose_result: dict) -> float:
        flared = nose_result.get('flared', 0.0)
        if flared > 0.1:  # Umbral para dilatación
            return 100 * min(1.0, flared / 0.3)  # Gradual
        return 0.0  # No neutral, sino dilatada

    def calculate_mouth_score(self, mouth_result: dict) -> float:
        score = 0.0
        tension = mouth_result.get('tension', 0.0)
        no_smile = mouth_result.get('no_smile', 0.0)  # 1 si curvatura neutra/tensa
        score += 50 * tension  # Grado de tensión
        score += 50 * no_smile
        return min(100.0, score)

