from emotion_processor.emotions_recognition.features.weights_emotion_score import WeightedEmotionScore

class AngryScore(WeightedEmotionScore):
    def __init__(self):
        super().__init__(eyebrows_weight=0.40, eyes_weight=0.25, nose_weight=0.1, mouth_weight=0.25)

    def calculate_eyebrows_score(self, eyebrows_result: dict) -> float:
        score = 0.0
        score += 50 * eyebrows_result.get('together', 0.0)  # Cejas juntas (AU4)
        # Para ira, cejas bajas (no elevadas)
        score += 25 * (1.0 - eyebrows_result.get('right_raised', 0.0))  # Inverso: bajo si no elevado
        score += 25 * (1.0 - eyebrows_result.get('left_raised', 0.0))
        return min(100.0, score)

    def calculate_eyes_score(self, eyes_result: dict) -> float:
        openness = eyes_result.get('openness', 0.0)
        tension = eyes_result.get('tension', 0.0)
        score = 0.0
        if openness < 0.4:  # Ojos entrecerrados (AU7, lid tightener), no completamente cerrados
            score += 50 + 50 * (0.4 - openness) / 0.4  # Gradual: más cerrados, más puntaje
        score += 50 * tension  # Tensión en párpados (AU7)
        return min(100.0, score)

    def calculate_nose_score(self, nose_result: dict) -> float:
        flared = nose_result.get('flared', 0.0)
        if flared > 0.1:  # Nariz arrugada o dilatada (AU9)
            return 100 * min(1.0, flared / 0.3)  # Gradual
        return 0.0

    def calculate_mouth_score(self, mouth_result: dict) -> float:
        score = 0.0
        tension = mouth_result.get('tension', 0.0)
        no_smile = mouth_result.get('no_smile', 0.0)
        score += 50 * tension  # Boca tensa (AU17, AU23, AU24)
        score += 50 * no_smile  # Sin sonrisa, esquinas neutras o hacia abajo
        return min(100.0, score)