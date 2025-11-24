from emotion_processor.emotions_recognition.features.weights_emotion_score import WeightedEmotionScore

class AngryScore(WeightedEmotionScore):
    def __init__(self):
        super().__init__(
            eyebrows_weight=0.70,
            eyes_weight=0.85,
            nose_weight=0.65,
            mouth_weight=0.25
        )

    # --------------------------
    # CEJAS — AU4 (principal)
    # --------------------------
    def calculate_eyebrows_score(self, eyebrows_result: dict) -> float:
        lowered = eyebrows_result.get('lowered', 0.0)        # AU4
        together = eyebrows_result.get('together', 0.0)      # brow distance
        score = (0.65 * lowered + 0.35 * together) * 100.0
        return min(100.0, score)

    # --------------------------
    # OJOS — AU7
    # --------------------------
    def calculate_eyes_score(self, eyes_result: dict) -> float:
        tightness = eyes_result.get('tightness', 0.0)  # AU7
        openness = eyes_result.get('openness', 0.0)

        score = 0.0

        # AU7 = tensión del párpado, muy importante
        score += tightness * 100.0 * 0.7

        # En ira, ojos *ligeramente* entrecerrados (pero no cerrados)
        if openness < 0.45:
            score += (0.45 - openness) / 0.45 * 100.0 * 0.3

        return min(100.0, score)

    # --------------------------
    # NARIZ — AU9 (wrinkle)
    # --------------------------
    def calculate_nose_score(self, nose_result: dict) -> float:
        wrinkle = nose_result.get('wrinkle', 0.0)  # AU9
        flare = nose_result.get('flare', 0.0)      # opcional AU38
        score = wrinkle * 100.0 * 0.8 + flare * 100.0 * 0.2
        return min(100.0, score)

    # --------------------------
    # BOCA — AU23/24/17
    # --------------------------
    def calculate_mouth_score(self, mouth_result: dict) -> float:
        press = mouth_result.get('press', 0.0)     # AU24
        tighten = mouth_result.get('tighten', 0.0) # AU23
        chin_raise = mouth_result.get('chin_raise', 0.0)  # AU17

        score = (
            0.5 * press +
            0.35 * tighten +
            0.15 * chin_raise
        ) * 100.0

        return min(100.0, score)
