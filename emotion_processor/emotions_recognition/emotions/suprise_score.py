from emotion_processor.emotions_recognition.features.weights_emotion_score import WeightedEmotionScore


class SurpriseScore(WeightedEmotionScore):
    def __init__(self):
        super().__init__(
            eyebrows_weight=0.80,
            eyes_weight=0.75,     # MÁS peso a los ojos (AU5)
            nose_weight=0.00,     # NO relevante en sorpresa
            mouth_weight=0.75
        )

    # --------------------------
    # CEJAS (AU1 + AU2)
    # --------------------------
    def calculate_eyebrows_score(self, eyebrows_result: dict) -> float:
        raised_left = eyebrows_result.get('left_raised', 0.0)
        raised_right = eyebrows_result.get('right_raised', 0.0)

        # Cejas levantadas → principal indicador
        score = (raised_left + raised_right) / 2.0 * 100.0
        return min(100.0, score)

    # --------------------------
    # OJOS (AU5)
    # --------------------------
    def calculate_eyes_score(self, eyes_result: dict) -> float:
        openness = eyes_result.get('openness', 0.0)

        # Sorpresa real: ojos muy abiertos
        if openness < 0.3:
            return 0.0
        if openness > 1.0:
            openness = 1.0

        return openness * 100.0

    # --------------------------
    # NARIZ (NO RELEVANTE)
    # --------------------------
    def calculate_nose_score(self, nose_result: dict) -> float:
        return 0.0   # No aporta

    # --------------------------
    # BOCA (AU25/26)
    # --------------------------
    def calculate_mouth_score(self, mouth_result: dict) -> float:
        openness = mouth_result.get('openness', 0.0)  # necesitas este valor en el extractor

        # Boca muy abierta → sorpresa
        if openness < 0.2:
            return 0.0

        return min(100.0, openness * 100.0)
