from emotion_processor.emotions_recognition.features.weights_emotion_score import WeightedEmotionScore


class HappyScore(WeightedEmotionScore):
    """
    Detección mejorada de FELICIDAD con mayor sensibilidad:
    - Sonrisa genuina (AU12 - Duchenne smile)
    - Ojos entrecerrados (AU6)
    - Cejas relajadas
    """
    def __init__(self):
        # La boca es EL indicador principal de felicidad
        super().__init__(eyebrows_weight=0.10, eyes_weight=0.25, nose_weight=0.05, mouth_weight=0.60)

    def calculate_eyebrows_score(self, eyebrows_result: dict) -> float:
        """
        Felicidad: cejas relajadas y separadas
        """
        score = 0.0
        together = eyebrows_result.get('together', 0.0)
        right_raised = eyebrows_result.get('right_raised', 0.0)
        left_raised = eyebrows_result.get('left_raised', 0.0)
        
        # Cejas separadas (no juntas)
        score += 50 * (1.0 - together)
        
        # Cejas bajas o neutras (no levantadas)
        score += 25 * (1.0 - right_raised)
        score += 25 * (1.0 - left_raised)
        
        return min(100.0, score)

    def calculate_eyes_score(self, eyes_result: dict) -> float:
        """
        Felicidad: ojos ligeramente cerrados por sonrisa (AU6)
        Sonrisa genuina = mejillas suben y ojos se entrecierran
        """
        openness = eyes_result.get('openness', 0.0)
        
        # Rango óptimo: 0.3-0.7 (ni muy abiertos ni cerrados)
        # MÁS sensible a sonrisas genuinas
        if 0.2 <= openness <= 0.8:  # Rango ampliado
            # Pico en 0.5 (ojos ligeramente entrecerrados)
            distance_from_optimal = abs(openness - 0.5)
            score = 100 * (1.0 - distance_from_optimal / 0.3)
            return max(0.0, min(100.0, score))
        
        return 0.0

    def calculate_nose_score(self, nose_result: dict) -> float:
        """
        Nariz neutra en felicidad
        """
        flared = nose_result.get('flared', 0.0)
        if flared < 0.1:
            return 100 * (1.0 - flared / 0.1)
        return 0.0

    def calculate_mouth_score(self, mouth_result: dict) -> float:
        """
        Felicidad: SONRISA (AU12) - MUCHO MÁS sensible
        Este es el indicador principal
        """
        score = 0.0
        tension = mouth_result.get('tension', 0.0)
        no_smile = mouth_result.get('no_smile', 0.0)
        
        # Boca relajada (no tensa) - MÁS sensible
        if tension < 0.6:  # Umbral aumentado
            score += 50 * (1.0 - tension)
        
        # SONRISA - MÁS sensible y más peso
        # no_smile bajo = sonrisa alta
        if no_smile < 0.7:  # Umbral muy generoso
            smile_score = 1.0 - no_smile
            score += 70 * smile_score  # MÁS peso a sonrisa
            
            # BONUS: Si sonrisa muy clara
            if smile_score > 0.6:
                score += 30  # Bonus extra
        
        return min(100.0, score)