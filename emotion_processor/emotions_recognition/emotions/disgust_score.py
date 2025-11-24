from emotion_processor.emotions_recognition.features.weights_emotion_score import WeightedEmotionScore


class DisgustScore(WeightedEmotionScore):
    """
    Detección ESTRICTA de DISGUSTO/ASCO:
    - Nariz arrugada (AU9) - MUY importante
    - Labio superior levantado (AU10)
    - Ojos muy entrecerrados (AU6/AU7)
    - Cejas bajas y juntas (AU4)
    
    DEBE SER MUY ESTRICTO para no confundirse con felicidad
    """
    def __init__(self):
        # La NARIZ es el indicador principal de disgusto
        super().__init__(eyebrows_weight=0.05, eyes_weight=0.20, nose_weight=0.40, mouth_weight=0.10)

    def calculate_eyebrows_score(self, eyebrows_result: dict) -> float:
        """
        Disgusto: Cejas bajas y juntas (pero no tanto como en enojo)
        """
        score = 0.0
        together = eyebrows_result.get('together', 0.0)
        right_raised = eyebrows_result.get('right_raised', 0.0)
        left_raised = eyebrows_result.get('left_raised', 0.0)
        
        # Cejas juntas - UMBRAL ALTO (debe ser muy evidente)
        if together > 0.7:  # MUY estricto
            score += 50 * ((together - 0.7) / 0.3)
        
        # Cejas bajas (no elevadas) - UMBRAL ESTRICTO
        if right_raised < 0.2 and left_raised < 0.2:  # Ambas deben estar bajas
            score += 25 * (1.0 - right_raised)
            score += 25 * (1.0 - left_raised)
        
        return min(100.0, score)

    def calculate_eyes_score(self, eyes_result: dict) -> float:
        """
        Disgusto: Ojos MUY entrecerrados (más que en felicidad)
        AU6 (cheek raiser) + AU7 (lid tightener)
        """
        openness = eyes_result.get('openness', 0.0)
        tension = eyes_result.get('tension', 0.0)
        score = 0.0
        
        # Ojos MUY cerrados (más que en sonrisa) - UMBRAL MUY BAJO
        if openness < 0.15:  # Extremadamente cerrados
            score += 60 * (0.15 - openness) / 0.15
        
        # Tensión MUY alta en párpados - UMBRAL ALTO
        if tension > 0.7:  # Tensión muy evidente
            score += 40 * ((tension - 0.7) / 0.3)
        
        return min(100.0, score)

    def calculate_nose_score(self, nose_result: dict) -> float:
        """
        NARIZ ARRUGADA - EL INDICADOR MÁS IMPORTANTE DE DISGUSTO
        AU9 (nose wrinkler) - Debe ser MUY evidente
        """
        flared = nose_result.get('flared', 0.0)
        
        # UMBRAL MUY ALTO - La nariz DEBE estar muy arrugada
        if flared > 0.5:  # Umbral extremadamente alto
            # Escala exponencial: más arrugada = mucho más puntaje
            normalized = (flared - 0.5) / 0.5  # 0.5-1.0 -> 0-1
            score = 100 * min(1.0, normalized ** 0.7)  # Exponencial para ser más estricto
            return score
        
        # Si no pasa el umbral alto, retorna 0
        return 0.0

    def calculate_mouth_score(self, mouth_result: dict) -> float:
        """
        Disgusto: Boca fruncida, labio superior levantado (AU10)
        Muy diferente de sonrisa
        """
        score = 0.0
        tension = mouth_result.get('tension', 0.0)
        no_smile = mouth_result.get('no_smile', 0.0)
        
        # Boca MUY tensa y fruncida - UMBRAL ALTO
        if tension > 0.8:  # Extremadamente tensa
            score += 60 * ((tension - 0.8) / 0.2)
        
        # NO sonrisa - UMBRAL MUY ALTO
        if no_smile > 0.9:  # Casi sin sonrisa
            score += 40 * ((no_smile - 0.9) / 0.1)
        
        return min(100.0, score)