from emotion_processor.emotions_recognition.features.weights_emotion_score import WeightedEmotionScore


class HappyScore(WeightedEmotionScore):
    """
    Detección MEJORADA de FELICIDAD con MÁXIMA sensibilidad:
    - Sonrisa genuina (AU12 - Duchenne smile)
    - Ojos ligeramente entrecerrados por sonrisa (AU6)
    - Cejas relajadas
    - Boca relajada con comisuras hacia arriba
    """
    def __init__(self):
        # La boca es EL indicador principal de felicidad
        super().__init__(eyebrows_weight=0.08, eyes_weight=0.27, nose_weight=0.05, mouth_weight=0.60)

    def calculate_eyebrows_score(self, eyebrows_result: dict) -> float:
        """
        Felicidad: cejas relajadas y separadas
        """
        score = 0.0
        together = eyebrows_result.get('together', 0.0)
        right_raised = eyebrows_result.get('right_raised', 0.0)
        left_raised = eyebrows_result.get('left_raised', 0.0)
        
        # Cejas separadas (no juntas) - MÁS sensible
        if together < 0.3:  # Umbral más generoso
            score += 60 * (1.0 - together)
        
        # Cejas bajas o neutras (no levantadas)
        if right_raised < 0.4 and left_raised < 0.4:  # Umbral más generoso
            score += 20 * (1.0 - right_raised)
            score += 20 * (1.0 - left_raised)
        
        return min(100.0, score)

    def calculate_eyes_score(self, eyes_result: dict) -> float:
        """
        Felicidad: ojos ligeramente cerrados por sonrisa (AU6)
        Sonrisa genuina = mejillas suben y ojos se entrecierran
        MUCHO MÁS sensible
        """
        openness = eyes_result.get('openness', 0.0)
        
        # Rango MUY amplio: 0.15-0.85 (cualquier entrecerramiento ligero)
        if 0.15 <= openness <= 0.85:
            # Distribución gaussiana centrada en 0.5
            # Pico en 0.5 (ojos ligeramente entrecerrados)
            distance_from_optimal = abs(openness - 0.5)
            
            # Curva muy suave - acepta rango amplio
            if distance_from_optimal <= 0.35:  # Rango amplio de aceptación
                score = 100 * (1.0 - (distance_from_optimal / 0.35) ** 0.5)  # Raíz para suavizar
                
                # BONUS: Si está en el rango ideal (0.3-0.7)
                if 0.3 <= openness <= 0.7:
                    score = min(100.0, score * 1.2)  # Boost del 20%
                
                return max(0.0, min(100.0, score))
        
        # Fuera del rango óptimo pero aún válido
        if 0.1 <= openness < 0.15:
            return 40.0  # Algo de puntaje
        if 0.85 < openness <= 0.95:
            return 50.0  # Algo de puntaje
        
        return 0.0

    def calculate_nose_score(self, nose_result: dict) -> float:
        """
        Nariz neutra en felicidad (no arrugada)
        """
        flared = nose_result.get('flared', 0.0)
        
        # Nariz debe estar relajada (bajo flared)
        if flared < 0.2:  # Umbral generoso
            return 100 * (1.0 - flared / 0.2)
        
        return 0.0

    def calculate_mouth_score(self, mouth_result: dict) -> float:
        """
        SONRISA - EL INDICADOR MÁS IMPORTANTE
        MUY sensible para captar cualquier sonrisa
        """
        score = 0.0
        tension = mouth_result.get('tension', 0.0)
        no_smile = mouth_result.get('no_smile', 0.0)
        
        # Calcular "smile_score" (inverso de no_smile)
        smile_score = 1.0 - no_smile
        
        # PARTE 1: Boca relajada (no muy tensa)
        if tension < 0.7:  # Umbral muy generoso
            relaxation_score = 1.0 - tension
            score += 40 * relaxation_score
            
            # BONUS: Si muy relajada (sonrisa natural)
            if tension < 0.4:
                score += 10  # Bonus extra
        
        # PARTE 2: SONRISA (lo más importante)
        # MUY sensible: cualquier indicación de sonrisa
        if smile_score > 0.2:  # Umbral MUY bajo
            # Escala exponencial para recompensar sonrisas claras
            smile_contribution = 70 * (smile_score ** 0.7)  # Raíz para ser más generoso
            score += smile_contribution
            
            # BONUS PROGRESIVO según claridad de sonrisa
            if smile_score > 0.4:
                score += 10  # Bonus 1
            if smile_score > 0.6:
                score += 10  # Bonus 2
            if smile_score > 0.8:
                score += 10  # Bonus 3 - sonrisa muy clara
        
        return min(100.0, score)
