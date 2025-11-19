import numpy as np
from typing import Dict

class EmotionNormalizer:
    """
    Normaliza las emociones considerando relaciones y conflictos entre ellas.
    Emociones opuestas se penalizan mutuamente.
    """
    
    def __init__(self):
        # Define pares de emociones opuestas (se penalizan entre sí)
        self.opposite_pairs = [
            ('happy', 'sad'),
            ('happy', 'angry'),
            ('happy', 'anxiety'),  # Felicidad vs ansiedad
            ('happy', 'fear'),
            ('surprise', 'disgust'),
        ]
        
        # Define grupos de emociones similares (se refuerzan)
        self.similar_groups = [
            ['fear', 'anxiety'],          # Miedo y ansiedad se refuerzan
            ['sad', 'anxiety'],           # Tristeza puede indicar ansiedad
            ['angry', 'disgust'],         # Emociones negativas activas
        ]
        
    def normalize(self, emotions: Dict[str, float]) -> Dict[str, float]:
        """
        Normaliza las emociones aplicando penalizaciones y refuerzos.
        
        Args:
            emotions: Diccionario con scores de emociones (0-100)
            
        Returns:
            Diccionario con scores normalizados
        """
        normalized = emotions.copy()
        
        # Paso 1: Aplicar penalizaciones por emociones opuestas
        for emotion1, emotion2 in self.opposite_pairs:
            if emotion1 in normalized and emotion2 in normalized:
                score1 = normalized[emotion1]
                score2 = normalized[emotion2]
                
                # Si ambas están altas, penaliza la menor
                if score1 > 30 and score2 > 30:
                    penalty_factor = min(score1, score2) / 100.0
                    if score1 > score2:
                        normalized[emotion2] *= (1 - penalty_factor * 0.7)
                    else:
                        normalized[emotion1] *= (1 - penalty_factor * 0.7)
        
        # Paso 2: Reforzar emociones similares
        for group in self.similar_groups:
            group_scores = [normalized.get(e, 0) for e in group]
            if max(group_scores) > 50:  # Si una está alta
                avg_score = np.mean(group_scores)
                for emotion in group:
                    if emotion in normalized and normalized[emotion] > 20:
                        # Refuerzo leve hacia el promedio del grupo
                        normalized[emotion] += (avg_score - normalized[emotion]) * 0.2
        
        # Paso 3: Aplicar softmax suave para redistribuir probabilidades
        normalized = self._soft_competition(normalized)
        
        # Paso 4: Asegurar que estén en rango [0, 100]
        for emotion in normalized:
            normalized[emotion] = max(0.0, min(100.0, normalized[emotion]))
        
        return normalized
    
    def _soft_competition(self, emotions: Dict[str, float]) -> Dict[str, float]:
        """
        Aplica competencia suave entre emociones para que la suma tienda a ser más coherente.
        No es un softmax estricto, sino una redistribución suave.
        """
        total = sum(emotions.values())
        if total == 0:
            return emotions
        
        # Normalizar para que la suma sea aproximadamente 100-150 (rango razonable)
        target_sum = 120.0
        if total > target_sum * 1.5:  # Si la suma es muy alta
            factor = target_sum / total
            return {k: v * factor for k, v in emotions.items()}
        
        return emotions
    
    def get_dominant_emotion(self, emotions: Dict[str, float]) -> tuple:
        """
        Retorna la emoción dominante y su confianza.
        
        Returns:
            (emotion_name, confidence_score)
        """
        if not emotions:
            return ("neutral", 0.0)
        
        dominant = max(emotions, key=emotions.get)
        confidence = emotions[dominant]
        
        return (dominant, confidence)