"""
Analizador especializado de ansiedad que combina múltiples emociones
y detecta patrones específicos de ansiedad
"""
from typing import Dict, Tuple
import numpy as np

class AnxietyAnalyzer:
    """
    Analiza ansiedad considerando:
    - Ansiedad directa (score propio)
    - Miedo prolongado
    - Tristeza + tensión
    - Cambios rápidos de emoción
    """
    
    def __init__(self):
        self.anxiety_history = []
        self.max_history = 30  # 30 frames (~1-2 segundos a 20fps)
        
    def analyze_anxiety_level(self, emotions: Dict[str, float]) -> Dict:
        """
        Analiza el nivel de ansiedad actual combinando múltiples indicadores
        
        Returns:
            {
                'anxiety_score': float (0-100),
                'anxiety_level': str ('baja', 'media', 'alta', 'muy_alta'),
                'confidence': float (0-100),
                'contributing_emotions': list,
                'recommendations': str
            }
        """
        # Extraer scores relevantes
        anxiety_direct = emotions.get('anxiety', 0.0)
        fear = emotions.get('fear', 0.0)
        sad = emotions.get('sad', 0.0)
        angry = emotions.get('angry', 0.0)
        happy = emotions.get('happy', 0.0)
        
        # Calcular score compuesto de ansiedad
        anxiety_score = self._calculate_composite_anxiety(
            anxiety_direct, fear, sad, angry, happy
        )
        
        # Agregar a historial
        self.anxiety_history.append(anxiety_score)
        if len(self.anxiety_history) > self.max_history:
            self.anxiety_history.pop(0)
        
        # Calcular tendencia
        trend = self._calculate_trend()
        
        # Determinar nivel
        level, confidence = self._determine_anxiety_level(anxiety_score, trend)
        
        # Identificar emociones contribuyentes
        contributing = self._identify_contributing_emotions(emotions)
        
        # Generar recomendaciones
        recommendations = self._generate_recommendations(level, contributing)
        
        return {
            'anxiety_score': round(anxiety_score, 2),
            'anxiety_level': level,
            'confidence': round(confidence, 2),
            'trend': trend,
            'contributing_emotions': contributing,
            'recommendations': recommendations
        }
    
    def _calculate_composite_anxiety(self, anxiety: float, fear: float, 
                                     sad: float, angry: float, happy: float) -> float:
        """
        Calcula score compuesto de ansiedad usando múltiples emociones
        """
        # Peso base de ansiedad directa
        score = anxiety * 0.50
        
        # Miedo contribuye fuertemente
        score += fear * 0.30
        
        # Tristeza contribuye moderadamente
        score += sad * 0.15
        
        # Enojo bajo tensión puede ser ansiedad
        score += angry * 0.10
        
        # Felicidad reduce ansiedad
        score -= happy * 0.05
        
        return max(0.0, min(100.0, score))
    
    def _calculate_trend(self) -> str:
        """
        Calcula tendencia de ansiedad (aumentando, estable, disminuyendo)
        """
        if len(self.anxiety_history) < 10:
            return 'insuficiente_data'
        
        recent = self.anxiety_history[-5:]
        older = self.anxiety_history[-10:-5]
        
        avg_recent = np.mean(recent)
        avg_older = np.mean(older)
        
        diff = avg_recent - avg_older
        
        if diff > 10:
            return 'aumentando'
        elif diff < -10:
            return 'disminuyendo'
        else:
            return 'estable'
    
    def _determine_anxiety_level(self, score: float, trend: str) -> Tuple[str, float]:
        """
        Determina nivel de ansiedad y confianza
        """
        # Ajustar umbrales basados en tendencia
        if trend == 'aumentando':
            score *= 1.1  # Aumenta sensibilidad
        elif trend == 'disminuyendo':
            score *= 0.9  # Reduce sensibilidad
        
        # Calcular confianza basada en historial
        if len(self.anxiety_history) >= 10:
            std = np.std(self.anxiety_history[-10:])
            confidence = max(50.0, 100.0 - std)  # Más estable = más confianza
        else:
            confidence = 50.0
        
        # Determinar nivel
        if score < 25:
            return 'baja', confidence
        elif score < 45:
            return 'media', confidence
        elif score < 70:
            return 'alta', confidence
        else:
            return 'muy_alta', confidence
    
    def _identify_contributing_emotions(self, emotions: Dict[str, float]) -> list:
        """
        Identifica qué emociones están contribuyendo más a la ansiedad
        """
        contributors = []
        
        if emotions.get('anxiety', 0) > 30:
            contributors.append('ansiedad_directa')
        if emotions.get('fear', 0) > 30:
            contributors.append('miedo')
        if emotions.get('sad', 0) > 30:
            contributors.append('tristeza')
        if emotions.get('angry', 0) > 30:
            contributors.append('tensión/enojo')
        
        return contributors if contributors else ['no_identificado']
    
    def _generate_recommendations(self, level: str, contributing: list) -> str:
        """
        Genera recomendaciones basadas en nivel de ansiedad
        """
        recommendations = {
            'baja': 'Estado emocional estable. Continúa con actividades normales.',
            'media': 'Ansiedad leve detectada. Considera técnicas de respiración o pausas breves.',
            'alta': 'Ansiedad significativa detectada. Recomendado: ejercicios de relajación, respiración profunda, o hablar con alguien.',
            'muy_alta': 'Ansiedad muy alta detectada. Recomendado: buscar apoyo inmediato, técnicas de grounding, o contactar con profesional de salud mental.'
        }
        
        return recommendations.get(level, 'Monitorear estado emocional.')
    
    def get_anxiety_summary(self) -> Dict:
        """
        Obtiene resumen estadístico de ansiedad en el historial
        """
        if not self.anxiety_history:
            return {
                'mean': 0.0,
                'max': 0.0,
                'min': 0.0,
                'current': 0.0
            }
        
        return {
            'mean': round(np.mean(self.anxiety_history), 2),
            'max': round(np.max(self.anxiety_history), 2),
            'min': round(np.min(self.anxiety_history), 2),
            'current': round(self.anxiety_history[-1], 2)
        }