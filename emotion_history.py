import json
from datetime import datetime
from typing import Dict, List
import numpy as np

class EmotionHistory:
    """
    Gestiona el historial de emociones durante la grabación.
    """
    
    def __init__(self):
        self.recording = False
        self.history: List[Dict] = []
        self.start_time = None
        self.end_time = None
        
    def start_recording(self):
        """Inicia la grabación del historial."""
        self.recording = True
        self.history = []
        self.start_time = datetime.now()
        print(f"📹 Historial de emociones iniciado: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
    def stop_recording(self):
        """Detiene la grabación del historial."""
        self.recording = False
        self.end_time = datetime.now()
        print(f"⏹️ Historial de emociones detenido: {self.end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
    def add_frame(self, emotions: Dict[str, float], frame_number: int):
        """
        Agrega un frame al historial si está grabando.
        
        Args:
            emotions: Diccionario con scores de emociones
            frame_number: Número de frame actual
        """
        if not self.recording:
            return
        
        timestamp = datetime.now()
        elapsed_seconds = (timestamp - self.start_time).total_seconds()
        
        self.history.append({
            'frame': frame_number,
            'timestamp': timestamp.isoformat(),
            'elapsed_seconds': round(elapsed_seconds, 2),
            'emotions': emotions.copy()
        })
    
    def get_summary(self) -> Dict:
        """
        Genera un resumen del historial para enviar a un LLM.
        
        Returns:
            Diccionario con análisis completo del historial
        """
        if not self.history:
            return {
                'status': 'no_data',
                'message': 'No hay datos de emociones registrados'
            }
        
        # Calcular estadísticas por emoción
        emotions_stats = self._calculate_emotion_statistics()
        
        # Detectar cambios significativos
        emotion_transitions = self._detect_transitions()
        
        # Generar timeline
        timeline = self._generate_timeline()
        
        # Calcular duración
        duration = (self.end_time - self.start_time).total_seconds() if self.end_time else 0
        
        summary = {
            'recording_info': {
                'start_time': self.start_time.strftime('%Y-%m-%d %H:%M:%S') if self.start_time else None,
                'end_time': self.end_time.strftime('%Y-%m-%d %H:%M:%S') if self.end_time else None,
                'duration_seconds': round(duration, 2),
                'total_frames': len(self.history)
            },
            'emotion_statistics': emotions_stats,
            'emotion_transitions': emotion_transitions,
            'timeline': timeline,
            'llm_summary': self._generate_llm_prompt()
        }
        
        return summary
    
    def _calculate_emotion_statistics(self) -> Dict:
        """Calcula estadísticas agregadas por emoción."""
        if not self.history:
            return {}
        
        emotion_names = list(self.history[0]['emotions'].keys())
        stats = {}
        
        for emotion in emotion_names:
            values = [frame['emotions'][emotion] for frame in self.history]
            stats[emotion] = {
                'mean': round(np.mean(values), 2),
                'max': round(np.max(values), 2),
                'min': round(np.min(values), 2),
                'std': round(np.std(values), 2),
                'dominant_percentage': self._calculate_dominance(emotion)
            }
        
        return stats
    
    def _calculate_dominance(self, emotion: str) -> float:
        """Calcula el porcentaje de frames donde esta emoción fue dominante."""
        if not self.history:
            return 0.0
        
        dominant_count = 0
        for frame in self.history:
            dominant = max(frame['emotions'], key=frame['emotions'].get)
            if dominant == emotion:
                dominant_count += 1
        
        return round((dominant_count / len(self.history)) * 100, 2)
    
    def _detect_transitions(self) -> List[Dict]:
        """Detecta cambios significativos de emoción."""
        if len(self.history) < 2:
            return []
        
        transitions = []
        previous_dominant = None
        transition_threshold = 20.0  # Cambio mínimo para considerar transición
        
        for i, frame in enumerate(self.history):
            current_dominant = max(frame['emotions'], key=frame['emotions'].get)
            current_score = frame['emotions'][current_dominant]
            
            if previous_dominant and current_dominant != previous_dominant:
                if current_score > transition_threshold:
                    transitions.append({
                        'frame': frame['frame'],
                        'time': frame['elapsed_seconds'],
                        'from': previous_dominant,
                        'to': current_dominant,
                        'confidence': round(current_score, 2)
                    })
            
            if current_score > transition_threshold:
                previous_dominant = current_dominant
        
        return transitions
    
    def _generate_timeline(self, segments: int = 10) -> List[Dict]:
        """Genera un timeline segmentado del historial."""
        if not self.history:
            return []
        
        segment_size = max(1, len(self.history) // segments)
        timeline = []
        
        for i in range(0, len(self.history), segment_size):
            segment = self.history[i:i + segment_size]
            if not segment:
                continue
            
            # Calcular emoción dominante en el segmento
            emotion_sums = {}
            for frame in segment:
                for emotion, score in frame['emotions'].items():
                    emotion_sums[emotion] = emotion_sums.get(emotion, 0) + score
            
            dominant = max(emotion_sums, key=emotion_sums.get)
            avg_score = emotion_sums[dominant] / len(segment)
            
            timeline.append({
                'segment': i // segment_size + 1,
                'start_time': segment[0]['elapsed_seconds'],
                'end_time': segment[-1]['elapsed_seconds'],
                'dominant_emotion': dominant,
                'confidence': round(avg_score, 2)
            })
        
        return timeline
    
    def _generate_llm_prompt(self) -> str:
        """Genera un prompt optimizado para análisis por LLM."""
        if not self.history:
            return "No hay datos para analizar."
        
        stats = self._calculate_emotion_statistics()
        transitions = self._detect_transitions()
        timeline = self._generate_timeline()
        
        # Encontrar emoción dominante general
        dominant_emotion = max(stats.items(), key=lambda x: x[1]['mean'])
        
        prompt = f"""Análisis de Emociones Faciales:

Duración: {(self.end_time - self.start_time).total_seconds():.1f} segundos
Frames analizados: {len(self.history)}

EMOCIÓN DOMINANTE: {dominant_emotion[0].upper()} ({dominant_emotion[1]['mean']:.1f}% promedio)

ESTADÍSTICAS POR EMOCIÓN:
"""
        for emotion, values in sorted(stats.items(), key=lambda x: x[1]['mean'], reverse=True):
            prompt += f"- {emotion.capitalize()}: Promedio {values['mean']:.1f}%, Máximo {values['max']:.1f}%, Dominancia {values['dominant_percentage']:.1f}%\n"
        
        if transitions:
            prompt += f"\nTRANSICIONES DETECTADAS ({len(transitions)}):\n"
            for trans in transitions[:5]:  # Primeras 5 transiciones
                prompt += f"- Segundo {trans['time']:.1f}s: {trans['from']} → {trans['to']} (confianza {trans['confidence']:.1f}%)\n"
        
        prompt += "\nLÍNEA DE TIEMPO:\n"
        for seg in timeline:
            prompt += f"- Seg {seg['segment']} ({seg['start_time']:.1f}s-{seg['end_time']:.1f}s): {seg['dominant_emotion']} ({seg['confidence']:.1f}%)\n"
        
        return prompt
    
    def save_to_file(self, filename: str = None):
        """Guarda el historial en un archivo JSON."""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"emotion_history_{timestamp}.json"
        
        summary = self.get_summary()
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Historial guardado en: {filename}")
        return filename