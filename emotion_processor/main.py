import numpy as np
from emotion_processor.face_mesh.face_mesh_processor import FaceMeshProcessor
from emotion_processor.data_processing.main import PointsProcessing
from emotion_processor.emotions_recognition.main import EmotionRecognition
from emotion_normalizer import EmotionNormalizer
from emotion_history import EmotionHistory

class EmotionRecognitionSystem:
    def __init__(self):
        self.face_mesh = FaceMeshProcessor()
        self.data_processing = PointsProcessing()
        self.emotions_recognition = EmotionRecognition()
        
        # Nuevos componentes
        self.emotion_normalizer = EmotionNormalizer()
        self.emotion_history = EmotionHistory()
        
        self.emotion_history_list = []
        self.landmarks_history = []
        self.frame_count = 0

    def frame_processing(self, face_image: np.ndarray):
        # Procesar sin dibujar (draw=False)
        face_points, control_process, original_image = self.face_mesh.process(face_image, draw=False)
        
        if control_process:
            # Procesar características faciales
            processed_features = self.data_processing.main(face_points)
            
            # Reconocer emociones (scores crudos)
            raw_emotions = self.emotions_recognition.recognize_emotion(processed_features)
            
            # Normalizar emociones para evitar conflictos
            normalized_emotions = self.emotion_normalizer.normalize(raw_emotions)
            
            # Agregar al historial
            self.emotion_history_list.append(normalized_emotions)
            self.landmarks_history.append(face_points)
            self.frame_count += 1
            
            # Registrar en historial de grabación si está activo
            self.emotion_history.add_frame(normalized_emotions, self.frame_count)
            
            # Devolver la imagen original sin visualización de emociones
            return original_image
        else:
            return face_image

    def start_recording(self):
        """Inicia la grabación del historial de emociones."""
        self.emotion_history.start_recording()
        
    def stop_recording(self):
        """Detiene la grabación y retorna el resumen."""
        self.emotion_history.stop_recording()
        return self.emotion_history.get_summary()
    
    def save_history(self, filename: str = None):
        """Guarda el historial en un archivo."""
        return self.emotion_history.save_to_file(filename)
    
    def get_current_summary(self):
        """Obtiene un resumen del estado actual."""
        return self.emotion_history.get_summary()

    def get_current_emotions(self):
        """Obtiene las emociones actuales sin mostrarlas visualmente."""
        if not self.emotion_history_list:
            return None
        return self.emotion_history_list[-1]

    def summarize_emotions(self):
        """Resumen de emociones (método existente, ahora con normalización)."""
        if not self.emotion_history_list:
            return {
                "facial_emotions": {
                    "emotion_facial": "none",
                    "confidence": 0.0,
                    "top_emotions": [],
                    "landmarks": {}
                }
            }

        # Calcular promedio de emociones normalizadas
        emotions_avg = {}
        for emotion in self.emotion_history_list[0].keys():
            scores = [frame[emotion] for frame in self.emotion_history_list]
            emotions_avg[emotion] = sum(scores) / len(scores)

        # Emoción dominante
        dominant_emotion, confidence = self.emotion_normalizer.get_dominant_emotion(emotions_avg)

        # Top 2 emociones
        top_emotions = sorted(emotions_avg, key=emotions_avg.get, reverse=True)[:2]

        # Últimos landmarks
        landmarks = self.landmarks_history[-1] if self.landmarks_history else {}

        return {
            "facial_emotions": {
                "emotion_facial": dominant_emotion,
                "confidence": confidence,
                "top_emotions": top_emotions,
                "landmarks": landmarks
            }
        }
        
        
        
        
        
#         import numpy as np
# from emotion_processor.face_mesh.face_mesh_processor import FaceMeshProcessor
# from emotion_processor.data_processing.main import PointsProcessing
# from emotion_processor.emotions_recognition.main import EmotionRecognition
# from emotion_processor.emotions_visualizations.main import EmotionsVisualization
# # Importar los nuevos módulos
# from emotion_normalizer import EmotionNormalizer
# from emotion_history import EmotionHistory

# class EmotionRecognitionSystem:
#     def __init__(self):
#         self.face_mesh = FaceMeshProcessor()
#         self.data_processing = PointsProcessing()
#         self.emotions_recognition = EmotionRecognition()
#         self.emotions_visualization = EmotionsVisualization()
        
#         # Nuevos componentes
#         self.emotion_normalizer = EmotionNormalizer()
#         self.emotion_history = EmotionHistory()
        
#         self.emotion_history_list = []
#         self.landmarks_history = []
#         self.frame_count = 0

#     def frame_processing(self, face_image: np.ndarray):
#         face_points, control_process, original_image = self.face_mesh.process(face_image, draw=True)
        
#         if control_process:
#             # Procesar características faciales
#             processed_features = self.data_processing.main(face_points)
            
#             # Reconocer emociones (scores crudos)
#             raw_emotions = self.emotions_recognition.recognize_emotion(processed_features)
            
#             # NUEVO: Normalizar emociones para evitar conflictos
#             normalized_emotions = self.emotion_normalizer.normalize(raw_emotions)
            
#             # Agregar al historial
#             self.emotion_history_list.append(normalized_emotions)
#             self.landmarks_history.append(face_points)
#             self.frame_count += 1
            
#             # NUEVO: Registrar en historial de grabación si está activo
#             self.emotion_history.add_frame(normalized_emotions, self.frame_count)
            
#             # Visualizar emociones normalizadas
#             draw_emotions = self.emotions_visualization.main(normalized_emotions, original_image)
#             return draw_emotions
#         else:
#             return face_image

#     def start_recording(self):
#         """Inicia la grabación del historial de emociones."""
#         self.emotion_history.start_recording()
        
#     def stop_recording(self):
#         """Detiene la grabación y retorna el resumen."""
#         self.emotion_history.stop_recording()
#         return self.emotion_history.get_summary()
    
#     def save_history(self, filename: str = None):
#         """Guarda el historial en un archivo."""
#         return self.emotion_history.save_to_file(filename)
    
#     def get_current_summary(self):
#         """Obtiene un resumen del estado actual."""
#         return self.emotion_history.get_summary()

#     def summarize_emotions(self):
#         """Resumen de emociones (método existente, ahora con normalización)."""
#         if not self.emotion_history_list:
#             return {
#                 "facial_emotions": {
#                     "emotion_facial": "none",
#                     "confidence": 0.0,
#                     "top_emotions": [],
#                     "landmarks": {}
#                 }
#             }

#         # Calcular promedio de emociones normalizadas
#         emotions_avg = {}
#         for emotion in self.emotion_history_list[0].keys():
#             scores = [frame[emotion] for frame in self.emotion_history_list]
#             emotions_avg[emotion] = sum(scores) / len(scores)

#         # Emoción dominante
#         dominant_emotion, confidence = self.emotion_normalizer.get_dominant_emotion(emotions_avg)

#         # Top 2 emociones
#         top_emotions = sorted(emotions_avg, key=emotions_avg.get, reverse=True)[:2]

#         # Últimos landmarks
#         landmarks = self.landmarks_history[-1] if self.landmarks_history else {}

#         return {
#             "facial_emotions": {
#                 "emotion_facial": dominant_emotion,
#                 "confidence": confidence,
#                 "top_emotions": top_emotions,
#                 "landmarks": landmarks
#             }
#         }