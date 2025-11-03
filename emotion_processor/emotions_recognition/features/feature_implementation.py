from emotion_processor.emotions_recognition.features.feature_check import (EyebrowsCheck, EyesCheck, NoseCheck, MouthCheck)
import numpy as np

class BasicEyebrowsCheck(EyebrowsCheck):
    def check_eyebrows(self, eyebrows: dict) -> dict:
        # Extrae métricas
        eye_right = eyebrows['eye_right_distance']
        forehead_right = eyebrows['forehead_right_distance']
        eye_left = eyebrows['eye_left_distance']
        forehead_left = eyebrows['forehead_left_distance']
        eyebrows_distance = eyebrows['eyebrows_distance']
        forehead_distance = eyebrows['eyebrow_distance_forehead']

        # Estima ancho del rostro (usa valor de eyes si disponible)
        face_width = eyebrows.get('face_width', 100.0)

        # Normaliza distancias
        norm_eyebrows_dist = eyebrows_distance / face_width
        norm_forehead_dist = forehead_distance / face_width
        norm_eye_right = eye_right / face_width
        norm_forehead_right = forehead_right / face_width
        norm_eye_left = eye_left / face_width
        norm_forehead_left = forehead_left / face_width

        # Calcula 'together' (cejas juntas, AU4): alto si distancia central reducida
        together_score = max(0.0, 1.0 - (norm_eyebrows_dist / 0.2)) if norm_eyebrows_dist < 0.2 else 0.0  # Umbral 0.2 típico para juntas

        # Calcula raised (elevadas, AU1/AU2): grado por comparación normalizada
        right_raised_score = min(1.0, (norm_eye_right - norm_forehead_right) / 0.1) if norm_eye_right > norm_forehead_right else 0.0
        left_raised_score = min(1.0, (norm_eye_left - norm_forehead_left) / 0.1) if norm_eye_left > norm_forehead_left else 0.0

        return {
            'together': together_score,       # 0-1, alto si juntas
            'right_raised': right_raised_score,  # 0-1, grado de elevación derecha
            'left_raised': left_raised_score    # 0-1, grado de elevación izquierda
        }

class BasicEyesCheck(EyesCheck):
    def check_eyes(self, eyes: dict) -> dict:
        # Tu código actual está bien, pero ajusta umbrales basados en búsquedas: openness >0.15 para moderado
        right_upper = eyes['right_upper_eyelid_distance']
        right_lower = eyes['right_lower_eyelid_distance']
        left_upper = eyes['left_upper_eyelid_distance']
        left_lower = eyes['left_lower_eyelid_distance']
        right_arch = eyes['arch_right']
        left_arch = eyes['arch_left']

        face_width = eyes.get('face_width', 100.0)

        norm_right_upper = right_upper / face_width
        norm_left_upper = left_upper / face_width
        norm_right_lower = right_lower / face_width
        norm_left_lower = left_lower / face_width

        openness = ((norm_right_upper + norm_right_lower) + (norm_left_upper + norm_left_lower)) / 4.0  # Promedio total para EAR-like
        openness_score = min(1.0, (openness - 0.15) / (0.3 - 0.15)) if openness > 0.15 else 0.0  # Gradual entre 0.15-0.3

        tension = abs(right_arch - left_arch) / max(abs(right_arch), abs(left_arch), 1e-6)
        tension_score = min(1.0, tension / 0.1) if tension > 0.05 else 0.0

        return {
            'openness': openness_score,
            'tension': tension_score
        }

class BasicNoseCheck(NoseCheck):
    def check_nose(self, nose: dict) -> dict:
        mouth_upper = nose['mouth_upper_distance']
        nose_lower = nose['nose_lower_distance']

        # Asume 'nose_width' o similar; si no, calcula de puntos (ej. distancia entre fosas)
        nose_width = nose.get('nose_width', 50.0)  # Agrega esto en nose_processing.py si falta
        face_width = nose.get('face_width', 100.0)
        norm_nose_width = nose_width / face_width
        norm_mouth_upper = mouth_upper / face_width
        norm_nose_lower = nose_lower / face_width

        # Flared (dilatada, AU9): alto si ancho > umbral o arrugada si mouth_upper > nose_lower
        flared_score = min(1.0, (norm_nose_width - 0.15) / (0.25 - 0.15)) if norm_nose_width > 0.15 else 0.0  # Umbral 0.15-0.25 típico

        # Opcional: wrinkled si mouth_upper > nose_lower
        wrinkled = 1.0 if norm_mouth_upper > norm_nose_lower else 0.0
        flared_score = max(flared_score, wrinkled)  # Combina

        return {
            'flared': flared_score  # 0-1, alto si dilatada/arrugada
        }

class BasicMouthCheck(MouthCheck):
    def check_mouth(self, mouth: dict) -> dict:
        lips_upper = mouth['mouth_upper_distance']
        lips_lower = mouth['mouth_lower_distance']
        right_smile = mouth['right_smile_distance']
        right_lip = mouth['right_lip_distance']
        left_smile = mouth['left_smile_distance']
        left_lip = mouth['left_lip_distance']

        face_width = mouth.get('face_width', 100.0)

        norm_lips_upper = lips_upper / face_width
        norm_lips_lower = lips_lower / face_width
        norm_right_smile = right_smile / face_width
        norm_right_lip = right_lip / face_width
        norm_left_smile = left_smile / face_width
        norm_left_lip = left_lip / face_width

        # Tensión (AU17): alto si apertura pequeña pero >0 (labios tensos)
        open_dist = (norm_lips_upper + norm_lips_lower) / 2.0
        tension_score = min(1.0, 1.0 - (open_dist / 0.1)) if 0.01 < open_dist < 0.1 else 0.0  # Umbral 0.01-0.1 para tenso

        # No sonrisa: alto si esquinas no elevadas
        right_no_smile = 1.0 if norm_right_lip <= norm_right_smile else 0.0
        left_no_smile = 1.0 if norm_left_lip <= norm_left_smile else 0.0
        no_smile_score = (right_no_smile + left_no_smile) / 2.0

        return {
            'tension': tension_score,  # 0-1, alto si tenso
            'no_smile': no_smile_score  # 0-1, alto si sin sonrisa
        }