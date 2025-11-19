"""
Fusiona emociones de texto y rostro
"""


class EmotionFusion:
    def __init__(self):
        # Mapeo de emociones español -> inglés
        self.emotion_map = {
            "alegre": "happy",
            "triste": "sad",
            "enojado": "angry",
            "miedo": "fear",
            "sorprendido": "surprise",
            "repugnante": "disgust",
            "ansiedad": "anxiety",
            "aterrado": "fear",
            "furioso": "angry",
            "feliz": "happy",
            "felicidad": "happy",
            "tristeza": "sad",
            "ira": "angry",
            "temor": "fear",
            "sorpresa": "surprise",
            "disgusto": "disgust"
        }

        # Pesos: PRIORIDAD AL DIÁLOGO
        self.text_weight = 0.65
        self.face_weight = 0.35

        print(f"EmotionFusion: Texto={self.text_weight}, Rostro={self.face_weight}")

    def normalize_text_emotion(self, text_emotion: str) -> str:
        """Normaliza emoción de texto a inglés"""
        text_lower = text_emotion.lower().strip()
        return self.emotion_map.get(text_lower, text_lower)

    def fuse(self, text_emotions: dict, face_emotions: dict) -> dict:
        """
        Fusiona emociones de texto y rostro
        """
        # Normalizar emociones de texto
        normalized_text = {}
        for emotion, score in text_emotions.items():
            norm_emotion = self.normalize_text_emotion(emotion)
            normalized_text[norm_emotion] = float(score)

        # Asegurar que face_emotions sean float
        normalized_face = {k: float(v) for k, v in face_emotions.items()}

        # Obtener todas las emociones únicas
        all_emotions = set(normalized_text.keys()) | set(normalized_face.keys())

        # Fusionar con pesos
        fused = {}
        for emotion in all_emotions:
            text_val = normalized_text.get(emotion, 0.0)
            face_val = normalized_face.get(emotion, 0.0)
            fused[emotion] = (self.text_weight * text_val) + (self.face_weight * face_val)

        # Encontrar emoción primaria
        if fused:
            primary_emotion = max(fused, key=fused.get)
            confidence = fused[primary_emotion]
        else:
            primary_emotion = "neutral"
            confidence = 0.0

        # Detectar conflicto
        text_primary = max(normalized_text, key=normalized_text.get) if normalized_text else "none"
        face_primary = max(normalized_face, key=normalized_face.get) if normalized_face else "none"
        has_conflict = text_primary != face_primary

        return {
            "fused_emotions": fused,
            "primary_emotion": primary_emotion,
            "confidence": round(confidence, 4),
            "text_primary": text_primary,
            "face_primary": face_primary,
            "has_conflict": has_conflict,
            "text_emotions": normalized_text,
            "face_emotions": normalized_face,
            "weights": {
                "text": self.text_weight,
                "face": self.face_weight
            }
        }

    def to_llm_format(self, fusion_result: dict, text_transcribed: str = "", session_id: str = "default_session") -> dict:
        """
        Formato JSON para LLM de terapia TCC
        
        Retorna el formato exacto que espera el LLM:
        {
            "session_id": "...",
            "user_message": "...",
            "emotional_analysis": {
                "emocion_principal": "sad",
                "confianza_principal": 70.0,
                "emociones_texto": {"sad": 70.0, "neutral": 30.0},
                "emociones_rostro": {"sad": 50.0, "neutral": 50.0},
                "emocion_texto_dominante": "sad",
                "emocion_rostro_dominante": "sad",
                "hay_conflicto": false,
                "interpretacion_conflicto": "..." (opcional)
            }
        }
        """
        fr = fusion_result

        # Construir interpretación del conflicto si existe
        interpretacion_conflicto = None
        if fr['has_conflict']:
            interpretacion_conflicto = (
                f"El usuario dice sentir '{fr['text_primary']}' pero su rostro muestra '{fr['face_primary']}'. "
                f"Esto puede indicar enmascaramiento emocional o disonancia entre lo que siente y expresa."
            )

        # Construir el formato JSON para el LLM
        prompt = {
            "session_id": session_id,
            "user_message": text_transcribed,
            "emotional_analysis": {
                "emocion_principal": fr['primary_emotion'],
                "confianza_principal": round(fr['confidence'], 2),
                "emociones_texto": {k: round(v, 2) for k, v in fr['text_emotions'].items()},
                "emociones_rostro": {k: round(v, 2) for k, v in fr['face_emotions'].items()},
                "emocion_texto_dominante": fr['text_primary'],
                "emocion_rostro_dominante": fr['face_primary'],
                "hay_conflicto": fr['has_conflict']
            }
        }

        # Agregar interpretación solo si hay conflicto
        if interpretacion_conflicto:
            prompt["emotional_analysis"]["interpretacion_conflicto"] = interpretacion_conflicto

        return prompt

