"""
TTS en Español con VOZ NATURAL y BUENA ENTONACIÓN
Optimizado para voz femenina expresiva
"""
import os
import pygame
import tempfile
import time
from threading import Thread
from TTS.api import TTS


class NaturalSpanishTTS:
    """
    TTS optimizado para español con las MEJORES voces naturales disponibles
    """
    
    def __init__(self):
        """
        Inicializa con el mejor modelo de voz en español
        
        MODELOS RECOMENDADOS (del mejor al más básico):
        1. "tts_models/es/css10/vits" - ⭐ MEJOR: Voz natural española
        2. "tts_models/multilingual/multi-dataset/xtts_v2" - ⭐⭐ EXCELENTE: Multilingüe, muy expresivo
        3. "tts_models/es/mai/tacotron2-DDC" - Buena alternativa
        """
        print("🎭 Inicializando TTS Natural en Español...")
        print("⏳ Descargando modelo (solo la primera vez)...\n")
        
        # Intentar modelos en orden de calidad
        models_to_try = [
            ("tts_models/multilingual/multi-dataset/xtts_v2", "es", "La mejor calidad - Multilingüe XTTS"),
            ("tts_models/es/css10/vits", None, "Voz natural española VITS"),
            ("tts_models/es/mai/tacotron2-DDC", None, "Voz española Tacotron2"),
        ]
        
        self.tts = None
        self.model_name = None
        self.language = None
        
        for model, lang, desc in models_to_try:
            try:
                print(f"🔄 Probando: {desc}")
                self.tts = TTS(model_name=model)
                self.model_name = model
                self.language = lang
                print(f"✅ Modelo cargado: {model}\n")
                break
            except Exception as e:
                print(f"⚠️  No disponible, probando siguiente...\n")
                continue
        
        if not self.tts:
            raise Exception("❌ No se pudo cargar ningún modelo de TTS")
        
        self.is_speaking = False
        self.temp_dir = tempfile.gettempdir()
        
        # Inicializar pygame mixer con mejor calidad
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
        
        # Detectar si el modelo soporta múltiples voces
        self.voices = None
        if hasattr(self.tts, 'speakers') and self.tts.speakers:
            self.voices = self.tts.speakers
            print(f"🎤 Voces disponibles: {len(self.voices)}")
            
            # Filtrar voces femeninas si es posible
            self.female_voices = self._detect_female_voices()
            if self.female_voices:
                print(f"👩 Voces femeninas detectadas: {len(self.female_voices)}")
        
        print("✅ TTS listo para sintetizar con voz natural\n")
    
    def _detect_female_voices(self):
        """Intenta detectar voces femeninas por nombre"""
        if not self.voices:
            return None
        
        # Palabras clave para voces femeninas
        female_keywords = [
            'female', 'woman', 'girl', 'fem', 'mujer', 'nina',
            'maria', 'ana', 'elena', 'sofia', 'laura', 'carmen',
            'speaker_0', 'speaker_2', 'speaker_4', 'speaker_6'  # Números pares suelen ser femeninas
        ]
        
        female_voices = []
        for voice in self.voices:
            voice_lower = voice.lower()
            if any(keyword in voice_lower for keyword in female_keywords):
                female_voices.append(voice)
        
        return female_voices if female_voices else None
    
    def speak(self, text, emotion="neutral", speed=1.0, speaker=None):
        """
        Sintetiza texto con voz natural y expresiva
        
        Args:
            text: Texto a sintetizar
            emotion: Emoción a expresar (agrega contexto al texto)
                    - "neutral": Normal
                    - "happy": Alegre, entusiasta
                    - "calm": Tranquila, relajada
                    - "empathy": Empática, comprensiva
                    - "excited": Emocionada
            speed: Velocidad (0.5 = lento, 1.0 = normal, 1.5 = rápido)
            speaker: Voz específica (None = usa la mejor voz femenina)
        """
        if self.is_speaking:
            print("⚠️  Ya está hablando...")
            return
        
        # Mejorar el texto según la emoción
        enhanced_text = self._enhance_text_for_emotion(text, emotion)
        
        # Seleccionar mejor voz femenina si no se especifica
        if not speaker:
            if hasattr(self, 'female_voices') and self.female_voices:
                speaker = self.female_voices[0]  # Primera voz femenina
            elif self.voices:
                speaker = self.voices[0]  # Primera voz disponible
        
        Thread(
            target=self._speak_sync, 
            args=(enhanced_text, speed, speaker), 
            daemon=True
        ).start()
    
    def _enhance_text_for_emotion(self, text, emotion):
        """Mejora el texto para expresar emociones de forma natural"""
        
        # Agregar contexto emocional sin ser demasiado obvio
        emotion_enhancements = {
            "happy": text,  # El tono ya se ajusta con la síntesis
            "calm": text.replace(".", "...") if not text.endswith("...") else text,
            "empathy": text,
            "excited": f"¡{text}!" if not text.endswith("!") else text,
            "sad": text,
        }
        
        return emotion_enhancements.get(emotion, text)
    
    def _speak_sync(self, text, speed, speaker):
        """Síntesis sincrónica con mejor calidad"""
        self.is_speaking = True
        
        try:
            audio_file = os.path.join(self.temp_dir, f"tts_natural_{int(time.time())}.wav")
            
            print(f"🎤 Sintetizando: '{text[:50]}{'...' if len(text) > 50 else ''}'")
            
            # Parámetros de síntesis según el modelo
            tts_kwargs = {
                "text": text,
                "file_path": audio_file,
                "speed": speed
            }
            
            # Agregar speaker si está disponible
            if speaker:
                tts_kwargs["speaker"] = speaker
                print(f"👩 Usando voz: {speaker}")
            
            # Agregar idioma si es necesario (para XTTS)
            if self.language:
                tts_kwargs["language"] = self.language
            
            # Generar audio
            self.tts.tts_to_file(**tts_kwargs)
            
            # Reproducir con mejor calidad
            print("🔊 Reproduciendo voz natural...")
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()
            
            # Esperar a que termine
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            
            print("✅ Reproducción finalizada\n")
            
            # Limpiar archivo temporal
            try:
                os.remove(audio_file)
            except:
                pass
        
        except Exception as e:
            print(f"❌ Error en síntesis: {e}")
        
        finally:
            self.is_speaking = False
    
    def speak_therapeutic(self, text, emotion="empathy"):
        """
        Método específico para respuestas terapéuticas
        Usa velocidad y tono óptimos para terapia
        """
        # Velocidad ligeramente más lenta para terapia (más tranquila)
        self.speak(text, emotion=emotion, speed=0.95)
    
    def list_voices(self):
        """Muestra todas las voces disponibles"""
        if not self.voices:
            print("ℹ️  Este modelo no soporta múltiples voces")
            return []
        
        print("\n🎤 VOCES DISPONIBLES:")
        print("="*50)
        
        if self.female_voices:
            print("\n👩 VOCES FEMENINAS (Recomendadas):")
            for i, voice in enumerate(self.female_voices, 1):
                print(f"  {i}. {voice}")
        
        print("\n📋 TODAS LAS VOCES:")
        for i, voice in enumerate(self.voices, 1):
            is_female = "👩" if self.female_voices and voice in self.female_voices else "  "
            print(f"  {is_female} {i}. {voice}")
        
        print("="*50 + "\n")
        return self.voices
    
    def stop(self):
        """Detiene la reproducción actual"""
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
            self.is_speaking = False
            print("⏹️  Reproducción detenida")
    
    def is_busy(self):
        """Verifica si está hablando"""
        return self.is_speaking


# ===================== EJEMPLO DE USO =====================
def demo_therapeutic():
    """Demo con frases terapéuticas"""
    print("\n" + "="*70)
    print("🐱 DEMO: VOZ NATURAL PARA TERAPEUTA")
    print("="*70 + "\n")
    
    # Inicializar TTS
    tts = NaturalSpanishTTS()
    
    # Mostrar voces disponibles
    tts.list_voices()
    
    # Frases terapéuticas de ejemplo
    therapeutic_phrases = [
        ("Hola, soy tu terapeuta. ¿Cómo te sientes hoy?", "calm"),
        ("Entiendo que estés pasando por un momento difícil. Estoy aquí para ayudarte.", "empathy"),
        ("Es completamente normal sentirse así. No estás solo en esto.", "empathy"),
        ("¡Me alegro mucho de escuchar eso! Es un gran progreso.", "happy"),
        ("Cuéntame más sobre lo que sientes. Te escucho.", "calm"),
    ]
    
    print("🎭 Reproduciendo frases terapéuticas con voz natural...\n")
    
    for i, (frase, emocion) in enumerate(therapeutic_phrases, 1):
        print(f"\n{i}. [{emocion.upper()}] {frase}")
        tts.speak_therapeutic(frase, emotion=emocion)
        
        # Esperar a que termine
        while tts.is_busy():
            time.sleep(0.1)
        
        time.sleep(1)  # Pausa entre frases
    
    print("\n✅ Demo finalizada")
    print("\n💡 TIPS PARA VOZ MÁS NATURAL:")
    print("  • Usa puntos suspensivos (...) para pausas reflexivas")
    print("  • Agrega exclamaciones (¡!) para entusiasmo")
    print("  • Varía la velocidad según la emoción (0.9 = tranquila)")
    print("  • Usa frases cortas y naturales")


def main():
    """Programa principal interactivo"""
    print("\n" + "="*70)
    print("🎤 TTS ESPAÑOL NATURAL - VOZ FEMENINA")
    print("="*70 + "\n")
    
    # Inicializar
    tts = NaturalSpanishTTS()
    
    print("\n📋 OPCIONES:")
    print("1. 🎭 Demo terapéutica")
    print("2. 💬 Modo interactivo")
    print("3. 🎤 Ver voces disponibles")
    
    choice = input("\nOpción: ").strip()
    
    if choice == "1":
        demo_therapeutic()
    
    elif choice == "2":
        print("\n💬 MODO INTERACTIVO")
        print("="*70)
        print("Escribe el texto que quieres escuchar (o 'salir' para terminar)\n")
        
        while True:
            text = input("\n📝 Texto: ").strip()
            
            if text.lower() in ['salir', 'exit', 'quit']:
                print("👋 ¡Hasta luego!")
                break
            
            if not text:
                continue
            
            # Seleccionar emoción
            print("\n😊 Emoción:")
            print("1. Neutral  2. Alegre  3. Tranquila  4. Empática  5. Emocionada")
            emotion_choice = input("Opción (Enter = neutral): ").strip()
            
            emotions = {
                "1": "neutral",
                "2": "happy",
                "3": "calm",
                "4": "empathy",
                "5": "excited"
            }
            emotion = emotions.get(emotion_choice, "neutral")
            
            # Velocidad
            speed_input = input("⚡ Velocidad (0.5-2.0, Enter = 1.0): ").strip()
            try:
                speed = float(speed_input) if speed_input else 1.0
                speed = max(0.5, min(2.0, speed))  # Limitar rango
            except:
                speed = 1.0
            
            # Sintetizar
            tts.speak(text, emotion=emotion, speed=speed)
            
            # Esperar
            while tts.is_busy():
                time.sleep(0.1)
    
    elif choice == "3":
        tts.list_voices()
        
        if tts.voices:
            test_voice = input("\n¿Probar una voz? (número o Enter para salir): ").strip()
            if test_voice.isdigit():
                idx = int(test_voice) - 1
                if 0 <= idx < len(tts.voices):
                    voice = tts.voices[idx]
                    tts.speak(
                        "Hola, esta es mi voz. ¿Te gusta cómo sueno?",
                        speaker=voice
                    )
                    while tts.is_busy():
                        time.sleep(0.1)
    
    else:
        print("❌ Opción inválida")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Programa interrumpido. ¡Hasta luego!")
    except Exception as e:
        print(f"\n❌ Error: {e}")