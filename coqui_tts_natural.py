# coqui_tts_natural.py
import os
import pygame
import tempfile
import time
from threading import Thread
from TTS.api import TTS

class NaturalSpanishTTS:
    def __init__(self):
        print("Inicializando TTS Natural en Español...")
        self.is_speaking = False
        self.temp_dir = tempfile.gettempdir()
        self.tts = None
        self.model_name = None
        self.voices = []
        self.female_voices = []

        # Modelos probados en orden de calidad
        models = [
            "tts_models/multilingual/multi-dataset/xtts_v2",
            "tts_models/es/mai/tacotron2-DDC",
            "tts_models/es/css10/vits",
        ]

        for model in models:
            try:
                print(f"Cargando modelo: {model}")
                self.tts = TTS(model_name=model, progress_bar=False)
                self.model_name = model
                if hasattr(self.tts, 'speakers') and self.tts.speakers:
                    self.voices = self.tts.speakers
                print(f"Modelo cargado: {model}")
                break
            except Exception as e:
                print(f"No disponible: {model} → {e}")
                continue

        if not self.tts:
            raise Exception("No se pudo cargar ningún modelo TTS")

        self._detect_female_voices()
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        print(f"TTS listo. Voces femeninas: {len(self.female_voices)}")

    def _detect_female_voices(self):
        keywords = ['female', 'woman', 'mujer', 'ana', 'laura', 'maria', 'elena', 'speaker_0', 'speaker_2']
        for voice in self.voices:
            if any(k in voice.lower() for k in keywords):
                self.female_voices.append(voice)

    def speak(self, text, emotion="neutral", speed=1.0, speaker=None):
        if self.is_speaking or not text.strip():
            return
        enhanced_text = self._enhance_text(text, emotion)
        speaker = speaker or (self.female_voices[0] if self.female_voices else None)

        Thread(target=self._speak_thread, args=(enhanced_text, speed, speaker), daemon=True).start()

    def _enhance_text(self, text, emotion):
        enhancements = {
            "happy": f"¡{text}!" if not text.endswith("!") else text,
            "excited": f"¡¡{text}!!",
            "sad": text + "..." if not text.endswith("...") else text,
            "calm": text.replace(".", "..."),
            "empathy": f"{text}.",
        }
        return enhancements.get(emotion, text)

    def _speak_thread(self, text, speed, speaker):
        self.is_speaking = True
        audio_file = os.path.join(self.temp_dir, f"tts_{int(time.time())}.wav")

        try:
            kwargs = {"text": text, "file_path": audio_file, "speed": speed}
            if speaker and self.model_name != "tts_models/es/css10/vits":
                kwargs["speaker"] = speaker
            if "xtts" in self.model_name:
                kwargs["language"] = "es"

            self.tts.tts_to_file(**kwargs)

            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)

            os.remove(audio_file)
            print("Voz reproducida")
        except Exception as e:
            print(f"Error TTS: {e}")
        finally:
            self.is_speaking = False

    def speak_therapeutic(self, text, emotion="empathy"):
        self.speak(text, emotion=emotion, speed=0.93)

    def stop(self):
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
        self.is_speaking = False
        print("Voz detenida")

    def is_busy(self):
        return self.is_speaking

    def list_voices(self):
        print("\nVoces disponibles:")
        for i, v in enumerate(self.voices, 1):
            marker = "FEMENINA" if v in self.female_voices else "MASCULINA"
            print(f"  {i}. {v} [{marker}]")