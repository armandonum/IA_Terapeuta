"""
Sintetizador de voz para el avatar terapeuta
Usa Edge-TTS para voces de alta calidad
"""
import edge_tts
import asyncio
import pygame
import tempfile
import os
from threading import Thread
import time


class VoiceSynthesizer:
    def __init__(self, voice="es-CO-SalomeNeural"):
        """
        Inicializa el sintetizador de voz
        
        Voces femeninas tiernas en español:
        - "es-MX-DaliaNeural" (México - MUY TIERNA, perfecta para gatito)
        - "es-ES-ElviraNeural" (España - Dulce)
        - "es-AR-ElenaNeural" (Argentina - Cálida)
        - "es-CO-SalomeNeural" (Colombia - Suave)
        
        Para voz de niña:
        - "es-MX-BeatrizNeural" (México - Voz infantil)
        """
        self.voice = voice
        self.is_speaking = False
        self.temp_dir = tempfile.gettempdir()
        
        # Inicializar pygame para reproducir audio
        pygame.mixer.init()
        
        print(f"🎙️ VoiceSynthesizer inicializado con voz: {voice}")
    
    def speak(self, text: str, rate="+0%", pitch="+5Hz"):
        """
        Habla el texto proporcionado
        
        Args:
            text: Texto a sintetizar
            rate: Velocidad de habla (ej: "+10%" más rápido, "-10%" más lento)
            pitch: Tono de voz (ej: "+10Hz" más agudo, "-10Hz" más grave)
                   Para voz tierna de gatito usa: "+5Hz" a "+10Hz"
        """
        if self.is_speaking:
            print("⚠️ Ya está hablando, esperando...")
            return
        
        # Ejecutar síntesis en thread separado para no bloquear
        thread = Thread(target=self._speak_sync, args=(text, rate, pitch), daemon=True)
        thread.start()
    
    def _speak_sync(self, text: str, rate: str, pitch: str):
        """Versión sincrónica de speak (para ejecutar en thread)"""
        try:
            asyncio.run(self._speak_async(text, rate, pitch))
        except Exception as e:
            print(f"❌ Error en síntesis de voz: {e}")
    
    async def _speak_async(self, text: str, rate: str, pitch: str):
        """Síntesis asíncrona con Edge-TTS"""
        self.is_speaking = True
        
        try:
            # Crear archivo temporal
            audio_file = os.path.join(self.temp_dir, f"tts_{int(time.time())}.mp3")
            
            print(f"🎤 Sintetizando: '{text[:50]}...'")
            
            # Generar audio con Edge-TTS
            communicate = edge_tts.Communicate(
                text=text,
                voice=self.voice,
                rate=rate,
                pitch=pitch
            )
            
            await communicate.save(audio_file)
            
            # Reproducir audio
            print("🔊 Reproduciendo voz...")
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()
            
            # Esperar a que termine de reproducir
            while pygame.mixer.music.get_busy():
                await asyncio.sleep(0.1)
            
            print("✅ Reproducción finalizada")
            
            # Limpiar archivo temporal
            try:
                os.remove(audio_file)
            except:
                pass
                
        except Exception as e:
            print(f"❌ Error en síntesis asíncrona: {e}")
        
        finally:
            self.is_speaking = False
    
    def stop(self):
        """Detiene la reproducción actual"""
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
            self.is_speaking = False
            print("⏹️ Reproducción detenida")
    
    def is_busy(self):
        """Verifica si está hablando actualmente"""
        return self.is_speaking
    
    def set_voice(self, voice: str):
        """Cambia la voz del sintetizador"""
        self.voice = voice
        print(f"🔄 Voz cambiada a: {voice}")


# === VERSIÓN ALTERNATIVA CON GTTS (Más simple pero menos calidad) ===
class SimpleVoiceSynthesizer:
    """Versión simple usando gTTS (Google Text-to-Speech)"""
    def __init__(self):
        from gtts import gTTS
        pygame.mixer.init()
        self.temp_dir = tempfile.gettempdir()
        self.is_speaking = False
        print("🎙️ SimpleVoiceSynthesizer inicializado con gTTS")
    
    def speak(self, text: str, slow=False):
        """Habla el texto con gTTS"""
        if self.is_speaking:
            return
        
        thread = Thread(target=self._speak_sync, args=(text, slow), daemon=True)
        thread.start()
    
    def _speak_sync(self, text: str, slow: bool):
        from gtts import gTTS
        
        self.is_speaking = True
        
        try:
            audio_file = os.path.join(self.temp_dir, f"tts_{int(time.time())}.mp3")
            
            # Generar audio
            tts = gTTS(text=text, lang='es', slow=slow)
            tts.save(audio_file)
            
            # Reproducir
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()
            
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            
            # Limpiar
            try:
                os.remove(audio_file)
            except:
                pass
                
        except Exception as e:
            print(f"❌ Error en síntesis: {e}")
        
        finally:
            self.is_speaking = False
    
    def stop(self):
        pygame.mixer.music.stop()
        self.is_speaking = False


# === EJEMPLO DE USO ===
if __name__ == "__main__":
    import time
    
    print("\n" + "="*60)
    print("🐱 DEMO: VOZ TIERNA DEL GATITO TERAPEUTA")
    print("="*60)
    
    # Inicializar sintetizador con voz tierna
    voice = VoiceSynthesizer(voice="es-CO-SalomeNeural")
    
    # Ejemplos de respuestas del terapeuta
    frases = [
        "Hola, soy tu gatito terapeuta. ¿Cómo te sientes hoy?",
        "Entiendo que estés pasando por un momento difícil. Estoy aquí para ayudarte.",
        "Es completamente normal sentirse triste a veces. Cuéntame más sobre lo que sientes.",
        "Estoy muy orgulloso de ti por compartir tus emociones conmigo."
    ]
    
    print("\n🎭 Reproduciendo frases con voz tierna (+5Hz)...\n")
    
    for i, frase in enumerate(frases, 1):
        print(f"\n{i}. {frase}")
        voice.speak(frase, rate="+0%", pitch="+5Hz")
        
        # Esperar a que termine de hablar
        while voice.is_busy():
            time.sleep(0.1)
        
        time.sleep(1)  # Pausa entre frases
    
    print("\n✅ Demo finalizada")
    print("\n💡 VOCES DISPONIBLES:")
    print("   • es-MX-DaliaNeural (México) - ⭐ MÁS TIERNA")
    print("   • es-MX-BeatrizNeural (México) - Voz infantil")
    print("   • es-ES-ElviraNeural (España) - Dulce")
    print("   • es-AR-ElenaNeural (Argentina) - Cálida")
    print("   • es-CO-SalomeNeural (Colombia) - Suave")