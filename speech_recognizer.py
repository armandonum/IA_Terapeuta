"""
Reconocedor de voz con Whisper - Transcribe cuando dejas de hablar
"""
import whisper
import sounddevice as sd
import numpy as np
from threading import Thread
from queue import Queue
import torch
import time

class SpeechRecognizer:
    def __init__(self, model_size="base"):
        """
        Inicializa el reconocedor con Whisper
        model_size: 'tiny', 'base', 'small', 'medium', 'large'
        """
        print("🎤 Cargando modelo Whisper...")
        
        # Cargar modelo Whisper
        self.model = whisper.load_model(model_size)
        
        # Configuración de audio
        self.sample_rate = 16000
        self.channels = 1
        
        # Configuración de detección de voz
        self.silence_threshold = 0.01  # Umbral de energía para considerar silencio
        self.silence_duration = 1.5    # Segundos de silencio para considerar que terminaste de hablar
        self.min_audio_length = 0.5    # Mínimo de audio en segundos para procesar
        
        # Colas y estado
        self.audio_queue = Queue()
        self.text_queue = Queue()
        self.is_listening = False
        self.audio_buffer = []
        self.is_speaking = False
        self.last_sound_time = None
        
        # Detectar si hay GPU disponible
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"✅ Modelo '{model_size}' cargado en {self.device}")
        print("🎤 SpeechRecognizer inicializado")
    
    def reset_session(self):
        """
        Limpia completamente el historial de texto y audio
        Llamar esto antes de iniciar una nueva sesión
        """
        # Limpiar cola de texto
        while not self.text_queue.empty():
            try:
                self.text_queue.get_nowait()
            except:
                break
        
        # Limpiar cola de audio
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except:
                break
        
        # Limpiar buffer de audio
        self.audio_buffer = []
        
        # Resetear estado de voz
        self.is_speaking = False
        self.last_sound_time = None
        
        print("🔄 Sesión de voz reseteada - Todo el historial limpiado")
    
    def start_listening(self):
        """Inicia captura de voz - LIMPIA EL HISTORIAL AUTOMÁTICAMENTE"""
        if self.is_listening:
            return
        
        # IMPORTANTE: Limpiar historial al iniciar nueva sesión
        self.reset_session()
        
        self.is_listening = True
        
        # Thread para capturar audio
        capture_thread = Thread(target=self._capture_audio, daemon=True)
        capture_thread.start()
        
        # Thread para procesar audio
        process_thread = Thread(target=self._process_audio, daemon=True)
        process_thread.start()
        
        print("🔴 Escuchando... (Sesión nueva)")
    
    def stop_listening(self):
        """Detiene captura"""
        self.is_listening = False
        
        # Procesar audio restante si hay
        if self.audio_buffer:
            self._transcribe_buffer()
        
        print("⏹️  Voz detenida")
    
    def get_all_text(self):
        """Obtiene todo el texto acumulado de la sesión actual"""
        texts = []
        while not self.text_queue.empty():
            texts.append(self.text_queue.get())
        
        result = " ".join(texts) if texts else ""
        
        if result:
            print(f"📋 Texto acumulado en esta sesión: {result}")
        else:
            print("📋 No hay texto en esta sesión")
        
        return result
    
    def _audio_callback(self, indata, frames, time_info, status):
        """Callback para capturar audio del micrófono"""
        if status:
            print(f"⚠️  Estado: {status}")
        self.audio_queue.put(indata.copy())
    
    def _capture_audio(self):
        """Captura audio del micrófono"""
        print("⚙️  Ajustando al ruido ambiente...")
        
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            callback=self._audio_callback,
            blocksize=2048
        ):
            print("✅ Listo para escuchar")
            while self.is_listening:
                sd.sleep(100)
    
    def _calculate_energy(self, audio_chunk):
        """Calcula la energía del audio para detectar voz"""
        return np.sqrt(np.mean(audio_chunk.astype(np.float32) ** 2))
    
    def _process_audio(self):
        """Procesa audio y transcribe cuando detecta silencio"""
        while self.is_listening:
            try:
                if not self.audio_queue.empty():
                    chunk = self.audio_queue.get()
                    audio_flat = chunk.flatten()
                    
                    # Calcular energía del chunk actual
                    energy = self._calculate_energy(audio_flat)
                    current_time = time.time()
                    
                    # Detectar si hay voz
                    if energy > self.silence_threshold:
                        # Hay voz
                        if not self.is_speaking:
                            self.is_speaking = True
                            print("🎙️  Detectado inicio de voz...")
                        
                        self.audio_buffer.append(audio_flat)
                        self.last_sound_time = current_time
                    
                    else:
                        # Silencio detectado
                        if self.is_speaking:
                            # Si estábamos hablando, agregamos el silencio al buffer
                            self.audio_buffer.append(audio_flat)
                            
                            # Verificar si ha pasado suficiente tiempo de silencio
                            if self.last_sound_time and (current_time - self.last_sound_time) >= self.silence_duration:
                                print("⏸️  Silencio detectado, transcribiendo...")
                                self._transcribe_buffer()
                                self.is_speaking = False
                                self.audio_buffer = []
                                self.last_sound_time = None
                
                else:
                    time.sleep(0.01)  # Pequeña pausa si no hay audio
                    
            except Exception as e:
                print(f"❌ Error en procesamiento: {e}")
                break
    
    def _transcribe_buffer(self):
        """Transcribe el buffer de audio acumulado"""
        if not self.audio_buffer:
            return
        
        try:
            # Concatenar todo el audio
            audio_data = np.concatenate(self.audio_buffer)
            
            # Verificar longitud mínima
            min_samples = int(self.min_audio_length * self.sample_rate)
            if len(audio_data) < min_samples:
                return
            
            # Normalizar audio
            audio_float = audio_data.astype(np.float32)
            max_val = np.max(np.abs(audio_float))
            
            if max_val > 0:
                audio_float = audio_float / max_val
            
            # Transcribir con Whisper
            result = self.model.transcribe(
                audio_float,
                language="es",
                fp16=(self.device == "cuda"),
                task="transcribe",
                without_timestamps=True
            )
            
            text = result["text"].strip()
            
            if text:
                self.text_queue.put(text)
                print(f"📝 Transcrito: {text}")
            
        except Exception as e:
            print(f"❌ Error en transcripción: {e}")


# Ejemplo de uso
if __name__ == "__main__":
    import time
    
    # Crear reconocedor
    recognizer = SpeechRecognizer(model_size="base")
    
    print("\n" + "="*60)
    print("PRUEBA DE MÚLTIPLES SESIONES")
    print("="*60)
    
    # === SESIÓN 1 ===
    print("\n🔵 SESIÓN 1 - Habla algo...")
    recognizer.start_listening()
    
    print("💡 Habla y haz pausas. Se transcribirá automáticamente.")
    print("💡 Esperando 10 segundos...")
    time.sleep(10)
    
    recognizer.stop_listening()
    text1 = recognizer.get_all_text()
    print(f"\n✅ SESIÓN 1 FINALIZADA")
    print(f"📄 Texto capturado: {text1}")
    
    # Pausa entre sesiones
    print("\n⏳ Pausa de 3 segundos...")
    time.sleep(3)
    
    # === SESIÓN 2 ===
    print("\n🟢 SESIÓN 2 - Habla algo DIFERENTE...")
    recognizer.start_listening()  # Esto limpiará automáticamente la sesión anterior
    
    print("💡 Esta es una NUEVA sesión. El texto anterior NO se acumulará.")
    print("💡 Esperando 10 segundos...")
    time.sleep(10)
    
    recognizer.stop_listening()
    text2 = recognizer.get_all_text()
    print(f"\n✅ SESIÓN 2 FINALIZADA")
    print(f"📄 Texto capturado: {text2}")
    
    # Comparación
    print("\n" + "="*60)
    print("COMPARACIÓN DE SESIONES")
    print("="*60)
    print(f"Sesión 1: {text1}")
    print(f"Sesión 2: {text2}")
    print(f"¿Son diferentes? {text1 != text2}")
    print("="*60)