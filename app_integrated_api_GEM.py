# app.py - Versión ChatGPT Gratuita Mejorada
import os
import sys
import cv2
import json
import requests
from flask import Flask, Response, render_template, jsonify, request
from examples.camera import Camera
from emotion_processor.main import EmotionRecognitionSystem
from text_emotion_classifier import TextEmotionClassifier
from speech_recognizer import SpeechRecognizer
from emotion_fusion import EmotionFusion
from voice_synthesizer import VoiceSynthesizer

app = Flask(__name__)

# === CONFIGURACIÓN ===
SESSION_ID = f"facesense_{os.getpid()}"


class FreeChatGPTTherapist:
    """Cliente mejorado para APIs de ChatGPT gratuitas"""
    
    def __init__(self):
        self.base_urls = [
            "https://api.openai-proxy.org/v1/chat/completions",  # Proxy alternativo
            "https://chatgpt.apinepdev.workers.dev/",  # Worker de Cloudflare
        ]
        self.current_url_index = 0
        
    def get_therapy_response(self, user_message, emotional_analysis, session_id):
        """Obtiene respuesta usando APIs gratuitas de ChatGPT"""
        
        # Preparar el prompt para el terapeuta
        system_prompt = self._build_therapist_prompt(emotional_analysis)
        
        for attempt in range(len(self.base_urls)):
            try:
                url = self.base_urls[self.current_url_index]
                
                # Rotar a la siguiente URL para el próximo intento
                self.current_url_index = (self.current_url_index + 1) % len(self.base_urls)
                
                headers = {
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "model": "gpt-3.5-turbo",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 150  # Reducido para respuestas más cortas
                }
                
                print(f"🤖 Consultando API gratuita: {url}")
                response = requests.post(url, headers=headers, json=payload, timeout=15)
                
                if response.status_code == 200:
                    result = response.json()
                    therapist_response = result["choices"][0]["message"]["content"]
                    
                    return {
                        'success': True,
                        'response': therapist_response
                    }
                else:
                    print(f"❌ Error API {url}: {response.status_code} - {response.text}")
                    continue
                    
            except Exception as e:
                print(f"❌ Error con API {url}: {e}")
                continue
        
        # Si todas las APIs fallan, usar respuestas predefinidas
        return self._get_fallback_response(user_message, emotional_analysis)
    
    def _build_therapist_prompt(self, emotional_analysis):
        """Construye el prompt del sistema para el terapeuta"""
        emocion_principal = emotional_analysis.get('emocion_principal', 'No detectada')
        hay_conflicto = emotional_analysis.get('hay_conflicto', False)
        emociones = emotional_analysis.get('emociones', {})
        
        prompt = f"""Eres un terapeuta empático y compasivo. Responde al usuario de manera:

1. Empática y comprensiva
2. Profesional pero cálida  
3. Centrada en escuchar y apoyar
4. Con frases cortas (máximo 50 palabras)
5. Valida sus emociones sin juzgar

Análisis emocional:
- Emoción principal: {emocion_principal}
- Conflicto emocional: {'Sí' if hay_conflicto else 'No'}
- Emociones detectadas: {emociones}

Responde directamente al mensaje del usuario de forma natural."""

        return prompt
    
    def _get_fallback_response(self, user_message, emotional_analysis):
        """Respuestas de fallback mejoradas cuando las APIs no están disponibles"""
        emocion_principal = emotional_analysis.get('emocion_principal', 'neutral')
        
        # Mapeo completo de emociones en inglés y español
        emociones_mapeo = {
            # Español
            "tristeza": "tristeza", "enojo": "enojo", "miedo": "miedo", 
            "felicidad": "felicidad", "neutral": "neutral", "sorpresa": "sorpresa",
            # Inglés
            "sad": "tristeza", "angry": "enojo", "fear": "miedo", "disgust": "enojo",
            "happy": "felicidad", "neutral": "neutral", "surprise": "sorpresa",
            "fearful": "miedo", "angry": "enojo", "surprised": "sorpresa"
        }
        
        emocion_fallback = emociones_mapeo.get(emocion_principal.lower(), "neutral")
        
        fallback_responses = {
            "tristeza": "Veo que estás pasando por un momento difícil. Quiero que sepas que tus sentimientos son válidos y estoy aquí para escucharte. ¿Quieres compartir más sobre lo que te preocupa?",
            "enojo": "Entiendo que puedas sentir frustración en este momento. Es normal sentirse así a veces. Tomemos un respiro juntos. Estoy aquí para apoyarte.",
            "miedo": "El miedo puede ser abrumador, pero quiero que sepas que estás a salvo aquí. Podemos enfrentar esto juntos, paso a paso. ¿Qué es lo que más te inquieta?",
            "felicidad": "Me alegra mucho ver que te sientes bien hoy. Celebrar estos momentos positivos es muy importante. ¿Quieres contarme qué te hace feliz?",
            "neutral": "Hola, gracias por compartir este tiempo conmigo. Estoy aquí para escucharte y apoyarte en lo que necesites. ¿Hay algo en particular de lo que te gustaría hablar?",
            "sorpresa": "Parece que algo te ha tomado por sorpresa. A veces lo inesperado puede ser confuso. Estoy aquí para ayudarte a procesar lo que estás sintiendo."
        }
        
        response = fallback_responses.get(emocion_fallback, fallback_responses["neutral"])
        
        return {
            'success': True,
            'response': response,
            'fallback': True,
            'emocion_detectada': emocion_principal,
            'emocion_mapeada': emocion_fallback
        }


class IntegratedVideoStream:
    def __init__(self, cam, face_system, text_classifier, speech_rec, fusion, voice_synth):
        self.camera = cam
        self.face_system = face_system
        self.text_classifier = text_classifier
        self.speech_recognizer = speech_rec
        self.fusion_engine = fusion
        self.is_recording = False
        self.video_writer = None
        self.accumulated_text = ""
        self.latest_fusion = None
        self.text_mode = True
        self.voice_synthesizer = voice_synth
        self.chatgpt_therapist = FreeChatGPTTherapist()

    def start_recording(self):
        """Inicia grabación de video + voz/texto"""
        self.is_recording = True
        self.accumulated_text = ""

        # Iniciar video
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = 20.0
        frame_size = (
            int(self.camera.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.camera.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )
        # Crear nombre de archivo incremental
        base_name = "output"
        ext = ".avi"
        filename = f"{base_name}{ext}"
        idx = 1
        while os.path.exists(filename):
            filename = f"{base_name}_{idx}{ext}"
            idx += 1
        self.video_writer = cv2.VideoWriter(filename, fourcc, fps, frame_size)
        print(f"🎬 Grabando en: {filename}")

        # Iniciar voz solo si no está en modo texto
        if not self.text_mode:
            print("🎤 Iniciando reconocimiento de voz...")
            self.speech_recognizer.start_listening()

        # Iniciar historial facial
        self.face_system.start_recording()

        mode = "Texto" if self.text_mode else "Voz"
        print(f"✅ Grabación iniciada: Video + {mode} + Rostro")

    def stop_recording(self, text_from_chat=None):
        """Detiene grabación y genera resultado fusionado"""
        if not self.is_recording:
            return None

        self.is_recording = False

        # Detener video
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None

        # Obtener texto según el modo
        if self.text_mode and text_from_chat:
            self.accumulated_text = text_from_chat
            print(f"📝 Texto recibido: {self.accumulated_text}")
        else:
            print("🎤 Deteniendo reconocimiento de voz...")
            self.speech_recognizer.stop_listening()
            self.accumulated_text = self.speech_recognizer.get_all_text()
            print(f"📝 Texto transcrito: {self.accumulated_text}")

        # Detener historial facial
        print("😊 Deteniendo análisis facial...")
        face_summary = self.face_system.stop_recording()

        # Clasificar texto
        text_result = self.text_classifier.classify(self.accumulated_text)
        print(f"📊 Análisis de texto: {text_result}")

        # Preparar emociones faciales
        if face_summary and 'emotion_statistics' in face_summary:
            face_emotions = {
                emotion: stats['mean']
                for emotion, stats in face_summary['emotion_statistics'].items()
            }
            print(f"😊 Emociones faciales: {face_emotions}")
        else:
            face_emotions = {"neutral": 50.0}
            print("😊 No se detectaron emociones faciales, usando neutral")

        # Fusionar resultados
        fusion_result = self.fusion_engine.fuse(
            text_result['emotions'],
            face_emotions
        )
        self.latest_fusion = fusion_result
        print(f"🔀 Fusión emocional: {fusion_result}")

        # Generar payload para ChatGPT
        llm_payload = self.fusion_engine.to_llm_format(
            fusion_result=fusion_result,
            text_transcribed=self.accumulated_text,
            session_id=SESSION_ID
        )

        # === LLAMAR A CHATGPT GRATUITO ===
        therapist_response = "Error al conectar con el terapeuta"
        fallback_used = False
        
        try:
            print(f"🤖 Consultando ChatGPT gratuito...")
            print(f"📤 Análisis emocional:")
            print(f"   - Emoción: {llm_payload['emotional_analysis'].get('emocion_principal', 'N/A')}")
            print(f"   - Conflicto: {llm_payload['emotional_analysis'].get('hay_conflicto', False)}")

            # Llamar a ChatGPT gratuito
            result = self.chatgpt_therapist.get_therapy_response(
                user_message=self.accumulated_text,
                emotional_analysis=llm_payload['emotional_analysis'],
                session_id=SESSION_ID
            )

            if result['success']:
                therapist_response = result['response']
                fallback_used = result.get('fallback', False)
                
                if fallback_used:
                    emocion_detectada = result.get('emocion_detectada', 'N/A')
                    emocion_mapeada = result.get('emocion_mapeada', 'N/A')
                    print(f"⚠️ Usando respuesta de fallback (emoción: {emocion_detectada} -> {emocion_mapeada})")
                else:
                    print("✅ Respuesta de ChatGPT obtenida")

                # Sintetizar voz
                print("🔊 Sintetizando voz...")
                self.voice_synthesizer.speak(
                    text=therapist_response,
                    rate="+0%",
                    pitch="+7Hz"
                )
            else:
                print(f"❌ Error en ChatGPT: {result.get('error', 'Unknown')}")
                therapist_response = result['response']

        except Exception as e:
            print(f"❌ Error llamando a ChatGPT: {e}")
            therapist_response = "Entiendo que estás pasando por un momento difícil. Estoy aquí para escucharte y apoyarte."

        print("⏹️ Grabación detenida")

        return {
            "text_transcribed": self.accumulated_text,
            "text_result": text_result,
            "face_summary": face_summary,
            "fusion_result": fusion_result,
            "llm_payload": llm_payload,
            "therapist_response": therapist_response,
            "fallback_used": fallback_used,
            "llm_output": json.dumps(llm_payload, indent=2, ensure_ascii=False)
        }

    def generate_frames(self):
        """Stream de video con análisis facial"""
        while True:
            ret, frame = self.camera.read()
            if not ret:
                continue

            try:
                processed_frame = self.face_system.frame_processing(frame)

                # Grabar si está activo
                if self.is_recording and self.video_writer:
                    self.video_writer.write(processed_frame)

                ret, buffer = cv2.imencode('.jpg', processed_frame)
                if not ret:
                    continue

                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            except Exception as e:
                print(f"❌ Error en frame: {e}")
                continue


# === RUTAS FLASK ===
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(video_stream.generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_session', methods=['POST'])
def start_session():
    data = request.get_json(force=True, silent=True) or {}
    text_mode = data.get('text_mode', True)
    
    video_stream.text_mode = text_mode
    video_stream.start_recording()
    
    return jsonify({
        'status': 'started',
        'message': 'Sesión iniciada',
        'session_id': SESSION_ID,
        'mode': 'text' if text_mode else 'voice'
    })
@app.route('/stop_session', methods=['POST'])
def stop_session():
    """Detiene sesión y retorna resultado fusionado + respuesta del LLM"""
    data = request.get_json() or {}
    
    # Obtener modo y texto del frontend
    text_mode = data.get('text_mode', True)
    accumulated_text = data.get('accumulated_text', '')
    
    # Actualizar modo
    video_stream.text_mode = text_mode
    
    # Detener grabación con el texto apropiado
    result = video_stream.stop_recording(text_from_chat=accumulated_text if text_mode else None)
    
    if result is None:
        return jsonify({'error': 'No hay sesión activa'}), 400
    
    return jsonify(result)

@app.route('/stop_voice', methods=['POST'])
def stop_voice():
    """Detiene la voz del gatito"""
    video_stream.voice_synthesizer.stop()
    return jsonify({'status': 'voice_stopped'})

@app.route('/get_current_emotions')
def get_current_emotions():
    """Retorna las emociones actuales del último frame procesado"""
    if video_stream.face_system.emotion_history_list:
        latest_emotions = video_stream.face_system.emotion_history_list[-1]
        return jsonify({'emotions': latest_emotions})
    else:
        return jsonify({'emotions': {}})


if __name__ == "__main__":
    try:
        # Inicializar componentes
        print("🚀 Inicializando componentes del sistema...")
        camera = Camera(0, 640, 480)
        face_system = EmotionRecognitionSystem()
        text_classifier = TextEmotionClassifier()
        speech_recognizer = SpeechRecognizer()
        fusion_engine = EmotionFusion()
        voice_synth = VoiceSynthesizer(voice="es-MX-DaliaNeural")

        # ✅ INICIALIZAR SISTEMA GRATUITO
        video_stream = IntegratedVideoStream(
            camera,
            face_system,
            text_classifier,
            speech_recognizer,
            fusion_engine,
            voice_synth
        )

        print("\n" + "="*60)
        print("🎭 Sistema FaceSense + ChatGPT Gratuito MEJORADO!")
        print("="*60)
        print("📊 Análisis emocional multimodal")
        print("🤖 Integrado con APIs gratuitas mejoradas")
        print("💰 100% GRATUITO - Sin costos de API")
        print("🌐 http://localhost:5001")
        print("="*60)
        print("MEJORAS IMPLEMENTADAS:")
        print("  ✅ APIs gratuitas más estables")
        print("  ✅ Mapeo completo de emociones (inglés/español)")
        print("  ✅ Respuestas de fallback mejoradas")
        print("  ✅ Timeout reducido para mejor respuesta")
        print("  ✅ Mensajes de debug más detallados")
        print("  ✅ Manejo robusto de emociones 'disgust' y otras")
        print("="*60)
        print("\n🎤 SISTEMA LISTO PARA ESCUCHAR")
        print("   Modos disponibles:")
        print("   - 🎤 Voz: Habla y el sistema transcribirá automáticamente")
        print("   - ⌨️  Texto: Escribe en el chat y presiona enviar")
        print("   - 📹 Análisis facial: Funciona en ambos modos")
        print("="*60)

        app.run(host='0.0.0.0', port=5001, debug=True, use_reloader=False)

    except Exception as e:
        print(f"❌ Error inicializando el sistema: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)