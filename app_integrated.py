import os
import sys
import cv2
import json
import requests  # AÑADIR ESTA IMPORTACIÓN
from flask import Flask, Response, render_template, jsonify, request
from examples.camera import Camera
from emotion_processor.main import EmotionRecognitionSystem
from text_emotion_classifier import TextEmotionClassifier
from speech_recognizer import SpeechRecognizer
from emotion_fusion import EmotionFusion
from voice_synthesizer import VoiceSynthesizer

app = Flask(__name__)

# === CONFIGURACIÓN DEL LLM ===
LLM_API_URL = 'https://hydrotactic-domical-rigoberto.ngrok-free.dev/chat'
SESSION_ID = f"facesense_{os.getpid()}"


class IntegratedVideoStream:
    # def __init__(self, cam, face_system, text_classifier, speech_rec, fusion):
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
        # Crear nombre de archivo incremental para no sobreescribir grabaciones previas
        base_name = "output"
        ext = ".avi"
        filename = f"{base_name}{ext}"
        idx = 1
        # Si existe, probar output_1.avi, output_2.avi, ...
        while os.path.exists(filename):
            filename = f"{base_name}_{idx}{ext}"
            idx += 1
        self.video_writer = cv2.VideoWriter(filename, fourcc, fps, frame_size)
        print(f"🎬 Grabando en: {filename}")

        # Iniciar voz solo si no está en modo texto
        if not self.text_mode:
            self.speech_recognizer.start_listening()

        # Iniciar historial facial
        self.face_system.start_recording()

        mode = "Texto" if self.text_mode else "Voz"
        print(f"✅ Grabación iniciada: Video + {mode} + Rostro")

    def stop_recording(self, text_from_chat=None):
        """Detiene grabación y genera resultado fusionado + llama al LLM"""
        if not self.is_recording:
            return None

        self.is_recording = False

        # Detener video
        if self.video_writer:
            self.video_writer.release()

        # Obtener texto según el modo
        if self.text_mode and text_from_chat:
            self.accumulated_text = text_from_chat
        else:
            self.speech_recognizer.stop_listening()
            self.accumulated_text = self.speech_recognizer.get_all_text()

        # Detener historial facial
        face_summary = self.face_system.stop_recording()

        # Clasificar texto
        text_result = self.text_classifier.classify(self.accumulated_text)

        # Preparar emociones faciales (convertir de 0-100 a 0-1)
        if face_summary and 'emotion_statistics' in face_summary:
            face_emotions = {
                emotion: stats['mean']  # Ya están en 0-100
                for emotion, stats in face_summary['emotion_statistics'].items()
            }
        else:
            face_emotions = {"neutral": 50.0}

        # Fusionar resultados
        fusion_result = self.fusion_engine.fuse(
            text_result['emotions'],
            face_emotions
        )
        self.latest_fusion = fusion_result

        # Generar payload para el LLM
        llm_payload = self.fusion_engine.to_llm_format(
            fusion_result=fusion_result,
            text_transcribed=self.accumulated_text,
            session_id=SESSION_ID
        )

        # === LLAMAR AL LLM ===
        therapist_response = "Error al conectar con el terapeuta"
        try:
            print(f"🔄 Enviando al LLM: {LLM_API_URL}")
            print(f"📤 Payload: {json.dumps(llm_payload, indent=2, ensure_ascii=False)}")
            
            response = requests.post(
                LLM_API_URL,
                json=llm_payload,
                timeout=30
            )
  
  

            if response.status_code == 200:
                llm_response = response.json()
                therapist_response = llm_response.get('response', 'Sin respuesta del LLM')
                print(f"✅ Respuesta del LLM: {therapist_response}")
                # ===  HACER QUE EL GATITO HABLE ===
                print("🐱 Gatito está hablando...")
                self.voice_synthesizer.speak(
                    text=therapist_response,
                    rate="+0%",      # Velocidad normal
                    pitch="+7Hz"     # Tono tierno de gatito
                )
            else:
                print(f"❌ Error LLM: Status {response.status_code}")
                therapist_response = f"Error del servidor LLM: {response.status_code}"
            
                
        except requests.exceptions.Timeout:
            print("⏱️ Timeout al llamar al LLM")
            therapist_response = "El terapeuta está tardando mucho en responder. Intenta de nuevo."
        except Exception as e:
            print(f"❌ Error llamando al LLM: {e}")
            therapist_response = f"Error de conexión: {str(e)}"

        print("⏹️ Grabación detenida")

        return {
            "text_transcribed": self.accumulated_text,
            "text_result": text_result,
            "face_summary": face_summary,
            "fusion_result": fusion_result,
            "llm_payload": llm_payload,
            "therapist_response": therapist_response,  # RESPUESTA DEL LLM
            "llm_output": json.dumps(llm_payload, indent=2, ensure_ascii=False)  # Para debug
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


# === RUTAS ===
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(video_stream.generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start_session', methods=['POST'])
def start_session():
    """Inicia sesión: Video + Voz/Texto + Rostro"""
    video_stream.start_recording()
    return jsonify({
        'status': 'started', 
        'message': 'Sesión iniciada',
        'session_id': SESSION_ID
    })


@app.route('/stop_session', methods=['POST'])
def stop_session():
    """Detiene sesión y retorna resultado fusionado + respuesta del LLM"""
    data = request.get_json()
    
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
        camera = Camera(0, 640, 480)
        face_system = EmotionRecognitionSystem()
        text_classifier = TextEmotionClassifier()
        speech_recognizer = SpeechRecognizer()
        fusion_engine = EmotionFusion()
        voice_synth = VoiceSynthesizer(voice="es-MX-DaliaNeural")  # Voz tierna de gatito

        video_stream = IntegratedVideoStream(
            camera,
            face_system,
            text_classifier,
            speech_recognizer,
            fusion_engine,
            voice_synth
        )

        print("\n" + "="*60)
        print("🎭 Sistema FaceSense + Terapeuta IA Listo!")
        print("="*60)
        print("📊 Análisis emocional multimodal")
        print("🤖 Integrado con LLM Terapeuta")
        print(f"🔗 LLM API: {LLM_API_URL}")
        print("🌐 http://localhost:5001\n")
        print("Funcionalidades:")
        print("  ✅ Análisis facial en tiempo real")
        print("  ✅ Reconocimiento de voz (modo micrófono)")
        print("  ✅ Entrada de texto (modo chat)")
        print("  ✅ Fusión emocional con historial")
        print("  ✅ Respuestas del terapeuta IA\n")
        print("="*60)

        app.run(host='0.0.0.0', port=5001, debug=True, use_reloader=False)

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)