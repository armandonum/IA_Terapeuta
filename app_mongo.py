# app.py
import os
import sys
import cv2
import json
import base64
import requests
from datetime import datetime
from flask import Flask, Response, render_template, jsonify, request
from pymongo import MongoClient
from bson import ObjectId
from examples.camera import Camera
from emotion_processor.main import EmotionRecognitionSystem
from text_emotion_classifier import TextEmotionClassifier
from coqui_tts_natural import NaturalSpanishTTS  # ← Archivo corregido abajo
from speech_recognizer import SpeechRecognizer
from emotion_fusion import EmotionFusion
from voice_synthesizer import VoiceSynthesizer

app = Flask(__name__)

# === CONFIGURACIÓN MONGODB ===
MONGO_URI = "mongodb://admin:password@localhost:27017/"
client = MongoClient(MONGO_URI)
db = client['facesense_db']
sessions_collection = db['therapy_sessions']

# === CONFIGURACIÓN DEL LLM ===
LLM_API_URL = 'https://hydrotactic-domical-rigoberto.ngrok-free.dev/chat'
SESSION_ID = f"facesense_{os.getpid()}"

class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

app.json_encoder = JSONEncoder

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
        self.current_session_id = None
        self.current_video_path = None

    def start_recording(self):
        self.is_recording = True
        self.accumulated_text = ""
        self.video_writer = None

        # Crear sesión en MongoDB
        session_doc = {
            'session_id': SESSION_ID,
            'start_time': datetime.utcnow(),
            'status': 'recording',
            'interactions': [],
            'emotion_history': [],
            'video_path': None
        }
        result = sessions_collection.insert_one(session_doc)
        self.current_session_id = result.inserted_id

        # Configurar video
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = 20.0
        frame_size = (int(self.camera.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      int(self.camera.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        os.makedirs('session_videos', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"session_videos/session_{timestamp}.avi"
        self.current_video_path = filename
        self.video_writer = cv2.VideoWriter(filename, fourcc, fps, frame_size)

        print(f"Grabando en: {filename}")

        if not self.text_mode:
            self.speech_recognizer.start_listening()

        self.face_system.start_recording()
        mode = "Texto" if self.text_mode else "Voz"
        print(f"Grabación iniciada: Video + {mode} + Rostro")

    def stop_recording(self, text_from_chat=None):
        if not self.is_recording:
            return None

        self.is_recording = False

        # Detener video
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None

        # Obtener texto
        if self.text_mode and text_from_chat:
            self.accumulated_text = text_from_chat
        else:
            self.speech_recognizer.stop_listening()
            self.accumulated_text = self.speech_recognizer.get_all_text()

        # Detener análisis facial
        face_summary = self.face_system.stop_recording()

        # Clasificar emociones del texto
        text_result = self.text_classifier.classify(self.accumulated_text or "silencio")

        # Emociones faciales
        face_emotions = {"neutral": 50.0}
        if face_summary and 'emotion_statistics' in face_summary:
            face_emotions = {
                emotion: stats['mean']
                for emotion, stats in face_summary['emotion_statistics'].items()
            }

        # Fusión
        fusion_result = self.fusion_engine.fuse(
            text_result['emotions'],
            face_emotions
        )
        self.latest_fusion = fusion_result

        # Contexto previo
        previous_context = self.get_llm_context()

        # Payload para LLM
        llm_payload = self.fusion_engine.to_llm_format(
            fusion_result=fusion_result,
            text_transcribed=self.accumulated_text,
            session_id=SESSION_ID
        )
        if previous_context:
            llm_payload['previous_interactions'] = previous_context

        # Llamar al LLM
        therapist_response = "Lo siento, no pude conectar con el terapeuta."
        try:
            print("Enviando al LLM...")
            response = requests.post(LLM_API_URL, json=llm_payload, timeout=30)
            if response.status_code == 200:
                llm_response = response.json()
                therapist_response = llm_response.get('response', 'Sin respuesta.')
                print(f"Respuesta del terapeuta: {therapist_response[:100]}...")
            else:
                therapist_response = f"Error del servidor: {response.status_code}"
        except Exception as e:
            print(f"Error LLM: {e}")
            therapist_response = "Hubo un problema de conexión con el terapeuta."

        # Reproducir voz
        if therapist_response:
            self.voice_synthesizer.speak_therapeutic(therapist_response, emotion="empathy")

        # Guardar en MongoDB
        interaction_doc = {
            'timestamp': datetime.utcnow(),
            'user_message': self.accumulated_text,
            'therapist_response': therapist_response,
            'text_emotions': text_result['emotions'],
            'face_emotions': face_emotions,
            'fusion_result': fusion_result,
            'emotion_statistics': face_summary.get('emotion_statistics', {}) if face_summary else {}
        }

        sessions_collection.update_one(
            {'_id': self.current_session_id},
            {
                '$push': {'interactions': interaction_doc},
                '$set': {
                    'end_time': datetime.utcnow(),
                    'status': 'completed',
                    'video_path': self.current_video_path
                }
            }
        )

        print(f"Sesión guardada: {self.current_video_path}")
        return {
            "session_db_id": str(self.current_session_id),
            "text_transcribed": self.accumulated_text,
            "therapist_response": therapist_response,
            "fusion_result": fusion_result,
            "video_path": self.current_video_path
        }

    def get_llm_context(self):
        if not self.current_session_id:
            return None
        session = sessions_collection.find_one({'_id': self.current_session_id})
        if not session or 'interactions' not in session:
            return None
        interactions = session['interactions'][-3:]
        return [
            {
                'user': i['user_message'],
                'therapist': i['therapist_response'],
                'emotions': i['fusion_result']
            } for i in interactions
        ]

    def generate_frames(self):
        while True:
            ret, frame = self.camera.read()
            if not ret:
                continue
            try:
                processed_frame = self.face_system.frame_processing(frame)
                if self.is_recording and self.video_writer:
                    self.video_writer.write(processed_frame)

                ret, buffer = cv2.imencode('.jpg', processed_frame)
                if not ret:
                    continue
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            except Exception as e:
                print(f"Error en frame: {e}")
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
    video_stream.start_recording()
    return jsonify({
        'status': 'started',
        'session_id': SESSION_ID,
        'db_session_id': str(video_stream.current_session_id)
    })

@app.route('/stop_session', methods=['POST'])
def stop_session():
    data = request.get_json() or {}
    text_mode = data.get('text_mode', True)
    accumulated_text = data.get('accumulated_text', '')
    video_stream.text_mode = text_mode
    result = video_stream.stop_recording(text_from_chat=accumulated_text if text_mode else None)
    if not result:
        return jsonify({'error': 'No hay sesión activa'}), 400
    return jsonify(result)

@app.route('/stop_voice', methods=['POST'])
def stop_voice():
    video_stream.voice_synthesizer.stop()
    return jsonify({'status': 'voice_stopped'})

@app.route('/get_current_emotions')
def get_current_emotions():
    if video_stream.face_system.emotion_history_list:
        return jsonify({'emotions': video_stream.face_system.emotion_history_list[-1]})
    return jsonify({'emotions': {}})

@app.route('/get_session_history/<session_id>')
def get_session_history(session_id):
    try:
        session = sessions_collection.find_one({'_id': ObjectId(session_id)})
        if not session:
            return jsonify({'error': 'Sesión no encontrada'}), 404
        session['_id'] = str(session['_id'])
        return jsonify(session)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_all_sessions')
def get_all_sessions():
    sessions = list(sessions_collection.find().sort('start_time', -1))
    for s in sessions:
        s['_id'] = str(s['_id'])
    return jsonify({'sessions': sessions})

if __name__ == "__main__":
    try:
        camera = Camera(0, 640, 480)
        face_system = EmotionRecognitionSystem()
        text_classifier = TextEmotionClassifier()
        speech_recognizer = SpeechRecognizer()
        fusion_engine = EmotionFusion()
        voice_synth = NaturalSpanishTTS()  # ← Usa el corregido

        video_stream = IntegratedVideoStream(
            camera, face_system, text_classifier,
            speech_recognizer, fusion_engine, voice_synth
        )

        print("\n" + "="*60)
        print("Sistema FaceSense + Terapeuta IA con Voz Natural")
        print("="*60)
        print("Análisis emocional multimodal + TTS natural en español")
        print("MongoDB + Contexto de sesiones + Grabación de video")
        print(f"LLM: {LLM_API_URL}")
        print("http://localhost:5001")
        print("="*60)

        app.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False, threaded=True)

    except Exception as e:
        print(f"Error crítico: {e}")
        sys.exit(1)