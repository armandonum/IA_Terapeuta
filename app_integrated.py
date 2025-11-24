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

# === CONFIGURACIÓN DEL LLM (MINIMAX) ===
MINIMAX_API_KEY = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJHcm91cE5hbWUiOiJBcm1hbmRvIE51w7FleiBDb25kb3JpIiwiVXNlck5hbWUiOiJpbXBvZWFzeSIsIkFjY291bnQiOiIiLCJTdWJqZWN0SUQiOiIxOTkyNzE5ODI1NzkyNjcxOTQxIiwiUGhvbmUiOiIiLCJHcm91cElEIjoiMTk5MjcxOTgyNTc4ODQ3MzU0MSIsIlBhZ2VOYW1lIjoiIiwiTWFpbCI6ImFybWFuZG9udW5lejQwNEBnbWFpbC5jb20iLCJDcmVhdGVUaW1lIjoiMjAyNS0xMS0yNCAwNjo1NjoyMyIsIlRva2VuVHlwZSI6MSwiaXNzIjoibWluaW1heCJ9.MHEZcUX1k29SH1fGaAWBWL8YKNUGCaphunpaSFge1yjp0ClD6iI_MTtHShCv0xpBADdJSM20FIek3wONEXYZ0jRkTPoPMyJxYCE8wPXX9o1T_GGjGEg9dgKeyRBvyJTCqHSBwtjbhoRjhrKcJRAM1YujaVw2klEf7XoFy_OZa4QXmSTdNhN66oLay1NVEPAWZXQG2VBwT682rWMMrMmzjftpF8AUtxcZShp3cEUvOK5MsH-Fd_KEYQ6BSU56K8JX_J7H1DRhf8Y5AQ1GGvpwkuJ3jPOPfKyzSlEkLIsRMmHjzROJVVIklLzOK4Zi09NHBQ_xXy8NDsY8hQwoExPDaA"  # <-- PÉGALA AQUÍ
MINIMAX_URL = "https://api.minimax.chat/v1/text/chatcompletion_pro"

SESSION_ID = f"facesense_{os.getpid()}"


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

    def start_recording(self):
        self.is_recording = True
        self.accumulated_text = ""

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = 20.0
        frame_size = (
            int(self.camera.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.camera.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )

        base_name = "output"
        ext = ".avi"
        filename = f"{base_name}{ext}"
        idx = 1

        while os.path.exists(filename):
            filename = f"{base_name}_{idx}{ext}"
            idx += 1

        self.video_writer = cv2.VideoWriter(filename, fourcc, fps, frame_size)
        print(f"🎬 Grabando en: {filename}")

        if not self.text_mode:
            self.speech_recognizer.start_listening()

        self.face_system.start_recording()

        mode = "Texto" if self.text_mode else "Voz"
        print(f"✅ Grabación iniciada: Video + {mode} + Rostro")

    def stop_recording(self, text_from_chat=None):
        if not self.is_recording:
            return None

        self.is_recording = False

        if self.video_writer:
            self.video_writer.release()

        if self.text_mode and text_from_chat:
            self.accumulated_text = text_from_chat
        else:
            self.speech_recognizer.stop_listening()
            self.accumulated_text = self.speech_recognizer.get_all_text()

        face_summary = self.face_system.stop_recording()
        text_result = self.text_classifier.classify(self.accumulated_text)

        if face_summary and 'emotion_statistics' in face_summary:
            face_emotions = {
                emotion: stats['mean']
                for emotion, stats in face_summary['emotion_statistics'].items()
            }
        else:
            face_emotions = {"neutral": 50.0}

        fusion_result = self.fusion_engine.fuse(
            text_result['emotions'],
            face_emotions
        )
        self.latest_fusion = fusion_result

        llm_payload = self.fusion_engine.to_llm_format(
            fusion_result=fusion_result,
            text_transcribed=self.accumulated_text,
            session_id=SESSION_ID
        )

        # ====================================================
        #                🔥 LLAMADA A MINIMAX 🔥
        # ====================================================
        therapist_response = "Error al conectar con Minimax"

        try:
            headers = {
                "Authorization": f"Bearer {MINIMAX_API_KEY}",
                "Content-Type": "application/json"
            }

            minimax_payload = {
                "model": "Minimax-Text-01",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Eres un terapeuta psicológico cálido y empático. "
                            "Interpretas emociones humanas y respondes de forma comprensiva."
                        )
                    },
                    {
                        "role": "user",
                        "content": json.dumps(llm_payload, ensure_ascii=False)
                    }
                ]
            }

            print("\n📤 Enviando a Minimax:")
            print(json.dumps(minimax_payload, indent=2, ensure_ascii=False))

            response = requests.post(
                MINIMAX_URL,
                headers=headers,
                json=minimax_payload,
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()

                therapist_response = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "Sin respuesta")
                )

                print(f"✅ Respuesta Minimax: {therapist_response}")

                # === EL GATITO HABLA ===
                self.voice_synthesizer.speak(
                    text=therapist_response,
                    rate="+0%",
                    pitch="+7Hz"
                )

            else:
                therapist_response = f"Error Minimax {response.status_code}: {response.text}"
                print(therapist_response)

        except Exception as e:
            therapist_response = f"Error Minimax: {str(e)}"
            print(therapist_response)

        print("⏹️ Grabación detenida")

        # --------------------------------------------------------------
        return {
            "text_transcribed": self.accumulated_text,
            "text_result": text_result,
            "face_summary": face_summary,
            "fusion_result": fusion_result,
            "llm_payload": llm_payload,
            "therapist_response": therapist_response,
            "llm_output": json.dumps(llm_payload, indent=2, ensure_ascii=False)
        }

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

                frame_bytes = buffer.tobytes()
                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
                )

            except Exception as e:
                print(f"❌ Error en frame: {e}")
                continue


# === RUTAS ===
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(
        video_stream.generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/start_session', methods=['POST'])
def start_session():
    video_stream.start_recording()
    return jsonify({
        'status': 'started',
        'message': 'Sesión iniciada',
        'session_id': SESSION_ID
    })


@app.route('/stop_session', methods=['POST'])
def stop_session():
    data = request.get_json()

    text_mode = data.get('text_mode', True)
    accumulated_text = data.get('accumulated_text', '')

    video_stream.text_mode = text_mode

    result = video_stream.stop_recording(
        text_from_chat=accumulated_text if text_mode else None
    )

    if result is None:
        return jsonify({'error': 'No hay sesión activa'}), 400

    return jsonify(result)


@app.route('/stop_voice', methods=['POST'])
def stop_voice():
    video_stream.voice_synthesizer.stop()
    return jsonify({'status': 'voice_stopped'})


@app.route('/get_current_emotions')
def get_current_emotions():
    if video_stream.face_system.emotion_history_list:
        latest = video_stream.face_system.emotion_history_list[-1]
        return jsonify({'emotions': latest})
    return jsonify({'emotions': {}})


if __name__ == "__main__":
    try:
        camera = Camera(0, 640, 480)
        face_system = EmotionRecognitionSystem()
        text_classifier = TextEmotionClassifier()
        speech_recognizer = SpeechRecognizer()
        fusion_engine = EmotionFusion()
        voice_synth = VoiceSynthesizer(voice="es-MX-DaliaNeural")

        video_stream = IntegratedVideoStream(
            camera,
            face_system,
            text_classifier,
            speech_recognizer,
            fusion_engine,
            voice_synth
        )

        print("\n" + "=" * 60)
        print("🎭 FaceSense + Terapeuta IA (Minimax) listo!")
        print("=" * 60)
        print("🌐 http://localhost:5001\n")

        app.run(host='0.0.0.0', port=5001, debug=True, use_reloader=False)

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
