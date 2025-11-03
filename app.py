import os
import sys
import cv2
import json
from flask import Flask, Response, render_template, jsonify, request
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from examples.camera import Camera
from emotion_processor.main import EmotionRecognitionSystem

app = Flask(__name__)

class VideoStream:
    def __init__(self, cam: Camera, emotion_recognition_system: EmotionRecognitionSystem):
        self.camera = cam
        self.emotion_recognition_system = emotion_recognition_system
        self.video_recording = False
        self.emotion_recording = False
        self.video_writer = None
        self.out_file = "output.avi"

    def start_video_recording(self, filename="output.avi"):
        """Inicia grabación de video."""
        self.out_file = filename
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = 20.0
        frame_size = (int(self.camera.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      int(self.camera.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.video_writer = cv2.VideoWriter(self.out_file, fourcc, fps, frame_size)
        self.video_recording = True
        print(f"📹 Video recording started: {self.out_file}")

    def stop_video_recording(self):
        """Detiene grabación de video."""
        if self.video_recording:
            self.video_recording = False
            if self.video_writer is not None:
                self.video_writer.release()
                print(f"⏹️ Video recording stopped: {self.out_file}")
    
    def start_emotion_recording(self):
        """Inicia grabación del historial de emociones."""
        self.emotion_recording = True
        self.emotion_recognition_system.start_recording()
        print("📊 Emotion recording started")
    
    def stop_emotion_recording(self):
        """Detiene grabación del historial de emociones."""
        if self.emotion_recording:
            self.emotion_recording = False
            summary = self.emotion_recognition_system.stop_recording()
            filename = self.emotion_recognition_system.save_history()
            print(f"📊 Emotion recording stopped, saved to: {filename}")
            return summary
        return None

    def generate_frames(self):
        """Genera frames para streaming."""
        while True:
            ret, frame = self.camera.read()
            if not ret:
                print("Error: Cannot read frame from camera")
                continue
            
            try:
                result = self.emotion_recognition_system.frame_processing(frame)
                if isinstance(result, tuple):
                    frame = result[0]
                else:
                    frame = result
                    
            except Exception as e:
                print(f"Error in frame_processing: {e}")
                continue
            
            # Grabar video si está activo
            if self.video_recording and self.video_writer is not None:
                self.video_writer.write(frame)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                print("Error: Cannot encode frame to JPEG")
                continue
                
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(video_stream.generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_recording', methods=['POST'])
def start_recording():
    """Inicia AMBAS grabaciones: video y emociones."""
    video_stream.start_video_recording()
    video_stream.start_emotion_recording()
    return jsonify({
        'status': 'success',
        'message': 'Recording started (video + emotions)'
    })

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    """Detiene AMBAS grabaciones y retorna resumen de emociones."""
    video_stream.stop_video_recording()
    summary = video_stream.stop_emotion_recording()
    
    return jsonify({
        'status': 'success',
        'message': 'Recording stopped',
        'emotion_summary': summary
    })

@app.route('/get_current_summary', methods=['GET'])
def get_current_summary():
    """Obtiene resumen actual sin detener la grabación."""
    summary = video_stream.emotion_recognition_system.get_current_summary()
    return jsonify(summary)

@app.route('/download_history', methods=['GET'])
def download_history():
    """Descarga el historial de emociones más reciente."""
    try:
        summary = video_stream.emotion_recognition_system.get_current_summary()
        return jsonify(summary)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/favicon.ico')
def favicon():
    return app.send_static_file('favicon.ico')

if __name__ == "__main__":
    try:
        camera = Camera(0, 640, 480)
        print("✅ Camera initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize camera: {e}")
        sys.exit(1)
    
    try:
        emotion_recognition_system = EmotionRecognitionSystem()
        print("✅ EmotionRecognitionSystem initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize EmotionRecognitionSystem: {e}")
        sys.exit(1)
    
    video_stream = VideoStream(camera, emotion_recognition_system)
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)