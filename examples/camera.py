import cv2

class Camera:
    def __init__(self, index: int, width: int, height: int):
        self.cap = None
        backends = [
            (cv2.CAP_V4L2, "V4L2"),
            (f'v4l2src device=/dev/video{index} ! videoconvert ! appsink', cv2.CAP_GSTREAMER, "GSTREAMER"),
            (cv2.CAP_ANY, "ANY")
        ]
        for backend, name in backends:
            if isinstance(backend, str):  # Para GSTREAMER
                self.cap = cv2.VideoCapture(backend, cv2.CAP_GSTREAMER)
            else:
                self.cap = cv2.VideoCapture(index, backend)
            if self.cap.isOpened():
                print(f"Camera opened successfully at index {index} with {name}")
                break
        else:
            if self.cap is not None:
                self.cap.release()
            print(f"Error: Cannot open camera at index {index} with any backend")
            raise Exception("No camera found")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self):
        if self.cap is None or not self.cap.isOpened():
            print("Error: Camera not accessible")
            return False, None
        ret, frame = self.cap.read()
        if not ret:
            print("Error: Cannot read frame from camera")
        return ret, frame

    def release(self):
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            print("Camera released")