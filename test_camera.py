import cv2

def test_camera(index):
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"Error: No se puede abrir la cámara en el índice {index} con CAP_V4L2")
        cap = cv2.VideoCapture(index, cv2.CAP_ANY)
        if not cap.isOpened():
            print(f"Error: No se puede abrir la cámara en el índice {index} con CAP_ANY")
            return
    print(f"Cámara abierta en el índice {index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Error: No se puede leer el frame en el índice {index}")
            break
        cv2.imshow(f'Camera Test Index {index}', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_camera(0)