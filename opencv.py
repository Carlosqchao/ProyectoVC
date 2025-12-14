import cv2
import mediapipe as mp
import socket
import json

# ------------ CONFIG ------------
GODOT_IP = "127.0.0.1"
GODOT_PORT = 4242
SEND_EVERY_N_FRAMES = 1
# -------------------------------

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("No se pudo abrir la webcam")
    exit(1)

frame_count = 0

with mp_hands.Hands(
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo leer frame de la webcam")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        hands_data = []

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks,
                results.multi_handedness
            ):
                hand_label = handedness.classification[0].label  # "Left"/"Right"

                # Landmarks del dedo índice: 5 (base) a 8 (punta)
                index_points = [5, 6, 7, 8]
                xs = []
                ys = []
                for idx in index_points:
                    lm = hand_landmarks.landmark[idx]
                    xs.append(int(lm.x * w))
                    ys.append(int(lm.y * h))

                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                center_x = int((x_min + x_max) / 2)
                center_y = int((y_min + y_max) / 2)
                length_x = x_max - x_min
                length_y = y_max - y_min

                hands_data.append({
                    "x": center_x,
                    "y": center_y,
                    "len_x": length_x,
                    "len_y": length_y,
                    "label": hand_label   # "Left" / "Right"
                })

                # Dibujo opcional de todo el dedo índice para debug
                for idx in index_points:
                    lm = hand_landmarks.landmark[idx]
                    px = int(lm.x * w)
                    py = int(lm.y * h)
                    cv2.circle(frame, (px, py), 4, (0, 255, 0), -1)

                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

        # Enviar a Godot
        if hands_data and frame_count % SEND_EVERY_N_FRAMES == 0:
            data = {
                "hands": hands_data,
                "w": w,
                "h": h
            }
            msg = json.dumps(data).encode("utf-8")
            sock.sendto(msg, (GODOT_IP, GODOT_PORT))

        frame_count += 1

        cv2.imshow("MediaPipe Index Finger - Envío a Godot", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
sock.close()
