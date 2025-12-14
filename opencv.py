import cv2
import mediapipe as mp
import socket
import json
<<<<<<< HEAD
import numpy as np
import math
=======
>>>>>>> cf12267087179fec7856bafc0df47ba69618d1c8

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

<<<<<<< HEAD
FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_PIPS = [3, 6, 10, 14, 18]

def get_finger_states(hand_landmarks):
    lm = hand_landmarks.landmark
    finger_states = {
        "thumb": False,
        "index": False,
        "middle": False,
        "ring": False,
        "pinky": False,
    }

    if lm[FINGER_TIPS[0]].y < lm[FINGER_PIPS[0]].y:
        finger_states["thumb"] = True
    if lm[FINGER_TIPS[1]].y < lm[FINGER_PIPS[1]].y:
        finger_states["index"] = True
    if lm[FINGER_TIPS[2]].y < lm[FINGER_PIPS[2]].y:
        finger_states["middle"] = True
    if lm[FINGER_TIPS[3]].y < lm[FINGER_PIPS[3]].y:
        finger_states["ring"] = True
    if lm[FINGER_TIPS[4]].y < lm[FINGER_PIPS[4]].y:
        finger_states["pinky"] = True

    return finger_states

def angle_3pts(a, b, c):
    v1 = np.array([a.x - b.x, a.y - b.y])
    v2 = np.array([c.x - b.x, c.y - b.y])
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm == 0:
        return 0.0
    cos = np.clip(dot / norm, -1.0, 1.0)
    return math.degrees(math.acos(cos))

def finger_is_straight(lm, joints, tol_deg=30.0):
    a = lm[joints[0]]
    b = lm[joints[1]]
    c = lm[joints[2]]
    ang = angle_3pts(a, b, c)
    return ang > (180.0 - tol_deg)

def thumb_is_extended(lm, threshold=0.1):
    wrist = lm[0]
    thumb_tip = lm[4]
    dx = thumb_tip.x - wrist.x
    dy = thumb_tip.y - wrist.y
    dist = math.sqrt(dx*dx + dy*dy)
    return dist > threshold

def classify_hand_shape(hand_landmarks):
    lm = hand_landmarks.landmark
    f = get_finger_states(hand_landmarks)

    index_joints = [5, 6, 7, 8]

    # index
    if f["index"] and not f["middle"] and not f["ring"] and not f["pinky"]:
        if not thumb_is_extended(lm, threshold=0.08):
            return "index", False

    # rock
    if f["index"] and f["pinky"] and not f["middle"] and not f["ring"]:
        return "rock", False

    # L mejorada con detección de invertida
    if f["index"] and not f["ring"] and not f["pinky"]:
        index_straight = finger_is_straight(lm, index_joints, tol_deg=45.0)
        thumb_ext = thumb_is_extended(lm, threshold=0.08)

        if index_straight and thumb_ext:
            base = lm[5]
            tip_index = lm[8]
            tip_thumb = lm[4]

            v_index = np.array([tip_index.x - base.x, tip_index.y - base.y])
            v_thumb = np.array([tip_thumb.x - base.x, tip_thumb.y - base.y])

            dot = np.dot(v_index, v_thumb)
            norm = np.linalg.norm(v_index) * np.linalg.norm(v_thumb)
            if norm > 0:
                cos = np.clip(dot / norm, -1.0, 1.0)
                ang = math.degrees(math.acos(cos))
                if 30.0 <= ang <= 150.0:
                    # Detectar si está invertida: pulgar debajo del índice
                    inverted = tip_thumb.y > tip_index.y
                    return "L", inverted

    # peace
    if f["index"] and f["middle"] and not f["ring"] and not f["pinky"]:
        return "peace", False

    return "unknown", False

def draw_rotated_box_for_finger(frame, lm, finger_indices, color=(255, 0, 0), thickness=2):
    pts = []
    for idx in finger_indices:
        l = lm[idx]
        px = int(l.x * frame.shape[1])
        py = int(l.y * frame.shape[0])
        pts.append([px, py])
        cv2.circle(frame, (px, py), 4, (0, 255, 0), -1)

    pts = np.array(pts, dtype=np.int32)
    rect = cv2.minAreaRect(pts)
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    cv2.polylines(frame, [box], True, color, thickness)

    return pts

=======
>>>>>>> cf12267087179fec7856bafc0df47ba69618d1c8
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
<<<<<<< HEAD
                hand_label = handedness.classification[0].label
                lm = hand_landmarks.landmark

                shape, inverted = classify_hand_shape(hand_landmarks)

                if shape == "index":
                    fingers_for_shape = [[5, 6, 7, 8]]
                elif shape == "rock":
                    fingers_for_shape = [[5, 6, 7, 8], [17, 18, 19, 20]]
                elif shape == "L":
                    fingers_for_shape = [[1, 2, 3, 4], [5, 6, 7, 8]]
                elif shape == "peace":
                    fingers_for_shape = [[5, 6, 7, 8], [9, 10, 11, 12]]
                else:
                    fingers_for_shape = [[5, 6, 7, 8]]

                all_pts = []

                for finger_indices in fingers_for_shape:
                    pts = draw_rotated_box_for_finger(frame, lm, finger_indices)
                    all_pts.append(pts)

                if shape == "rock":
                    base_index = lm[5]
                    base_pinky = lm[17]
                    bx1 = int(base_index.x * w)
                    by1 = int(base_index.y * h)
                    bx2 = int(base_pinky.x * w)
                    by2 = int(base_pinky.y * h)
                    cv2.line(frame, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
                    all_pts.append(np.array([[bx1, by1], [bx2, by2]], dtype=np.int32))

                if not all_pts:
                    continue

                all_pts = np.concatenate(all_pts, axis=0)
                x_min = int(np.min(all_pts[:, 0]))
                x_max = int(np.max(all_pts[:, 0]))
                y_min = int(np.min(all_pts[:, 1]))
                y_max = int(np.max(all_pts[:, 1]))

=======
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
>>>>>>> cf12267087179fec7856bafc0df47ba69618d1c8
                center_x = int((x_min + x_max) / 2)
                center_y = int((y_min + y_max) / 2)
                length_x = x_max - x_min
                length_y = y_max - y_min

<<<<<<< HEAD
                base = lm[5]
                tip_index = lm[8]
                vx = tip_index.x - base.x
                vy = tip_index.y - base.y
                angle_rad = math.atan2(vy, vx)
                angle_deg = math.degrees(angle_rad)

                hand_dict = {
=======
                hands_data.append({
>>>>>>> cf12267087179fec7856bafc0df47ba69618d1c8
                    "x": center_x,
                    "y": center_y,
                    "len_x": length_x,
                    "len_y": length_y,
<<<<<<< HEAD
                    "label": hand_label,
                    "shape": shape,
                    "angle": angle_deg,
                    "inverted": inverted  # NUEVO campo
                }

                hands_data.append(hand_dict)

                shape_text = shape + (" INV" if inverted else "")
                cv2.putText(
                    frame, shape_text, (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
                )

=======
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
>>>>>>> cf12267087179fec7856bafc0df47ba69618d1c8
        if hands_data and frame_count % SEND_EVERY_N_FRAMES == 0:
            data = {
                "hands": hands_data,
                "w": w,
                "h": h
            }
            msg = json.dumps(data).encode("utf-8")
            sock.sendto(msg, (GODOT_IP, GODOT_PORT))

        frame_count += 1

<<<<<<< HEAD
        cv2.imshow("MediaPipe Gestos - Envío a Godot", frame)
=======
        cv2.imshow("MediaPipe Index Finger - Envío a Godot", frame)
>>>>>>> cf12267087179fec7856bafc0df47ba69618d1c8
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
sock.close()
