import cv2
import mediapipe as mp
import socket
import json
import numpy as np
import math
import time

# ------------ CONFIG ------------
GODOT_IP = "127.0.0.1"
GODOT_PORT = 4242
SEND_EVERY_N_FRAMES = 1

MAX_MISSING_FRAMES = 5
ALPHA_SMOOTH = 0.5

MODEL_PATH = "hand_landmarker.task"  # ruta al modelo de MediaPipe Tasks
# -------------------------------

# --- MediaPipe Tasks (nueva API) ---
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
ImageMP = mp.Image

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("No se pudo abrir la webcam")
    exit(1)

frame_count = 0

FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_PIPS = [3, 6, 10, 14, 18]
FINGER_MCPS = [2, 5, 9, 13, 17]

# -------------------------------------------------------------------
# UTILIDADES ÁNGULOS / SUAVIZADO
# -------------------------------------------------------------------
def normalize_angle(angle: float) -> float:
    """Normaliza un ángulo al rango [-180, 180)."""
    while angle >= 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle


def smooth_angle_circular(new_angle, old_angle, alpha):
    """Suaviza ángulos considerando discontinuidad circular."""
    new_angle = normalize_angle(new_angle)
    old_angle = normalize_angle(old_angle)

    diff = new_angle - old_angle
    if diff > 180:
        diff -= 360
    elif diff < -180:
        diff += 360

    return normalize_angle(old_angle + alpha * diff)

# -------------------------------------------------------------------
# FUNCIONES DE GESTOS 2D (PORTADAS DEL PRIMER CÓDIGO)
# -------------------------------------------------------------------
def get_finger_states_2d_from_lm(lm, handedness: str):
    """
    Versión adaptada de get_finger_states_2d del primer script.
    'lm' es la lista de 21 landmarks (HandLandmarkerResult).
    """
    # En imagen de webcam flippeada, el eje Y crece hacia abajo.
    # Dedo extendido: tip más arriba (y menor) que pip y mcp.
    def is_extended_y(tip_idx, pip_idx, mcp_idx):
        tip_y = lm[tip_idx].y
        pip_y = lm[pip_idx].y
        mcp_y = lm[mcp_idx].y
        return tip_y < pip_y and tip_y < mcp_y

    # Pulgar: tratamos derecha/izquierda distinto en X.
    thumb_tip = lm[4]
    thumb_ip = lm[3]
    thumb_mcp = lm[2]

    if handedness == "Right":
        thumb_extended = thumb_tip.x < thumb_ip.x < thumb_mcp.x
    else:  # "Left"
        thumb_extended = thumb_tip.x > thumb_ip.x > thumb_mcp.x

    finger_states = {
        "thumb": thumb_extended,
        "index": is_extended_y(8, 6, 5),
        "middle": is_extended_y(12, 10, 9),
        "ring": is_extended_y(16, 14, 13),
        "pinky": is_extended_y(20, 18, 17),
    }
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


def finger_is_curved(lm, joints, min_bend_deg=40.0, max_bend_deg=120.0):
    """
    'joints' aquí se espera como [mcp, pip, tip] igual que en tu primer código.
    """
    a = lm[joints[0]]
    b = lm[joints[1]]
    c = lm[joints[2]]
    ang = angle_3pts(a, b, c)
    return min_bend_deg <= ang <= max_bend_deg


def classify_hand_shape_2d(lm, handedness: str):
    """
    Port de classify_hand_shape del primer script, adaptado a:
    - lm: lista de landmarks
    - devuelve (shape, inverted)
    """
    f = get_finger_states_2d_from_lm(lm, handedness)

    index_joints = [5, 6, 8]
    middle_joints = [9, 10, 12]

    # --- índice: solo índice extendido ---
    extended = {
        "thumb": f["thumb"],
        "index": f["index"],
        "middle": f["middle"],
        "ring": f["ring"],
        "pinky": f["pinky"],
    }
    num_extended = sum(1 for v in extended.values() if v)
    if num_extended == 1 and extended["index"]:
        return "index", False

    # --- C: índice y medio curvados, anular y meñique no extendidos ---
    index_curved = finger_is_curved(lm, index_joints, 40.0, 120.0)
    middle_curved = finger_is_curved(lm, middle_joints, 40.0, 120.0)
    if index_curved and middle_curved and not f["ring"] and not f["pinky"]:
        return "C", False

    # rock
    if f["index"] and f["pinky"] and not f["middle"] and not f["ring"]:
        return "rock", False

    # peace
    if f["index"] and f["middle"] and not f["ring"] and not f["pinky"]:
        return "peace", False

    return "unknown", False

# -------------------------------------------------------------------
# DIBUJO
# -------------------------------------------------------------------
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

# -------------------- ESTADO POR MANO --------------------
last_hands_data = {}
missing_frames_per_hand = {}
smooth_x = {}
smooth_y = {}
smooth_angle = {}
last_shape = {}
last_inverted = {}

# -------------------- CALLBACK TASKS ---------------------
last_result = None

def result_callback(result: HandLandmarkerResult, output_image: ImageMP, timestamp_ms: int):
    global last_result
    last_result = result

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=1,
    result_callback=result_callback,
)

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo leer frame de la webcam")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # enviar a MediaPipe Tasks
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = ImageMP(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = int(time.time() * 1000)
        landmarker.detect_async(mp_image, timestamp_ms)

        detected_hands = set()
        hands_data = []

        if last_result and last_result.hand_landmarks:
            for hand_idx, hand_lm in enumerate(last_result.hand_landmarks):
                lm = hand_lm  # lista de 21 landmarks
                handedness_list = last_result.handedness[hand_idx]
                hand_label = handedness_list[0].category_name  # "Left" o "Right"
                detected_hands.add(hand_label)

                # ---- NUEVO: usar la clasificación 2D portada del primer script ----
                raw_shape, raw_inverted = classify_hand_shape_2d(lm, hand_label)

                # inicialización
                if hand_label not in last_shape:
                    last_shape[hand_label] = "index"
                    last_inverted[hand_label] = False

                # mantener último gesto estable (no pisar con "unknown")
                if raw_shape != "unknown":
                    last_shape[hand_label] = raw_shape
                    last_inverted[hand_label] = raw_inverted

                shape = last_shape[hand_label]
                inverted = last_inverted[hand_label]

                # dedos usados para el bounding, incluyendo "C"
                if shape == "index":
                    fingers_for_shape = [[5, 6, 7, 8]]
                elif shape == "rock":
                    fingers_for_shape = [[5, 6, 7, 8], [17, 18, 19, 20]]
                elif shape == "peace":
                    fingers_for_shape = [[5, 6, 7, 8], [9, 10, 11, 12]]
                elif shape == "C":
                    fingers_for_shape = [[5, 6, 7, 8], [9, 10, 11, 12]]
                else:
                    fingers_for_shape = [[5, 6, 7, 8]]

                all_pts = []
                for finger_indices in fingers_for_shape:
                    pts = draw_rotated_box_for_finger(frame, lm, finger_indices)
                    all_pts.append(pts)

                # línea extra para rock (como en ambos scripts)
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

                center_x = int((x_min + x_max) / 2)
                center_y = int((y_min + y_max) / 2)
                length_x = x_max - x_min
                length_y = y_max - y_min

                base = lm[5]
                tip_index = lm[8]
                vx = tip_index.x - base.x
                vy = tip_index.y - base.y
                angle_rad = math.atan2(vy, vx)
                angle_deg = math.degrees(angle_rad)

                # suavizado
                if hand_label not in smooth_x:
                    smooth_x[hand_label] = center_x
                    smooth_y[hand_label] = center_y
                    smooth_angle[hand_label] = angle_deg
                else:
                    smooth_x[hand_label] = int(
                        ALPHA_SMOOTH * center_x
                        + (1 - ALPHA_SMOOTH) * smooth_x[hand_label]
                    )
                    smooth_y[hand_label] = int(
                        ALPHA_SMOOTH * center_y
                        + (1 - ALPHA_SMOOTH) * smooth_y[hand_label]
                    )
                    smooth_angle[hand_label] = smooth_angle_circular(
                        angle_deg, smooth_angle[hand_label], ALPHA_SMOOTH
                    )

                missing_frames_per_hand[hand_label] = 0

                hand_dict = {
                    "x": smooth_x[hand_label],
                    "y": smooth_y[hand_label],
                    "len_x": length_x,
                    "len_y": length_y,
                    "label": hand_label,
                    "shape": shape,
                    "angle": smooth_angle[hand_label],
                    "inverted": inverted,
                }

                hands_data.append(hand_dict)
                last_hands_data[hand_label] = hand_dict

                shape_text = shape + (" INV" if inverted else "")
                cv2.putText(
                    frame,
                    f"{hand_label}: {shape_text}",
                    (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )

        # manos no detectadas este frame
        for hand_label in list(last_hands_data.keys()):
            if hand_label not in detected_hands:
                if hand_label not in missing_frames_per_hand:
                    missing_frames_per_hand[hand_label] = 0

                missing_frames_per_hand[hand_label] += 1

                if missing_frames_per_hand[hand_label] <= MAX_MISSING_FRAMES:
                    hands_data.append(last_hands_data[hand_label])
                else:
                    del last_hands_data[hand_label]
                    del missing_frames_per_hand[hand_label]
                    if hand_label in smooth_x:
                        del smooth_x[hand_label]
                        del smooth_y[hand_label]
                        del smooth_angle[hand_label]
                        del last_shape[hand_label]
                        del last_inverted[hand_label]

        # envío a Godot
        if hands_data and frame_count % SEND_EVERY_N_FRAMES == 0:
            data = {
                "hands": hands_data,
                "w": w,
                "h": h,
            }
            msg = json.dumps(data).encode("utf-8")
            sock.sendto(msg, (GODOT_IP, GODOT_PORT))

        frame_count += 1

        cv2.imshow("MediaPipe Gestos - Envío a Godot (Tasks)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
sock.close()
