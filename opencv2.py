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
# GESTOS (adaptado para HandLandmarkerResult: lm es lista de 21 landmarks)
# -------------------------------------------------------------------
def get_palm_normal(lm):
    p0 = np.array([lm[0].x, lm[0].y, lm[0].z])
    p5 = np.array([lm[5].x, lm[5].y, lm[5].z])
    p17 = np.array([lm[17].x, lm[17].y, lm[17].z])

    v1 = p5 - p0
    v2 = p17 - p0
    normal = np.cross(v1, v2)

    norm = np.linalg.norm(normal)
    if norm > 0:
        normal /= norm

    return normal

def finger_is_extended_3d(lm, tip_idx, pip_idx, mcp_idx, palm_normal):
    tip = np.array([lm[tip_idx].x, lm[tip_idx].y, lm[tip_idx].z])
    pip = np.array([lm[pip_idx].x, lm[pip_idx].y, lm[pip_idx].z])
    mcp = np.array([lm[mcp_idx].x, lm[mcp_idx].y, lm[mcp_idx].z])

    tip_proj = np.dot(tip - mcp, palm_normal)
    pip_proj = np.dot(pip - mcp, palm_normal)

    return tip_proj > pip_proj

def thumb_is_extended_3d(lm, palm_normal, handedness: str):
    thumb_tip = np.array([lm[4].x, lm[4].y, lm[4].z])
    thumb_mcp = np.array([lm[2].x, lm[2].y, lm[2].z])
    wrist = np.array([lm[0].x, lm[0].y, lm[0].z])
    index_mcp = np.array([lm[5].x, lm[5].y, lm[5].z])

    thumb_vec = thumb_tip - thumb_mcp
    palm_vec = index_mcp - wrist

    dot = np.dot(thumb_vec, palm_vec)
    norms = np.linalg.norm(thumb_vec) * np.linalg.norm(palm_vec)
    if norms == 0:
        return False

    angle = math.degrees(math.acos(np.clip(dot / norms, -1.0, 1.0)))
    return angle > 40

def get_finger_states_3d(lm, handedness):
    palm_normal = get_palm_normal(lm)
    return {
        "thumb": thumb_is_extended_3d(lm, palm_normal, handedness),
        "index": finger_is_extended_3d(lm, 8, 6, 5, palm_normal),
        "middle": finger_is_extended_3d(lm, 12, 10, 9, palm_normal),
        "ring": finger_is_extended_3d(lm, 16, 14, 13, palm_normal),
        "pinky": finger_is_extended_3d(lm, 20, 18, 17, palm_normal),
    }

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

def classify_hand_shape(lm, handedness):
    f = get_finger_states_3d(lm, handedness)
    index_joints = [5, 6, 7, 8]

    # index
    if f["index"] and not f["middle"] and not f["ring"] and not f["pinky"]:
        if not f["thumb"]:
            return "index", False

    # rock
    if f["index"] and f["pinky"] and not f["middle"] and not f["ring"]:
        return "rock", False

    # L con invertida
    if f["index"] and not f["ring"] and not f["pinky"]:
        index_straight = finger_is_straight(lm, index_joints, tol_deg=45.0)
        thumb_ext = f["thumb"]

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

                raw_shape, raw_inverted = classify_hand_shape(lm, hand_label)

                if hand_label not in last_shape:
                    last_shape[hand_label] = "index"
                    last_inverted[hand_label] = False

                if raw_shape != "unknown":
                    last_shape[hand_label] = raw_shape
                    last_inverted[hand_label] = raw_inverted

                shape = last_shape[hand_label]
                inverted = last_inverted[hand_label]

                # dedos usados para el bounding
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
