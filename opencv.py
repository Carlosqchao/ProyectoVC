import cv2
import mediapipe as mp
import socket
import json
import numpy as np
import math


# ------------ CONFIG ------------
GODOT_IP = "127.0.0.1"
GODOT_PORT = 4242
SEND_EVERY_N_FRAMES = 1

MAX_MISSING_FRAMES = 5
ALPHA_SMOOTH = 0.5
# -------------------------------


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("No se pudo abrir la webcam")
    exit(1)

frame_count = 0

FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_PIPS = [3, 6, 10, 14, 18]
FINGER_MCPS = [2, 5, 9, 13, 17]


def normalize_angle(angle):
    """Normaliza un ángulo al rango [-180, 180)"""
    while angle >= 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle


def smooth_angle_circular(new_angle, old_angle, alpha):
    """Suaviza ángulos considerando la discontinuidad circular"""
    new_angle = normalize_angle(new_angle)
    old_angle = normalize_angle(old_angle)
    
    # Calcula la diferencia más corta
    diff = new_angle - old_angle
    if diff > 180:
        diff -= 360
    elif diff < -180:
        diff += 360
    
    # Suaviza la diferencia
    return normalize_angle(old_angle + alpha * diff)


def get_palm_normal(hand_landmarks):
    """Calcula el vector normal de la palma usando landmarks 0, 5, 17"""
    lm = hand_landmarks.landmark
    
    # Puntos de la palma: muñeca (0), base del índice (5), base del meñique (17)
    p0 = np.array([lm[0].x, lm[0].y, lm[0].z])
    p5 = np.array([lm[5].x, lm[5].y, lm[5].z])
    p17 = np.array([lm[17].x, lm[17].y, lm[17].z])
    
    # Calcula el vector normal mediante producto cruz
    v1 = p5 - p0
    v2 = p17 - p0
    normal = np.cross(v1, v2)
    
    # Normaliza
    norm = np.linalg.norm(normal)
    if norm > 0:
        normal = normal / norm
    
    return normal


def finger_is_extended_3d(lm, tip_idx, pip_idx, mcp_idx, palm_normal):
    """
    Detecta si un dedo está extendido usando el vector normal de la palma.
    Compara la distancia del tip vs pip desde la palma.
    """
    tip = np.array([lm[tip_idx].x, lm[tip_idx].y, lm[tip_idx].z])
    pip = np.array([lm[pip_idx].x, lm[pip_idx].y, lm[pip_idx].z])
    mcp = np.array([lm[mcp_idx].x, lm[mcp_idx].y, lm[mcp_idx].z])
    
    # Proyección en la dirección normal de la palma
    tip_projection = np.dot(tip - mcp, palm_normal)
    pip_projection = np.dot(pip - mcp, palm_normal)
    
    # Si el tip está más alejado de la palma que el pip, el dedo está extendido
    return tip_projection > pip_projection


def thumb_is_extended_3d(lm, palm_normal, handedness):
    """
    Detecta si el pulgar está extendido considerando la orientación de la mano.
    """
    thumb_tip = np.array([lm[4].x, lm[4].y, lm[4].z])
    thumb_mcp = np.array([lm[2].x, lm[2].y, lm[2].z])
    wrist = np.array([lm[0].x, lm[0].y, lm[0].z])
    index_mcp = np.array([lm[5].x, lm[5].y, lm[5].z])
    
    # Vector del pulgar
    thumb_vec = thumb_tip - thumb_mcp
    
    # Vector de referencia de la palma (de muñeca a índice)
    palm_vec = index_mcp - wrist
    
    # Ángulo entre el pulgar y la palma
    dot = np.dot(thumb_vec, palm_vec)
    norms = np.linalg.norm(thumb_vec) * np.linalg.norm(palm_vec)
    
    if norms == 0:
        return False
    
    angle = math.degrees(math.acos(np.clip(dot / norms, -1.0, 1.0)))
    
    # El pulgar está extendido si forma un ángulo significativo con la palma
    return angle > 40


def get_finger_states_3d(hand_landmarks, handedness):
    """Detecta estados de dedos usando vectores 3D y normal de palma"""
    lm = hand_landmarks.landmark
    palm_normal = get_palm_normal(hand_landmarks)
    
    finger_states = {
        "thumb": thumb_is_extended_3d(lm, palm_normal, handedness),
        "index": finger_is_extended_3d(lm, 8, 6, 5, palm_normal),
        "middle": finger_is_extended_3d(lm, 12, 10, 9, palm_normal),
        "ring": finger_is_extended_3d(lm, 16, 14, 13, palm_normal),
        "pinky": finger_is_extended_3d(lm, 20, 18, 17, palm_normal),
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


def finger_is_straight(lm, joints, tol_deg=30.0):
    a = lm[joints[0]]
    b = lm[joints[1]]
    c = lm[joints[2]]
    ang = angle_3pts(a, b, c)
    return ang > (180.0 - tol_deg)


def classify_hand_shape(hand_landmarks, handedness):
    lm = hand_landmarks.landmark
    f = get_finger_states_3d(hand_landmarks, handedness)
    
    index_joints = [5, 6, 7, 8]
    
    # index
    if f["index"] and not f["middle"] and not f["ring"] and not f["pinky"]:
        if not f["thumb"]:
            return "index", False
    
    # rock
    if f["index"] and f["pinky"] and not f["middle"] and not f["ring"]:
        return "rock", False
    
    # L con detección de invertida
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


# Estado por mano usando handedness como clave persistente
last_hands_data = {}
missing_frames_per_hand = {}
smooth_x = {}
smooth_y = {}
smooth_angle = {}
last_shape = {}
last_inverted = {}


with mp_hands.Hands(
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.7
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
        
        detected_hands = set()
        hands_data = []
        
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                hand_label = handedness.classification[0].label  # "Left" o "Right"
                detected_hands.add(hand_label)
                
                lm = hand_landmarks.landmark
                
                # Gesto detectado en este frame
                raw_shape, raw_inverted = classify_hand_shape(hand_landmarks, hand_label)
                
                # Inicializa último gesto válido si no existe
                if hand_label not in last_shape:
                    last_shape[hand_label] = "index"
                    last_inverted[hand_label] = False
                
                # Si el crudo no es unknown, actualizamos el último válido
                if raw_shape != "unknown":
                    last_shape[hand_label] = raw_shape
                    last_inverted[hand_label] = raw_inverted
                
                # Lo que usamos siempre es el último válido
                shape = last_shape[hand_label]
                inverted = last_inverted[hand_label]
                
                # Selección de dedos para el bounding
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
                
                # Suavizado por mano usando hand_label como identificador persistente
                if hand_label not in smooth_x:
                    smooth_x[hand_label] = center_x
                    smooth_y[hand_label] = center_y
                    smooth_angle[hand_label] = angle_deg
                else:
                    smooth_x[hand_label] = int(
                        ALPHA_SMOOTH * center_x + (1 - ALPHA_SMOOTH) * smooth_x[hand_label]
                    )
                    smooth_y[hand_label] = int(
                        ALPHA_SMOOTH * center_y + (1 - ALPHA_SMOOTH) * smooth_y[hand_label]
                    )
                    # Suavizado circular de ángulos
                    smooth_angle[hand_label] = smooth_angle_circular(
                        angle_deg, smooth_angle[hand_label], ALPHA_SMOOTH
                    )
                
                # Resetea contador de frames perdidos
                missing_frames_per_hand[hand_label] = 0
                
                hand_dict = {
                    "x": smooth_x[hand_label],
                    "y": smooth_y[hand_label],
                    "len_x": length_x,
                    "len_y": length_y,
                    "label": hand_label,
                    "shape": shape,
                    "angle": smooth_angle[hand_label],
                    "inverted": inverted
                }
                
                hands_data.append(hand_dict)
                last_hands_data[hand_label] = hand_dict
                
                shape_text = shape + (" INV" if inverted else "")
                cv2.putText(
                    frame, f"{hand_label}: {shape_text}", (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
                )
        
        # Gestiona manos que no fueron detectadas en este frame
        for hand_label in list(last_hands_data.keys()):
            if hand_label not in detected_hands:
                if hand_label not in missing_frames_per_hand:
                    missing_frames_per_hand[hand_label] = 0
                
                missing_frames_per_hand[hand_label] += 1
                
                if missing_frames_per_hand[hand_label] <= MAX_MISSING_FRAMES:
                    # Mantiene última data conocida
                    hands_data.append(last_hands_data[hand_label])
                else:
                    # Limpia estado de esta mano
                    del last_hands_data[hand_label]
                    del missing_frames_per_hand[hand_label]
                    if hand_label in smooth_x:
                        del smooth_x[hand_label]
                        del smooth_y[hand_label]
                        del smooth_angle[hand_label]
                        del last_shape[hand_label]
                        del last_inverted[hand_label]
        
        # Envío a Godot
        if hands_data and frame_count % SEND_EVERY_N_FRAMES == 0:
            data = {
                "hands": hands_data,
                "w": w,
                "h": h
            }
            msg = json.dumps(data).encode("utf-8")
            sock.sendto(msg, (GODOT_IP, GODOT_PORT))
        
        frame_count += 1
        
        cv2.imshow("MediaPipe Gestos - Envío a Godot", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
sock.close()