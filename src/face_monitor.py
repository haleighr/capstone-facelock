import cv2
import mediapipe as mp
import numpy as np
import json
import os
import time

# =========================
# CONFIG
# =========================
TEMPLATES_FILE = "authorized_templates.json"
LOCK_DELAY_SECONDS = 5

EAR_THRESHOLD = 0.20
EAR_CONSEC_FRAMES = 3

POSE_GUIDE = [
    "Look straight",
    "Tilt left",
    "Tilt right",
    "Look up",
    "Look down",
    "Smile / Neutral"
]

# =========================
# SECURITY LEVELS
# =========================
def get_thresholds(level):
    if level == 1:
        return 0.75, False
    elif level == 2:
        return 0.85, True
    elif level == 3:
        return 0.92, True

# =========================
# LOAD / SAVE
# =========================
def load_templates():
    if not os.path.exists(TEMPLATES_FILE):
        return {}
    try:
        with open(TEMPLATES_FILE, "r") as f:
            return json.load(f)
    except:
        return {}

authorized_templates = load_templates()

# =========================
# MEDIAPIPE
# =========================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=5, refine_landmarks=True)

# =========================
# LANDMARKS
# =========================
KEY_POINTS = [33, 263, 1, 152, 61, 291]
LEFT_EYE_POINTS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_POINTS = [263, 387, 385, 362, 380, 373]

# =========================
# HELPERS
# =========================
def get_landmark_array(landmarks, w, h):
    return np.array([[landmarks.landmark[i].x * w,
                      landmarks.landmark[i].y * h] for i in KEY_POINTS], dtype=np.float32)

def align_face(points):
    left_eye, right_eye = points[0], points[1]
    dx, dy = right_eye - left_eye
    angle = np.arctan2(dy, dx)
    R = np.array([
        [np.cos(-angle), -np.sin(-angle)],
        [np.sin(-angle),  np.cos(-angle)]
    ])
    centered = points - np.mean(points, axis=0)
    return centered @ R.T

def normalize(points):
    d = np.linalg.norm(points[0] - points[1])
    return points / d if d != 0 else points

def vectorize(points):
    return points.flatten()

# =========================
# BLINK DETECTION (FIXED)
# =========================
blink_counter = 0
blink_detected = False
blink_timer = 0

def compute_ear(landmarks, indices, w, h):
    pts = np.array([[landmarks.landmark[i].x * w, landmarks.landmark[i].y * h] for i in indices])
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    h_dist = np.linalg.norm(pts[0] - pts[3])
    return (v1 + v2) / (2.0 * h_dist) if h_dist != 0 else 0

def update_blink(face, w, h):
    global blink_counter, blink_detected, blink_timer

    ear = (compute_ear(face, LEFT_EYE_POINTS, w, h) +
           compute_ear(face, RIGHT_EYE_POINTS, w, h)) / 2

    if ear < EAR_THRESHOLD:
        blink_counter += 1
    else:
        if blink_counter >= EAR_CONSEC_FRAMES:
            blink_detected = True
            blink_timer = time.time()
        blink_counter = 0

    if blink_detected and time.time() - blink_timer > 1:
        blink_detected = False

    return blink_detected

# =========================
# MATCHING (MULTI-USER)
# =========================
def match_face(vec):
    best_score = 0
    best_user = None

    for user in authorized_templates:
        for t in authorized_templates[user]:
            t_vec = np.array(t["vector"])
            d = np.linalg.norm(vec - t_vec)
            score = 1 / (1 + d)

            if score > best_score:
                best_score = score
                best_user = user

    return best_score, best_user

# =========================
# LOCK LOGIC
# =========================
locked = False
violation_start = None

def should_lock(score, faces, blink_ok, threshold, require_blink):
    global violation_start

    violation = (score < threshold) or (faces > 1) or (require_blink and not blink_ok)

    if violation:
        if violation_start is None:
            violation_start = time.time()
        return time.time() - violation_start > LOCK_DELAY_SECONDS
    else:
        violation_start = None
        return False

# =========================
# MAIN
# =========================
cap = cv2.VideoCapture(0)

# ===== SECURITY LEVEL SELECT =====
SECURITY_LEVEL = 2
selected = False

while not selected:
    ret, frame = cap.read()

    cv2.putText(frame, "Select Security Level", (50,100), 0, 0.7, (0,255,255),2)
    cv2.putText(frame, "1: Low", (50,140), 0, 0.6, (0,255,0),2)
    cv2.putText(frame, "2: Medium", (50,170), 0, 0.6, (0,255,0),2)
    cv2.putText(frame, "3: High", (50,200), 0, 0.6, (0,255,0),2)

    cv2.imshow("FaceLock", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('1'):
        SECURITY_LEVEL = 1
        selected = True
    elif key == ord('2'):
        SECURITY_LEVEL = 2
        selected = True
    elif key == ord('3'):
        SECURITY_LEVEL = 3
        selected = True

threshold, require_blink = get_thresholds(SECURITY_LEVEL)

# ===== USERNAME INPUT =====
username = ""
typing = True

while typing:
    ret, frame = cap.read()

    cv2.putText(frame, "Enter Username:", (50,100), 0, 0.7, (0,255,255),2)
    cv2.putText(frame, username, (50,150), 0, 0.7, (0,255,0),2)
    cv2.putText(frame, "Press ENTER to confirm", (50,200), 0, 0.5, (255,255,255),1)

    cv2.imshow("FaceLock", frame)
    key = cv2.waitKey(1)

    if key == 13 and username.strip() != "":
        typing = False
    elif key == 8:
        username = username[:-1]
    elif 32 <= key <= 126:
        username += chr(key)

# ===== MAIN LOOP =====
enrolling = False
pose_idx = 0
enroll_data = []

while True:
    ret, frame = cap.read()
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    score = 0
    matched_user = None
    faces = 0
    blink = False

    if results.multi_face_landmarks:
        faces = len(results.multi_face_landmarks)
        face = results.multi_face_landmarks[0]

        blink = update_blink(face, w, h)

        pts = normalize(align_face(get_landmark_array(face, w, h)))
        vec = vectorize(pts)

        score, matched_user = match_face(vec)

        if score >= threshold:
            status = f"AUTHORIZED: {matched_user}"
        else:
            status = "UNAUTHORIZED"

        cv2.putText(frame, f"Score: {score:.2f}", (10,30), 0, 0.6, (0,255,0),2)
        cv2.putText(frame, status, (10,60), 0, 0.7, (0,255,255),2)
        cv2.putText(frame, f"Blink: {blink}", (10,90), 0, 0.6, (255,255,0),2)

    # ===== ENROLLMENT UI =====
    if enrolling:
        if pose_idx < len(POSE_GUIDE):
            cv2.putText(frame, f"{POSE_GUIDE[pose_idx]} (Press E)", (50,140), 0, 0.6, (0,255,0),2)
        else:
            cv2.putText(frame, "Done! Press G to save", (50,140), 0, 0.6, (0,255,0),2)

    # ===== LOCK =====
    if should_lock(score, faces, blink, threshold, require_blink):
        cv2.putText(frame, "LOCKED", (200,200), 0, 1.5, (0,0,255),3)
        if not locked:
            os.system("gnome-screensaver-command -l")
            locked = True
    else:
        locked = False

    cv2.imshow("FaceLock", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord('g'):
        if not enrolling:
            enrolling = True
            pose_idx = 0
            enroll_data = []
        else:
            if username not in authorized_templates:
                authorized_templates[username] = []
            authorized_templates[username].extend(enroll_data)

            with open(TEMPLATES_FILE, "w") as f:
                json.dump(authorized_templates, f, indent=2)

            enrolling = False

    elif key == ord('e') and enrolling and pose_idx < len(POSE_GUIDE) and results.multi_face_landmarks:
        enroll_data.append({"vector": vec.tolist(), "points": pts.tolist()})
        pose_idx += 1

cap.release()
cv2.destroyAllWindows()