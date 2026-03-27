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
MATCH_THRESHOLD = 0.70
LOCK_DELAY_SECONDS = 5
SMOOTHING_ALPHA = 0.8

EAR_THRESHOLD = 0.20
EAR_CONSEC_FRAMES = 3

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
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=5,
    refine_landmarks=True
)

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
    pts = []
    for idx in KEY_POINTS:
        lm = landmarks.landmark[idx]
        pts.append([lm.x * w, lm.y * h])
    return np.array(pts, dtype=np.float32)

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
    eye_dist = np.linalg.norm(points[0] - points[1])
    if eye_dist == 0:
        return points
    return points / eye_dist

def vectorize(points):
    return points.flatten()

# =========================
# BLINK DETECTION
# =========================
blink_counter = 0
blink_detected = False

def compute_ear(landmarks, indices, w, h):
    pts = []
    for idx in indices:
        lm = landmarks.landmark[idx]
        pts.append(np.array([lm.x * w, lm.y * h]))

    p1, p2, p3, p4, p5, p6 = pts

    v1 = np.linalg.norm(p2 - p6)
    v2 = np.linalg.norm(p3 - p5)
    h_dist = np.linalg.norm(p1 - p4)

    if h_dist == 0:
        return 0

    return (v1 + v2) / (2.0 * h_dist)

# =========================
# SCORING
# =========================
def sigmoid(x, midpoint=0.35, sharpness=10):
    return 1.0 / (1.0 + np.exp(sharpness * (x - midpoint)))

def vector_score(v1, v2):
    d = np.linalg.norm(v1 - v2)
    return sigmoid(d)

def ratio_score(p1, p2):
    def ratio(p):
        nose, chin = p[2], p[3]
        le, re = p[0], p[1]

        v = np.linalg.norm(nose - chin)
        h = np.linalg.norm(le - re)
        return v / h if h != 0 else 0

    return 1 - abs(ratio(p1) - ratio(p2))

prev_score = None

def smooth(score):
    global prev_score
    if prev_score is None:
        prev_score = score
        return score

    s = SMOOTHING_ALPHA * prev_score + (1 - SMOOTHING_ALPHA) * score
    prev_score = s
    return s

# =========================
# MATCHING
# =========================
def match_face(vec, pts):
    best = 0

    for user in authorized_templates:
        for t in authorized_templates[user]:
            t_vec = np.array(t["vector"])
            t_pts = np.array(t["points"])

            v = vector_score(vec, t_vec)
            g = ratio_score(pts, t_pts)

            score = 0.7 * v + 0.3 * g
            best = max(best, score)

    return smooth(best)

# =========================
# LOCK LOGIC
# =========================
violation_start = None

def should_lock(score, faces, blink_ok):
    global violation_start

    if len(authorized_templates) == 0:
        return False

    violation = (score < MATCH_THRESHOLD) or (faces > 1) or (not blink_ok)

    if violation:
        if violation_start is None:
            violation_start = time.time()

        if time.time() - violation_start > LOCK_DELAY_SECONDS:
            return True
    else:
        violation_start = None

    return False

# =========================
# MAIN LOOP
# =========================
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    score = 0
    num_faces = 0

    if results.multi_face_landmarks:
        num_faces = len(results.multi_face_landmarks)
        face = results.multi_face_landmarks[0]

        # ===== BLINK DETECTION =====
        left = compute_ear(face, LEFT_EYE_POINTS, w, h)
        right = compute_ear(face, RIGHT_EYE_POINTS, w, h)
        ear = (left + right) / 2.0

        if ear < EAR_THRESHOLD:
            blink_counter += 1
        else:
            if blink_counter >= EAR_CONSEC_FRAMES:
                blink_detected = True
            blink_counter = 0

        # ===== FACE PROCESSING =====
        pts = get_landmark_array(face, w, h)
        aligned = align_face(pts)
        norm = normalize(aligned)
        vec = vectorize(norm)

        score = match_face(vec, norm)

        # ===== UI =====
        cv2.putText(frame, f"Score: {score:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.putText(frame, f"Blink: {blink_detected}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    if should_lock(score, num_faces, blink_detected):
        cv2.putText(frame, "LOCKED", (200, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()