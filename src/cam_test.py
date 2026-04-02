import cv2
import mediapipe as mp
import numpy as np
import json
import os
import time

# CONFIG VARIABLES
TEMPLATE_FILE = "authorized_templates.json"
LOCK_DELAY = 5

EAR_THRESHOLD = 0.20
EAR_FRAMES = 3
BLINK_TIMEOUT = 5

POSES = ["STRAIGHT", "LEFT", "RIGHT", "UP", "DOWN", "SMILE"]

# DEFAULT STATE
enrolling = False
pose_index = 0
current_user = None
current_level = None
templates = []

blink_counter = 0
blink_detected = False
last_blink_time = 0

lock_timer = None
just_enrolled = 0
show_new_user_cue = False

score = 0
user = None


def load_db():
    """
    Loads database templates from our json files
    """
    if not os.path.exists(TEMPLATE_FILE):
        return {}
    try:
        with open(TEMPLATE_FILE, "r") as f:
            return json.load(f)
    except:
        return {}

def save_db(db):
    """
    Saves templates to database
    """
    with open(TEMPLATE_FILE, "w") as f:
        json.dump(db, f, indent=4)

db = load_db()

# MEDIAPIPE STUFF
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(False, 5, True)

# MEDIAPIPE LANDMARKS
KEY = [33, 263, 1, 152, 61, 291]
LEFT_EYE = [33,160,158,133,153,144]
RIGHT_EYE = [263,387,385,362,380,373]


def extract(face, w, h):
    """
    Returns an array of face landmarks extracted from a recognized face
    """
    return np.array([[face.landmark[i].x*w, face.landmark[i].y*h] for i in KEY])

def align(p):
    """
    Rotates all points so they're horizontally aligned before normalization.
    """
    le, re = p[0], p[1]
    angle = np.arctan2(re[1]-le[1], re[0]-le[0])
    R = np.array([[np.cos(-angle), -np.sin(-angle)],
                  [np.sin(-angle),  np.cos(-angle)]])
    return (p - np.mean(p, axis=0)) @ R.T

def normalize(p):
    """
    Rotates to front face pose for better calculations
    """
    d = np.linalg.norm(p[0]-p[1])
    return p/d if d != 0 else p

def vectorize(p):
    """
    Converts facial features into a vector
    """
    return p.flatten()

def ear(face, idx, w, h):
    """
    Ear Aspect Ratio: Used for blink detection.
    """
    pts = [np.array([face.landmark[i].x*w, face.landmark[i].y*h]) for i in idx]
    p1,p2,p3,p4,p5,p6 = pts
    return (np.linalg.norm(p2-p6)+np.linalg.norm(p3-p5))/(2*np.linalg.norm(p1-p4))

# MATCH CALCULATIONS
def sigmoid(x):
    """
    Converts geometric distance into a smoother output
    """
    return 1/(1+np.exp(10*(x-0.35)))

def match(vec, pts):
    """
    Uses different voters to calculate the best user that matches the current face, and also calculates an accuracy score for the match.
    """
    best = 0
    best_user = None

    for u in db:
        for t in db[u]["templates"]:
            v = np.array(t["vector"])
            p = np.array(t["points"])

            d = np.linalg.norm(vec - v)
            v_score = sigmoid(d)

            ratio = lambda p: np.linalg.norm(p[2]-p[3]) / np.linalg.norm(p[0]-p[1])
            g_score = 1 - abs(ratio(pts) - ratio(p))

            s = 0.7*v_score + 0.3*g_score

            if s > best:
                best = s
                best_user = u

    return best, best_user

# SECURITY EVALUATION
def evaluate(score, faces, blink, user, override=False):
    """
    Defines security levels, as well as their requirements
    """
    if user is None:
        return False, True

    if override:
        return True, False

    level = db[user]["security"]

    if level == "LOW":
        return score >= 0.60, score < 0.60

    if level == "MEDIUM":
        return score >= 0.65 and faces == 1, score < 0.65 or faces > 1

    if level == "HIGH":
        return score >= 0.70 and faces == 1 and blink, \
               score < 0.70 or faces > 1 or not blink

    return False, True

def should_lock(violation):
    """
    Function to determine when the computer should lock if there is a violation.
    """
    global lock_timer

    if not violation:
        lock_timer = None
        return False

    if lock_timer is None:
        lock_timer = time.time()

    return time.time() - lock_timer > LOCK_DELAY

# MAIN LOOP FOR RECOGNIZTION
cap = cv2.VideoCapture(0)

while True:
    # Webcam capture stuff
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    faces = 0

    key = cv2.waitKey(1) & 0xFF

    if results.multi_face_landmarks:
        faces = len(results.multi_face_landmarks)
        face = results.multi_face_landmarks[0]

        pts = extract(face, w, h)
        norm = normalize(align(pts))
        vec = vectorize(norm)

        # BLINK
        e = (ear(face, LEFT_EYE, w, h) + ear(face, RIGHT_EYE, w, h)) / 2

        if e < EAR_THRESHOLD:
            blink_counter += 1
        else:
            if blink_counter >= EAR_FRAMES:
                blink_detected = True
                last_blink_time = time.time()
            blink_counter = 0

        if time.time() - last_blink_time > BLINK_TIMEOUT:
            blink_detected = False

        # Enrollment
        if enrolling:
            if key == ord('x'):
                templates.append({"points": norm.tolist(), "vector": vec.tolist()})
                pose_index += 1

                if pose_index >= len(POSES):
                    db[current_user] = {
                        "security": current_level,
                        "templates": templates.copy()
                    }
                    save_db(db)
                    templates.clear()
                    enrolling = False
                    pose_index = 0
                    lock_timer = None
                    just_enrolled = 30
                    show_new_user_cue = True

                    score, user = match(vec, norm)

        else:
            if len(db) > 0:
                score, user = match(vec, norm)

    else:
        # No face detected so reset
        score = 0
        user = None


    # SECURITY ONLY IF USERS EXIST
    if not enrolling and len(db) > 0:
        authorized, violation = evaluate(score, faces, blink_detected, user, override=(just_enrolled > 0))
        if just_enrolled > 0:
            just_enrolled -= 1

        if should_lock(violation):
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            cv2.putText(frame, "LOCKED", (150, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)


    # UI
    y = [30]
    spacing = 30

    def draw(text, color=(255, 255, 255)):
        cv2.putText(frame, text, (10, y[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y[0] += spacing

    draw(f"User: {user}")
    draw(f"Score: {score:.2f}", (0, 255, 0))
    draw(f"Blink: {blink_detected}", (255, 255, 0))
    draw(f"Faces: {faces}")

    if enrolling:
        draw(f"ENROLLING: {POSES[pose_index]}", (0, 255, 255))
        draw("Press X to capture")

    if show_new_user_cue:
        cv2.putText(frame, f"NEW USER RECOGNIZED: {current_user}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if just_enrolled <= 0:
            show_new_user_cue = False

    cv2.imshow("Face Recognition", frame)

# INPUT HANDLING
    if key == ord('c'):
        current_user = input("Enter username: ")
        lvl = input("Security Level (1=LOW,2=MEDIUM,3=HIGH): ")
        current_level = ["LOW", "MEDIUM", "HIGH"][int(lvl)-1]

        enrolling = True
        pose_index = 0
        templates.clear()

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()