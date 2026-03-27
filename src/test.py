import cv2
import mediapipe as mp
import numpy as np
import json
import os
from collections import deque

mp_face_mesh = mp.solutions.face_mesh

# ------------------ Configurable Parameters ------------------

SMOOTH_BUFFER_SIZE = 5
MATCH_THRESHOLD = 0.6

# Temporal stabilization
CONFIRM_THRESHOLD = 3.0
DECAY_FACTOR = 0.90
identity_confidence = {}

# Logging
LOG_RESULTS = False  # Set True if you want to log scores to CSV

STABLE_INDICES = [1, 6, 9, 10, 33, 133, 168, 197, 234, 263,
                  291, 297, 332, 338, 352, 454, 152]

TEMPLATES_FILE = "authorized_enrolled_templates.json"

# ------------------ Template I/O ------------------

def load_authorized_templates():
    try:
        if os.path.exists(TEMPLATES_FILE):
            with open(TEMPLATES_FILE, "r") as f:
                return json.load(f)
    except json.JSONDecodeError:
        print("[ERROR] Template file corrupted. Starting fresh.")
    return {}

def save_authorized_templates(template_dict):
    with open(TEMPLATES_FILE, "w") as f:
        json.dump(template_dict, f, indent=4)

authorized_templates = load_authorized_templates()

# ------------------ Landmark Processing ------------------

def extract_landmark_points(landmarks, image_shape):
    image_height, image_width = image_shape[:2]
    return np.array(
        [[lm.x * image_width, lm.y * image_height] for lm in landmarks],
        dtype=np.float32
    )

def align_face(points_2d):
    if points_2d.shape[0] <= 263:
        return None

    left_eye = points_2d[33]
    right_eye = points_2d[263]

    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))

    centroid = tuple(points_2d.mean(axis=0))
    rot_mat = cv2.getRotationMatrix2D(centroid, angle, 1.0)

    aligned = cv2.transform(points_2d.reshape(-1, 1, 2), rot_mat)
    return aligned.reshape(-1, 2)

def normalize_landmark_points(points_2d, indices=None):
    if points_2d is None:
        return None

    pts = points_2d[indices] if indices else points_2d

    centroid = pts.mean(axis=0)
    centered = pts - centroid

    left_eye = points_2d[33]
    right_eye = points_2d[263]
    iod = np.linalg.norm(left_eye - right_eye)

    if not np.isfinite(iod) or iod < 1e-3:
        return None

    normalized = centered / iod
    flat = normalized.flatten()

    if not np.all(np.isfinite(flat)):
        return None

    return flat

def calculate_similarity_score(authorized_vector, current_vector):
    distance = np.linalg.norm(authorized_vector - current_vector)
    similarity = max(0.0, 1.0 - (distance / 0.6))
    return float(similarity)

def geometric_ratio_score(authorized_points, live_points):
    try:
        NOSE = 1
        CHIN = 152
        LEFT_EYE = 33
        RIGHT_EYE = 263

        a_ratio = np.linalg.norm(
            authorized_points[NOSE] - authorized_points[CHIN]
        ) / (np.linalg.norm(
            authorized_points[LEFT_EYE] - authorized_points[RIGHT_EYE]
        ) + 1e-6)

        l_ratio = np.linalg.norm(
            live_points[NOSE] - live_points[CHIN]
        ) / (np.linalg.norm(
            live_points[LEFT_EYE] - live_points[RIGHT_EYE]
        ) + 1e-6)

        diff = abs(a_ratio - l_ratio)
        return max(0.0, 1.0 - (diff / 0.25))

    except (IndexError, ValueError, TypeError) as e:
        print(f"[WARN] Geometry score skipped: {e}")
        return 0.0

# ------------------ Temporal Smoothing ------------------

smooth_buffer = deque(maxlen=SMOOTH_BUFFER_SIZE)

def buffered_average(points):
    smooth_buffer.append(points)
    return np.mean(np.stack(smooth_buffer), axis=0)

# ------------------ Webcam Setup ------------------

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Failed to open webcam.")

# ------------------ Main Loop ------------------

with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh_detector:

    print("Press 'a' to enroll.")
    print("Press 'q' to quit.")

    while True:
        success, frame = camera.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh_detector.process(frame_rgb)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0].landmark

            if len(face_landmarks) < 468:
                continue

            raw_points = extract_landmark_points(face_landmarks, frame.shape)
            aligned_points = align_face(raw_points)
            smoothed_points = buffered_average(aligned_points)

            if smoothed_points is None:
                continue

            current_vector = normalize_landmark_points(
                smoothed_points, STABLE_INDICES
            )

            if current_vector is None:
                continue

            best_match_name = None
            highest_vote_score = 0.0

            for person_name, template_data in authorized_templates.items():
                for tpl in template_data.get("templates", []):

                    template_vector = np.array(
                        tpl["normalized_vector"], dtype=np.float32
                    )

                    if template_vector.shape != current_vector.shape:
                        continue

                    stored_raw = np.array(
                        tpl.get("raw_points", []), dtype=np.float32
                    ).reshape(-1, 2)

                    landmark_sim = calculate_similarity_score(
                        template_vector, current_vector
                    )

                    geo_sim = geometric_ratio_score(
                        stored_raw, smoothed_points
                    )

                    combined = 0.75 * landmark_sim + 0.25 * geo_sim

                    if combined > highest_vote_score:
                        highest_vote_score = combined
                        best_match_name = person_name

            # -------- Temporal Confidence --------

            for name in authorized_templates.keys():
                identity_confidence.setdefault(name, 0.0)

            for name in identity_confidence:
                identity_confidence[name] *= DECAY_FACTOR

            if best_match_name and highest_vote_score >= MATCH_THRESHOLD:
                identity_confidence[best_match_name] += highest_vote_score

            confirmed_name = None
            if identity_confidence:
                top_name = max(identity_confidence,
                               key=identity_confidence.get)
                if identity_confidence[top_name] >= CONFIRM_THRESHOLD:
                    confirmed_name = top_name

            if confirmed_name:
                display_label = f"{confirmed_name} (CONFIRMED)"
                label_color = (0, 255, 0)
            else:
                display_label = f"Unknown ({highest_vote_score:.2f})"
                label_color = (0, 0, 255)

            # Optional logging
            if LOG_RESULTS:
                with open("recognition_log.csv", "a") as f:
                    f.write(f"{best_match_name},{highest_vote_score}\n")

            cv2.putText(frame, display_label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, label_color, 2)

        cv2.imshow("Face Monitor", frame)
        key = cv2.waitKey(1) & 0xFF

        # Enrollment
        if key == ord('a') and results.multi_face_landmarks:
            new_name = input("Enter name: ").strip()
            person = authorized_templates.get(new_name,
                                              {"templates": []})

            tpl = {
                "raw_points": aligned_points.flatten().tolist(),
                "normalized_vector": current_vector.tolist()
            }

            person["templates"].append(tpl)
            authorized_templates[new_name] = person
            save_authorized_templates(authorized_templates)

            print(f"[INFO] Saved template for '{new_name}'")

        if key == ord('q'):
            break

camera.release()
cv2.destroyAllWindows()