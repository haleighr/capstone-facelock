import cv2
import mediapipe as mp
import numpy as np
import json
import os
from collections import deque

# Path to original file (for reference): /mnt/data/72728265-a9e1-4e1f-944c-5ebe25d3cfab.py

mp_face_mesh = mp.solutions.face_mesh

# Configurable parameters
SMOOTH_BUFFER_SIZE = 5          # frames to average for temporal smoothing
MATCH_THRESHOLD = 0.6           # combined score threshold to accept a match
STABLE_INDICES = [1, 6, 9, 10, 33, 133, 168, 197, 234, 263, 291, 297, 332, 338, 352, 454, 152]
# These indices favor nose, eyes, cheeks, jaw/chin and are relatively expression-stable.

TEMPLATES_FILE = "authorized_enrolled_templates.json"

# ------------------ Utility: I/O for templates ------------------

def load_authorized_templates():
    if os.path.exists(TEMPLATES_FILE):
        with open(TEMPLATES_FILE, "r") as f:
            return json.load(f)
    return {}


def save_authorized_templates(template_dict):
    with open(TEMPLATES_FILE, "w") as f:
        json.dump(template_dict, f, indent=4)


authorized_templates = load_authorized_templates()

# ------------------ Landmark processing ------------------

def extract_landmark_points(landmarks, image_shape):
    """Convert MediaPipe landmarks into an array of (x, y) pixel coordinates."""
    image_height, image_width = image_shape[:2]
    points_2d = np.array([[lm.x * image_width, lm.y * image_height] for lm in landmarks], dtype=np.float32)
    return points_2d


def align_face(points_2d):
    """Rotate landmarks so the eye line is horizontal. Returns rotated points (same shape)."""
    left_eye = points_2d[33]
    right_eye = points_2d[263]

    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))

    centroid = tuple(points_2d.mean(axis=0))
    rot_mat = cv2.getRotationMatrix2D(centroid, angle, 1.0)
    aligned = cv2.transform(points_2d.reshape(-1, 1, 2), rot_mat).reshape(-1, 2)
    return aligned


def normalize_landmark_points(points_2d, indices=None):
    """Center and scale landmarks. Use inter-ocular distance for scale. Optionally pick a subset via indices."""
    if indices is not None:
        pts = points_2d[indices]
    else:
        pts = points_2d

    centroid = pts.mean(axis=0)
    centered = pts - centroid

    left_eye = points_2d[33]
    right_eye = points_2d[263]
    iod = np.linalg.norm(left_eye - right_eye)
    if iod < 1e-3:
        iod = 1.0

    normalized = centered / iod
    return normalized.flatten()


def calculate_similarity_score(authorized_vector, current_feature_vector):
    euclidean_distance = np.linalg.norm(authorized_vector - current_feature_vector)
    # scale factor tuned for reduced-dimension vectors; adjust if you change indices/PCA
    similarity = max(0.0, 1.0 - (euclidean_distance / 0.6))
    return float(similarity)


def geometric_ratio_score(authorized_points, live_points):
    try:
        NOSE_INDEX = 1
        CHIN_INDEX = 152
        LEFT_EYE_INDEX = 33
        RIGHT_EYE_INDEX = 263

        authorized_nose_point = authorized_points[NOSE_INDEX]
        authorized_chin_point = authorized_points[CHIN_INDEX]
        authorized_left_eye_point = authorized_points[LEFT_EYE_INDEX]
        authorized_right_eye_point = authorized_points[RIGHT_EYE_INDEX]

        live_nose_point = live_points[NOSE_INDEX]
        live_chin_point = live_points[CHIN_INDEX]
        live_left_eye_point = live_points[LEFT_EYE_INDEX]
        live_right_eye_point = live_points[RIGHT_EYE_INDEX]

        a_ratio = np.linalg.norm(authorized_nose_point - authorized_chin_point) / (np.linalg.norm(authorized_left_eye_point - authorized_right_eye_point) + 1e-6)
        l_ratio = np.linalg.norm(live_nose_point - live_chin_point) / (np.linalg.norm(live_left_eye_point - live_right_eye_point) + 1e-6)

        diff = abs(a_ratio - l_ratio)
        score = max(0.0, 1.0 - (diff / 0.25))
        return float(score)
    except Exception:
        return 0.0


# ------------------ Run-time buffers & helpers ------------------

smooth_buffer = deque(maxlen=SMOOTH_BUFFER_SIZE)


def buffered_average(points_2d):
    """Append to buffer and return the averaged points for stability."""
    smooth_buffer.append(points_2d)
    if len(smooth_buffer) == 0:
        return points_2d
    return np.mean(np.stack(list(smooth_buffer), axis=0), axis=0)


# ------------------ Main webcam loop ------------------

camera = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh_detector:

    print("Press 'a' to add current detected face to authorized list (can add multiple orientations).")
    print("Press 'q' to quit the program.")

    while True:
        success, frame = camera.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detection_results = face_mesh_detector.process(frame_rgb)
        frame_height, frame_width = frame.shape[:2]

        if detection_results.multi_face_landmarks:
            face_landmarks = detection_results.multi_face_landmarks[0].landmark
            current_landmark_points = extract_landmark_points(face_landmarks, frame.shape)

            # 1) Align the face so eyes are horizontal
            aligned_landmark_points = align_face(current_landmark_points)

            # 2) Smooth (temporal average)
            smoothed_landmark_points = buffered_average(aligned_landmark_points)

            # 3) Normalize and reduce to stable indices
            current_feature_vector = normalize_landmark_points(smoothed_landmark_points, indices=STABLE_INDICES)

            # Compare against stored authorized templates (each person may have multiple templates)
            best_match_name = None
            highest_vote_score = 0.0

            for person_name, template_data in authorized_templates.items():
                templates = template_data.get("templates", [])
                for tpl in templates:
                    template_feature_vector = np.array(tpl["normalized_vector"], dtype=np.float32)
                    stored_raw = np.array(tpl.get("raw_points", []), dtype=np.float32).reshape(-1, 2) if tpl.get("raw_points") else None

                    landmark_similarity = calculate_similarity_score(template_feature_vector, current_feature_vector)
                    geometric_similarity = geometric_ratio_score(stored_raw if stored_raw is not None else aligned_landmark_points, smoothed_landmark_points)

                    combined_vote = 0.75 * landmark_similarity + 0.25 * geometric_similarity

                    if combined_vote > highest_vote_score:
                        highest_vote_score = combined_vote
                        best_match_name = person_name

            if highest_vote_score >= MATCH_THRESHOLD:
                display_label = f"{best_match_name} ({highest_vote_score:.2f})"
                label_color = (0, 255, 0)
            else:
                display_label = f"Unknown ({highest_vote_score:.2f})"
                label_color = (0, 0, 255)

            # Draw the smoothed_landmark_points landmarks (only stable subset to reduce clutter)
            for idx in STABLE_INDICES:
                x, y = int(smoothed_landmark_points[idx][0]), int(smoothed_landmark_points[idx][1])
                cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)

            cv2.putText(frame, display_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, label_color, 2)

        cv2.imshow("Face Monitor", frame)
        key = cv2.waitKey(1) & 0xFF

        # Add new authorized face (append as another template for this name)
        if key == ord('a') and detection_results.multi_face_landmarks:
            new_name = input("Enter name for new authorized template: ").strip()
            person = authorized_templates.get(new_name, {"templates": []})

            # Save current aligned & normalized template
            tpl = {
                "raw_points": aligned_landmark_points.flatten().tolist(),
                "normalized_vector": current_feature_vector.tolist()
            }
            person["templates"].append(tpl)
            authorized_templates[new_name] = person
            save_authorized_templates(authorized_templates)
            print(f"[INFO] Saved authorized template for '{new_name}' (now {len(person['templates'])} templates)")

        if key == ord('q'):
            break

camera.release()
cv2.destroyAllWindows()
