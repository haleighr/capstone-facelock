import cv2
import mediapipe as mp
import numpy as np
import json
import os

# Initialize MediaPipe Face Mesh model
mp_face_mesh = mp.solutions.face_mesh

# FACE LANDMARK PROCESSING FUNCTIONS

def extract_landmark_points(landmarks, image_shape):
    """
    Convert MediaPipe landmarks into an array of (x, y) pixel coordinates.
    """
    image_height, image_width = image_shape[:2]
    points_2d = np.array([
        [lm.x * image_width, lm.y * image_height] for lm in landmarks
    ], dtype=np.float32)
    return points_2d


def normalize_landmark_points(points_2d):
    """
    Normalize landmark points so comparisons are scale- and position-invariant.
    Steps:
      - Center the face around its centroid
      - Scale by inter-ocular (eye-to-eye) distance
    """
    centroid = points_2d.mean(axis=0)
    centered_points = points_2d - centroid

    # Use approximate indices for left/right eye from FaceMesh model
    left_eye_point = points_2d[33]
    right_eye_point = points_2d[263]
    interocular_distance = np.linalg.norm(left_eye_point - right_eye_point)

    # Can't divide by zero
    if interocular_distance < 1e-3:
        interocular_distance = 1.0

    normalized_points = centered_points / interocular_distance
    return normalized_points.flatten()


def calculate_similarity_score(authorized_vector, live_vector):
    """
    Compute similarity between two normalized landmark vectors.
    Smaller distance = more similar.
    Convert Euclidean distance to a [0,1] similarity score.
    """
    euclidean_distance = np.linalg.norm(authorized_vector - live_vector)
    similarity = max(0.0, 1.0 - (euclidean_distance / 0.6))  # 0.6 is tunable
    return float(similarity)


def geometric_ratio_score(authorized_points, live_points):
    """
    Compare simple geometric ratios between two faces (very rough heuristic).
    Uses the ratio of nose-to-chin distance to eye distance.
    """
    try:
        # Landmark indices are based on MediaPipe Face Mesh
        NOSE_INDEX = 1
        CHIN_INDEX = 152
        LEFT_EYE_INDEX = 33
        RIGHT_EYE_INDEX = 263

        # Extract key feature points
        authorized_nose = authorized_points[NOSE_INDEX]
        authorized_chin = authorized_points[CHIN_INDEX]
        authorized_left_eye = authorized_points[LEFT_EYE_INDEX]
        authorized_right_eye = authorized_points[RIGHT_EYE_INDEX]

        live_nose = live_points[NOSE_INDEX]
        live_chin = live_points[CHIN_INDEX]
        live_left_eye = live_points[LEFT_EYE_INDEX]
        live_right_eye = live_points[RIGHT_EYE_INDEX]

        # Calculate the geometric ratios. We add a small epsilon to avoid division by zero.
        authorized_ratio = np.linalg.norm(authorized_nose - authorized_chin) / (
            np.linalg.norm(authorized_left_eye - authorized_right_eye) + 1e-6
        )
        live_ratio = np.linalg.norm(live_nose - live_chin) / (
            np.linalg.norm(live_left_eye - live_right_eye) + 1e-6
        )

        difference = abs(authorized_ratio - live_ratio)
        score = max(0.0, 1.0 - (difference / 0.25))  # 0.25 threshold is tunable
        return float(score)

    except Exception:
        return 0.0


# TEMPLATE STORAGE HANDLING

TEMPLATES_FILE = "authorized_templates.json"

def load_authorized_templates():
    """Load authorized face templates from a JSON file."""
    if os.path.exists(TEMPLATES_FILE):
        with open(TEMPLATES_FILE, "r") as f:
            return json.load(f)
    return {}

def save_authorized_templates(template_dict):
    """Save authorized face templates to disk."""
    with open(TEMPLATES_FILE, "w") as f:
        json.dump(template_dict, f)


authorized_templates = load_authorized_templates()


# WEBCAM LOOP

camera = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh_detector:

    print("Press 'a' to add current detected face to authorized list.")
    print("Press 'q' to quit the program.")

    while True:
        success, frame = camera.read()
        if not success:
            break

        # Convert to RGB for MediaPipe processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detection_results = face_mesh_detector.process(frame_rgb)
        frame_height, frame_width = frame.shape[:2]

        if detection_results.multi_face_landmarks:
            face_landmarks = detection_results.multi_face_landmarks[0].landmark

            # Convert and normalize current frame landmarks
            live_points_2d = extract_landmark_points(face_landmarks, frame.shape)
            live_vector = normalize_landmark_points(live_points_2d)

            # Compare against stored authorized templates
            best_match_name = None
            highest_vote_score = 0.0

            for person_name, template_data in authorized_templates.items():
                stored_points = np.array(template_data["raw_points"], dtype=np.float32).reshape(-1, 2)
                stored_vector = np.array(template_data["normalized_vector"], dtype=np.float32)

                # --- Council Voter 1: Landmark distance ---
                landmark_similarity = calculate_similarity_score(stored_vector, live_vector)

                # --- Council Voter 2: Simple geometric ratio ---
                geometric_similarity = geometric_ratio_score(stored_points, live_points_2d)

                # Combine the two votes (weights are tunable)
                combined_vote = 0.7 * landmark_similarity + 0.3 * geometric_similarity

                if combined_vote > highest_vote_score:
                    highest_vote_score = combined_vote
                    best_match_name = person_name

            # Decide recognition outcome
            if highest_vote_score >= 0.6:
                display_label = f"{best_match_name} ({highest_vote_score:.2f})"
                label_color = (0, 255, 0)
            else:
                display_label = f"Unknown ({highest_vote_score:.2f})"
                label_color = (0, 0, 255)

            # Draw landmarks and display result
            for (x, y) in live_points_2d.astype(int):
                cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)

            cv2.putText(frame, display_label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, label_color, 2)

        cv2.imshow("Face Monitor", frame)
        key = cv2.waitKey(1) & 0xFF

        # Add new authorized face
        if key == ord('a') and detection_results.multi_face_landmarks:
            new_name = input("Enter name for new authorized template: ").strip()
            authorized_templates[new_name] = {
                "raw_points": live_points_2d.flatten().tolist(),
                "normalized_vector": live_vector.tolist()
            }
            save_authorized_templates(authorized_templates)
            print(f"[INFO] Saved authorized template for '{new_name}'")

        # Exit
        if key == ord('q'):
            break

camera.release()
cv2.destroyAllWindows()
