import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
mesh = mp.solutions.face_mesh.FaceMesh()
while True:
    ret, frame = cap.read()
    if not ret:
        break
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = mesh.process(rgb)
    if res.multi_face_landmarks:
        for lm in res.multi_face_landmarks[0].landmark:
            h, w = frame.shape[:2]
            cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 1, (0,255,0), -1)
    cv2.imshow("Face", frame)
    if cv2.waitKey(1)&0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
