import cv2
import mediapipe as mp
import pyautogui
import numpy as np
from collections import deque
cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()
eye_q = deque(maxlen=10)
rec_points = []
while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape
    if landmark_points:
        landmarks = landmark_points[0].landmark
        eye_top = landmarks[386]
        eye_low = landmarks[374]
        eye_left = landmarks[362]
        eye_right = landmarks[263]
        eye_center = landmarks[473]

        x_pct = (eye_center.x - eye_left.x)/(eye_right.x - eye_left.x)
        y_pct = (eye_center.y - eye_top.y)/(eye_low.y - eye_top.y)
        eye_q.append((x_pct, y_pct))


    cv2.imshow('DL Project Milestone 1', frame)
    k = cv2.waitKey(1)
    if k == 27:
        break
    elif k == ord('e'):
        avg_p = np.mean(eye_q, axis=0)
        rec_points.append(avg_p)

cv2.destroyAllWindows()

print(rec_points)