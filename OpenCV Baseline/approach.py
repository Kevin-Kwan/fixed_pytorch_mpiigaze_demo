import cv2
import mediapipe as mp
import pyautogui
import numpy as np
from collections import deque


    
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()
corners = np.array([[0.3194148 , 0.39005927],
                    [0.54344393, 0.35243119],
                    [0.31115591, 0.25699012],
                    [0.52684643, 0.24305441]])
pos_q = deque(maxlen=10)
avg_w, avg_h = np.mean(corners[[1,3],0]) - np.mean(corners[[0,2],0]), np.mean(corners[[0,1],1]) - np.mean(corners[[2,3],1])
def baseline_pred_xy(rgb_frame):
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    if landmark_points:
        landmarks = landmark_points[0].landmark
        eye_top = landmarks[386]
        eye_low = landmarks[374]
        eye_left = landmarks[362]
        eye_right = landmarks[263]
        eye_center = landmarks[473]

        x_pct = (eye_center.x - eye_left.x)/(eye_right.x - eye_left.x)
        y_pct = (eye_center.y - eye_top.y)/(eye_low.y - eye_top.y)

        dx_pct = (x_pct-corners[0,0]) / avg_w
        dy_pct = -(y_pct-corners[0,1]) / avg_h

        screen_x = int(dx_pct * screen_w)
        screen_y = int(dy_pct * screen_h)
        pos_q.append((screen_x, screen_y))

        avg_p = np.mean(pos_q, axis=0)
        return avg_p.astype(np.int32)
    return None

