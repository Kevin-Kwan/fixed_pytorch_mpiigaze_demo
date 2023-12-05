import cv2
import mediapipe as mp
import pyautogui
import numpy as np
cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()
datap = [[] for i in range(4)]
corners = np.array([
    [0.30475251, 0.45618043],
    [0.53288567, 0.39947405],
    [0.46606915, 0.15025976],
    [0.34280464, 0.26507329]
])
avg_w, avg_h = np.mean(corners[[1,2],0]) - np.mean(corners[[0,3],0]), np.mean(corners[[0,1],1]) - np.mean(corners[[2,3],1])
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

        dx_pct = (x_pct-corners[0,0]) / avg_w
        dy_pct = -(y_pct-corners[0,1]) / avg_h

        screen_x = int(dx_pct * screen_w)
        screen_y = int(dy_pct * screen_h)
        pyautogui.moveTo(screen_x, screen_y)

        k = cv2.waitKey(1)

    cv2.imshow('DL Project Milestone 1', frame)
    k = cv2.waitKey(1)
    if k == 27:
        break

cv2.destroyAllWindows()
