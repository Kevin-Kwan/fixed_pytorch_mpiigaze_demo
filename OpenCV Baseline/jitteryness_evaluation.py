import cv2
import mediapipe as mp
import pyautogui
import numpy as np
from collections import deque
from approach import baseline_pred_xy
import time

cam = cv2.VideoCapture(0)
pyautogui.FAILSAFE = False

screen_w, screen_h = pyautogui.size()

pos_history = []

while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    frame_h, frame_w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    xy = baseline_pred_xy(rgb_frame)
    if xy is not None:
        pos_history.append(xy)

        pyautogui.moveTo(*xy)

    cv2.imshow('DL Project Milestone 1', frame)
    k = cv2.waitKey(1)
    if k == 27:
        break

print("var h and v:",np.var(pos_history, axis=0))
cv2.destroyAllWindows()
