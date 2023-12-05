import cv2
import mediapipe as mp
import pyautogui
import numpy as np
from collections import deque
from approach import baseline_pred_xy
cam = cv2.VideoCapture(0)
while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    xy = baseline_pred_xy(rgb_frame)
    if xy is not None:
        pyautogui.moveTo(*xy)

    cv2.imshow('DL Project Milestone 1', frame)
    k = cv2.waitKey(1)
    if k == 27:
        break

cv2.destroyAllWindows()
