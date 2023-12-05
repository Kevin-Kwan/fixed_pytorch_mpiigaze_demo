import cv2
import mediapipe as mp
import pyautogui
import numpy as np
from collections import deque
from approach import baseline_pred_xy
import time

cam = cv2.VideoCapture(0)
pyautogui.FAILSAFE = False

cv2.namedWindow("DL Project Milestone 1", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("DL Project Milestone 1", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

screen_w, screen_h = pyautogui.size()
times = []
r = 100
curr = np.array([screen_w/2, screen_h/2], dtype=np.float32)
screen_size = np.array([screen_w, screen_h])
i=-1
t = 10
start = None

while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    frame_h, frame_w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    xy = baseline_pred_xy(rgb_frame)
    if xy is not None:
        if np.sqrt(np.sum((xy - curr)**2)) <= r:
            now = time.time()
            if i >= 0:
                delta = now - start
                times.append(delta)
            start = now
            i += 1
            curr = np.array([np.random.randint(0,screen_w), np.random.randint(0,screen_h)], dtype=np.float32)
            print(i)
            

        pyautogui.moveTo(*xy)

    cv2.circle(frame, ((curr/screen_size)*np.array([frame_w,frame_h])).astype(np.int32).tolist(), r, (255,0,0))

    cv2.imshow('DL Project Milestone 1', frame)
    k = cv2.waitKey(1)
    if k == 27 or i >= t:
        break

print(times)
print(np.mean(times))
cv2.destroyAllWindows()
