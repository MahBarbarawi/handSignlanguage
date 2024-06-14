import mediapipe as mp
import cv2
from AI import x
import mediapipe as mp
import cv2
import numpy as np
import uuid
import os
import requests

import time

import tensorflow as tf
import pickle
AiProcess = x()
cap = cv2.VideoCapture(0)
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
model = tf.keras.models.load_model("./sighnLanguge-2023-6-2.h5")
with open("./Letters-2023-5-20.pkl", 'rb') as f:
    filesLetters = pickle.load(f)
def extract_keypoints(img , holistic):
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    resultsleft = holistic.process(image)
    if resultsleft.left_hand_landmarks:
        left=np.array([])
        for res in resultsleft.left_hand_landmarks.landmark:
            left = np.append(left, np.array([res.x, res.y, res.z]))
        lh = left.flatten ()
    elif resultsleft.right_hand_landmarks:
        image= cv2.flip(img,1).copy()
        left=np.array([])
        resultsleft = holistic.process(image)
        if resultsleft.left_hand_landmarks:
            for res in resultsleft.left_hand_landmarks.landmark:
                   left = np.append(left, np.array([res.x, res.y, res.z]))
            lh = left.flatten ()
        else:
            lh=np.zeros(21*3)
    else:
            lh=np.zeros(21*3)
    return np.concatenate([lh]),resultsleft
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        # Read feed
        ret, frame = cap.read()
        print(AiProcess.PicProce((frame)))
        cv2.putText(frame, AiProcess.PicProce(frame), (20, 20),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('OpenCV Feed', frame)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
