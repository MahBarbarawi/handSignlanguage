
import mediapipe as mp
import cv2
import numpy as np
import tensorflow as tf
import pickle
class x:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_holistic = mp.solutions.holistic
        self.model = tf.keras.models.load_model("./sighnLanguge-2023-5-20.h5")
        with open("./Letters-2023-5-20.pkl", 'rb') as f:
            self.filesLetters = pickle.load(f)
    def extract_keypoints(self,img, holistic):
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        resultsleft = holistic.process(image)
        if resultsleft.left_hand_landmarks:
            left = np.array([])
            for res in resultsleft.left_hand_landmarks.landmark:
                left = np.append(left, np.array([res.x, res.y, res.z]))
            lh = left.flatten()
        elif resultsleft.right_hand_landmarks:
            img = cv2.flip(img, 1).copy()
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            left = np.array([])
            resultsleft = holistic.process(image)
            if resultsleft.left_hand_landmarks:
                for res in resultsleft.left_hand_landmarks.landmark:
                    left = np.append(left, np.array([res.x, res.y, res.z]))
                lh = left.flatten()
            else:
                lh = np.zeros(21 * 3)
        else:
            lh = np.zeros(21 * 3)
        return np.concatenate([lh])
    def PicProce(self,img):
        # print(img)
        print('Done Ai entered')

        cv2.imshow("img",img)
        with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            keypoints = self.extract_keypoints(img, holistic)
            res = self.model.predict(np.array([keypoints]))
            if .8 < np.argmax(res):
                return self.filesLetters[np.argmax(res)]
            return "NO Capture"

# def show(self):
    #     cap = cv2.VideoCapture(0)
    #     while True:
    #         if cv2.waitKey(10) & 0xff == ord('q'):
    #             break
    #         ret, frame = cap.read()
    #         frame =self.PicProce(frame)

