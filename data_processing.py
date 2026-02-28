import os
import pickle

import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=True,
    min_detection_confidence=0.3
)

DATA_DIR = './data'

data, labels = [], []

for label in os.listdir(DATA_DIR):
    label_dir = os.path.join(DATA_DIR, label)
    if not os.path.isdir(label_dir):
        continue

    for img_file in os.listdir(label_dir):
        img_path = os.path.join(label_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]

                x_min, y_min = min(x_coords), min(y_coords)

                data_point = []
                for lm in hand_landmarks.landmark:
                    data_point.extend([lm.x - x_min, lm.y - y_min])

                data.append(data_point)
                labels.append(label)

with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)