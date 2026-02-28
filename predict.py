import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize video capture
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.3
)

# Labels dictionary (MAKE SURE this matches your training labels)
labels_dict = {
    0: 'Yes', 1: 'No', 2: 'Thank you', 3: "Love", 4: "Help",
    5: "More", 6: "Name", 7: "Stop", 8: "Eat", 9: "My",
    10: "Friend", 11: "Show me", 12: "Hurt", 13: "Bathroom",
    14: "A", 15: "B", 16: "C", 17: "D", 18: "E", 19: "F",
    20: "G", 21: "H", 22: "I", 23: "J", 24: "K", 25: "L",
    26: "M", 27: "N", 28: "O", 29: "P", 30: "Q", 31: "R",
    32: "S", 33: "T", 34: "U", 35: "V", 36: "W", 37: "X",
    38: "Y", 39: "Z", 40: "For", 41: "Watch", 42: "Hello", 43: "."
}


while True:
    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        # Draw landmarks
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

        # Extract x and y coordinates
        x_ = [lm.x for lm in hand_landmarks.landmark]
        y_ = [lm.y for lm in hand_landmarks.landmark]

        # Normalize landmarks
        data_aux = []
        for lm in hand_landmarks.landmark:
            data_aux.append(lm.x - min(x_))
            data_aux.append(lm.y - min(y_))

        # Ensure fixed length (84 features)
        while len(data_aux) < 84:
            data_aux.append(0)
        if len(data_aux) > 84:
            data_aux = data_aux[:84]

        data_array = np.asarray(data_aux)

        # Make prediction
        prediction = model.predict([data_array])
        predicted_index = int(prediction[0])

        # Safe label lookup
        if predicted_index in labels_dict:
            predicted_character = labels_dict[predicted_index]
        else:
            predicted_character = f"Class {predicted_index}"

        # Get prediction confidence (if supported)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba([data_array])
            confidence = np.max(proba)
        else:
            confidence = 1.0

        text_to_show = f"{predicted_character} ({confidence*100:.1f}%)"

        # Draw bounding box
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) + 10
        y2 = int(max(y_) * H) + 10

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(
            frame,
            text_to_show,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            3,
            cv2.LINE_AA
        )

    cv2.imshow('Hand Tracking', frame)

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()