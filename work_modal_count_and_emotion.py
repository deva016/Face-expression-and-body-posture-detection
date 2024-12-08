import keras
from time import sleep
import cv2
import datetime
import mediapipe as mp
import numpy as np
import os
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
classifier = keras.models.load_model('expressiondata1.h5')
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# assign
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
count = 0
total = 0

# Updated class labels with score
class_labels = ['angry: 20', 'happy: 10', 'sad: -5', 'neutral: 70', 'surprised: 50', 'disgusted: 5', 'fearful: 40']


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs((radians * 180.0 / np.pi))

    if angle > 180.0:
        angle = 360 - angle
    return angle


def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


if count == 0:
    cap = cv2.VideoCapture(0)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            frame = rescale_frame(frame, percent=160)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            original_frame = frame.copy()
            result = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            if result.pose_landmarks is not None:
                landmarks = result.pose_landmarks.landmark

                sholder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                angle = calculate_angle(sholder, elbow, wrist)

                s = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                e = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                w = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                a = calculate_angle(s, e, w)

                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                ang = calculate_angle(left_elbow, left_shoulder, left_hip)

                cv2.putText(image, str(angle), tuple(np.multiply(elbow, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, str(a), tuple(np.multiply(e, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, str(ang), tuple(np.multiply(left_shoulder, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                if angle > 60.0 and angle < 80.0 and a > 60.0 and a < 80.0:
                    count = 10
                    cv2.imwrite("selfi.png", original_frame)
                    break
                elif ang > 150.0 and ang < 170 and angle >= 30 and angle <= 40:
                    count = 20
                    cv2.imwrite("selfi.png", original_frame)
                    break

                mp_drawing.draw_landmarks(image, result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2,
                                                                 circle_radius=2),
                                          mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2,
                                                                 circle_radius=2))
            else:
                cv2.putText(image, 'No landmarks detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                            2)

            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    cap.release()
c = count
print(c)
if c == 10 or count == 20:
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        frame = rescale_frame(frame, percent=160)
        original_frame = frame.copy()
        labels = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            time_slot = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            file = f'selfie-{time_slot}.png'
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = keras.preprocessing.image.img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                preds = classifier.predict(roi)[0]
                label_with_score = class_labels[preds.argmax()]
                # Extract the score from the label (after the colon)
                label, score = label_with_score.split(':')
                score = int(score)  # Convert to integer

                label_position = (x, y)
                cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

                if score > 50:
                    cv2.imwrite(file, original_frame)
                    c = c + score
                    print(c)
                    break

            else:
                cv2.putText(frame, 'No Face Found', label_position, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                if c > 0:
                    cv2.imwrite("selfi.png", original_frame)
                    total = c + 80
            cv2.imshow('Emotion Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q') or c >= 70:
            break

    cap.release()

print(c)
