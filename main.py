# TechVidvan hand Gesture Recognizer

# import necessary packages

import time
import webbrowser
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
from _ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# init other libs
devices = AudioUtilities.GetSpeakers()
interface = devices. Activate(IAudioEndpointVolume.__iid__, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Load the gesture recognizer model
model = load_model('mp_hand_gesture')

# Load class names
f = open('gesture.names', 'r')
classNames = f.read().split('\n')
f.close()
# print(classNames)


# Initialize the webcam
cap = cv2.VideoCapture(1)

while True:
    # Read each frame from the webcam
    _, frame = cap.read()

    x, y, c = frame.shape

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get hand landmark prediction
    result = hands.process(framergb)

    # print(result)

    className = ''
    prev = ''

    # post process the result
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                # print(id, lm)
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)

                landmarks.append([lmx, lmy])

            # Drawing landmarks on frames
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            # Predict gesture
            prediction = model.predict([landmarks])
            # print(prediction)
            classID = np.argmax(prediction)

            className = classNames[classID]

        # show the prediction on the frame
        cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2, cv2.LINE_AA)

        if className == 'okay' and prev != 'okay':
            prev = 'okay'
            webbrowser.open('https://www.youtube.com/@gat_aiml', new=2)
            time.sleep(4)

        elif className == 'thumbs up' and prev != 'thumbs up':
            prev = 'thumbs up'
            # increase volume
            volume.SetMasterVolumeLevel(-25.0, None)
            time.sleep(1)

        elif className == 'thumbs down' and prev != 'thumbs down':
            prev = 'thumbs down'
            # increase volume
            time.sleep(1)

        elif (className == 'rock' or className == 'fist') and (prev != 'thumbs down'):
            prev = 'rock'
            # increase volume
            time.sleep(1)

    # Show the final output
    cv2.imshow("Output", frame)

    if cv2.waitKey(1) == ord('q'):
        break

# release the webcam and destroy all active windows
cap.release()

cv2.destroyAllWindows()
