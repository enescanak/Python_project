import cv2
import numpy as np
import mediapipe
import base64
import time




mpHands = mediapipe.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mediapipe.solutions.drawing_utils
tipIds = [4, 8, 12, 16, 20]

def cv2_base64(image):
  base64_str = cv2.imencode('.jpg',image)[1].tostring()
  base64_str = base64.b64encode(base64_str)
  return base64_str 
def base64_cv2(base64_str):
  imgString = base64.b64decode(base64_str)
  nparr = np.fromstring(imgString,np.uint8) 
  image = cv2.imdecode(nparr,cv2.IMREAD_COLOR)
  return image

def add(i):
    imgg = base64_cv2(i)
    imgg = cv2.putText(imgg, 'YongaTek', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (255, 0, 0), 2, cv2.LINE_AA)
    imgg = med_hand_try(imgg)
    base = cv2_base64(imgg)
    return base

def med_hand_try(frame):
    
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    hlms = hands.process(imgRGB)
    # print(hlms.multi_hand_landmarks) # x,y,z orandır.
    height, width, channel = frame.shape

    landmark_list = []

    if hlms.multi_hand_landmarks:
        # print(len(hlms.multi_hand_landmarks))
        for handlandmarks in hlms.multi_hand_landmarks:
            # print(handlandmarks.landmark)

            for fingerNum, landmark in enumerate(handlandmarks.landmark):
                positionX, positionY = int(landmark.x * width), int(landmark.y * height)

                landmark_list.append([fingerNum, positionX, positionY])
                x_max = 0
                y_max = 0
                x_min = width
                y_min = height
                for lm in handlandmarks.landmark:
                    x, y = int(lm.x * width), int(lm.y * height)
                    if x > x_max:
                        x_max = x
                    if x < x_min:
                        x_min = x
                    if y > y_max:
                        y_max = y
                    if y < y_min:
                        y_min = y
                    alan = abs((x_max-x_min) * (y_max-y_min))/100
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                if (fingerNum == 8 and landmark.y > handlandmarks.landmark[2].y):
                    #print("okaa")
                    durum = 1
                if (handlandmarks.landmark[4].y > handlandmarks.landmark[3].y):
                    #print("not okaa")
                    durum = 2
                mpDraw.draw_landmarks(frame, handlandmarks, mpHands.HAND_CONNECTIONS)
                cv2.putText(frame, str(fingerNum), (positionX, positionY),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                if len(landmark_list) == 21:
                    fingers = []

                    if landmark_list[tipIds[0]][1] < landmark_list[tipIds[0] - 2][1]:
                        fingers.append(1)

                    else:
                        fingers.append(0)

                    for tip in range(1, 5):
                        if landmark_list[tipIds[tip]][2] < landmark_list[tipIds[tip] - 2][2]:
                            fingers.append(1)
                        else:
                            fingers.append(0)
                    totalFingers = fingers.count(1)

    return frame