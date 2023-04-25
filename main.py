import mediapipe as mp
import cv2
import numpy as np
import time
import math
import MouseTracking
import VolumeTraking
from threading import Thread

#Диапазон громкости
minVol = -65
maxVol = -0

cordX1 = [0][0]
#Объекты
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mouse = MouseTracking.MouseTrack
volume = VolumeTraking.VolumeTracking


def drawline(img, pt1, pt2, color, thickness=1, style='dotted', gap=20):
    dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** .5
    pts = []
    for i in np.arange(0, dist, gap):
        r = i / dist
        x = int((pt1[0] * (1 - r) + pt2[0] * r) + .5)
        y = int((pt1[1] * (1 - r) + pt2[1] * r) + .5)
        p = (x, y)
        pts.append(p)
    if style == 'dotted':
        for p in pts:
            cv2.circle(img, p, thickness, color, -1)
    else:
        s = pts[0]
        e = pts[0]
        i = 0
        for p in pts:
            s = e
            e = p
            if i % 2 == 1:
                cv2.line(img, s, e, color, thickness)
            i += 1

################################
wCam, hCam = 1920,1080
################################
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8 , max_num_hands=1) as hands:
     while cap.isOpened():
        ret, frame = cap.read()

        # BGR 2 RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Flip on horizontal
        image = cv2.flip(image, 1)

        # Set flag
        image.flags.writeable = False

        # Detections
        results = hands.process(image)

        # Set flag to true
        image.flags.writeable = True

        # RGB 2 BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # Detections
        #print(results.multi_hand_landmarks)
        a = mp_hands.HandLandmark.INDEX_FINGER_MCP

        # Rendering results
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(255, 0, 0),   thickness=2, circle_radius=5),
                                          mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=3, circle_radius=2)
                                          )

            for hand_landmarks in results.multi_hand_landmarks:
                lmList = []
                for id, lm in enumerate(hand_landmarks.landmark):
                    h, w, c = image.shape
                   #h, w, c = 333,333,333
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])
                    # the following two lines are simply for styling our fingers
                    tips = [0, 4, 8, 12, 16, 20]
                    #tips = [0, 4, 8]

                    if id in tips:
                        cv2.circle(image, (cx, cy), 15, (192, 192, 192), cv2.FILLED)
                #Рисуем линию
                drawline(image, (lmList[4][1], lmList[4][2]), (lmList[8][1], lmList[8][2]), (192, 192, 192),thickness=3, style='dotted', gap=10)
                #Посчет растояние
                distanse = (int(math.hypot(lmList[8][1] - lmList[4][1], lmList[8][2] - lmList[4][2]) / 2)) / 2

                #Вызов котроля громкости
                volume.VolumeControl(distanse,minVol,maxVol)

                #print(lmList[8][1])
                #print (image.shape)

                #Вызов котроля мыши
                cordX = lmList[8][1]
                cordY = lmList[8][2]
                #print(cordX,cordY)
                #mouse.MouseControl(cordX, cordY)
                th = Thread(target=mouse.MouseControl(cordX, cordY), args=(1,))
                #th.start()

        cv2.imshow('Hand Tracking', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break


cap.release()
cv2.destroyAllWindows()



#intNum = int(main())

'''while cap.isOpened():
    cords = main()
    print(cords)
    #th1 = Thread(target=main(), args=(1,))
    #th2 = Thread(target=mouse.MouseControl(cords[0], cords[1]), args=(1,))
    #th1.start()
    #th2.start()
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break'''
"fvf"

