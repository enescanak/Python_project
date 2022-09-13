import cv2
import mediapipe as mp
import time

mp_facedetector = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils


cap = cv2.VideoCapture("/home/enes/video_detect/people_face.mp4")
fourcc = cv2.VideoWriter_fourcc(*'MJPG')

writer1 = cv2.VideoWriter('detectron.avi', fourcc, 28, (1280, 720))
with mp_facedetector.FaceDetection(min_detection_confidence = 0.7) as face_detection:
    
    while cap.isOpened():
        _,image = cap.read()

        start = time.time()

        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        results = face_detection.process(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.detections:
            for id, detection in enumerate(results.detections):
                mp_draw.draw_detection(image, detection)

        end = time.time()
        totalTime = end -start

        fps = 1/totalTime

        print("FPS", fps)
        writer1.write(image)
        cv2.imshow("detec", image)

        key = cv2.waitKey(1)
        if key == 27:
            break

cap.release()
writer1.release()

 