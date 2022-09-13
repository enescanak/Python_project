from Detector import *
import sys

print(sys.path)

detector = Detector(model_type = "LVIS")

#detector.onImage("img.jpeg")

detector.onVideo("/home/enes/yolov7/9.mp4")



#"/home/enes/Desktop/a/3.mp4"
'''

while True:
    _,frame = cap.read()
    str_bas = cv2_base64(frame)
    frame = base64_cv2(str_bas)
    imgg = onVideo(frame)
    cv2.imshow("Results", imgg)
    key = cv2.waitKey(1)

    if key == 27:
        break
cap.release()
'''
