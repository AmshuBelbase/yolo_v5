import cv2
import numpy as np
import time
def detect_yellow(frame): 
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
     
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
     
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
     
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
     
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        width = frame.shape[1]
        height = frame.shape[0]
        are = w*h
        if are > 80000 and x <= width/2 and x+w <= width/2:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            print(width/2, height)
            print(x, y, w, h)
    
    return frame 
cap = cv2.VideoCapture(2)

while(True): 
    ret, frame = cap.read() 
    detected = detect_yellow(frame) 
    cv2.imshow('Yellow Color Detection for R2', detected) 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
cap.release()
cv2.destroyAllWindows()
















# import cv2
# import numpy as np

# def empty(a):
#     pass

# cv2.namedWindow("TrackBars")
# cv2.resizeWindow("TrackBars", 600, 250)

# cv2.createTrackbar("Hue Min", "TrackBars", 0, 179, empty)
# cv2.createTrackbar("Hue Max", "TrackBars", 179, 179, empty)

# cv2.createTrackbar("Sat Min", "TrackBars", 0, 255, empty)
# cv2.createTrackbar("Sat Max", "TrackBars", 255, 255, empty)

# cv2.createTrackbar("Val Min", "TrackBars", 0, 255, empty)
# cv2.createTrackbar("Val Max", "TrackBars", 255, 255, empty)


# # Initialize camera
# cap = cv2.VideoCapture(0)

# while(True):
#     # Capture frame-by-frame
#     ret, img = cap.read()

# # while True:
# #     img = cv2.imread()
#     resize = cv2.resize(img, (400,300))

#     h_min = cv2.getTrackbarPos("Hue Min","TrackBars")
#     h_max = cv2.getTrackbarPos("Hue Max","TrackBars")
#     s_min = cv2.getTrackbarPos("Sat Min","TrackBars")
#     s_max = cv2.getTrackbarPos("Sat Max","TrackBars")
#     v_min = cv2.getTrackbarPos("Val Min","TrackBars")
#     v_max = cv2.getTrackbarPos("Val Max","TrackBars")
#     print(h_min, h_max, s_min, s_max, v_min, v_max)
#     lower = np.array([h_min, s_min, v_min])
#     upper = np.array([h_max, s_max, v_max])
#     mask = cv2.inRange(resize, lower, upper)
#     cv2.imshow("img", resize)
#     cv2.imshow("Ouput", mask)
#     result = cv2.bitwise_and(resize,resize,mask=mask)
#     cv2.imshow("Result", result)
#     if cv2.waitKey(1) & 0xFF == ord('a'):
#         break