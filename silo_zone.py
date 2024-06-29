def intersection_area(rect1, rect2):
    # rect1 and rect2 are tuples in the format (x, y, width, height)
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    
    # Calculate the coordinates of the intersection rectangle
    left_x = max(x1, x2)
    right_x = min(x1 + w1, x2 + w2)
    bottom_y = max(y1, y2)
    top_y = min(y1 + h1, y2 + h2)
    
    # Check if there is an intersection (width and height are positive)
    if left_x < right_x and bottom_y < top_y:
        intersection_width = right_x - left_x
        intersection_height = top_y - bottom_y
        intersection_area = intersection_width * intersection_height
    else:
        intersection_area = 0
    
    return intersection_area

# Example usage:
rect1 = (2, 2, 5, 3)  # (x, y, width, height)
rect2 = (4, 2, 4, 2)  # (x, y, width, height)
area_rect2 = rect2[2]*rect2[3]
int_area = intersection_area(rect1, rect2)
if int_area > ((60*area_rect2)/100):
    print("Inside")
else:
    print("Outside")
print("Area of intersection:", intersection_area(rect1, rect2))





# import cv2
# import numpy as np
# import time
# def detect_yellow(frame): 
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
     
#     # lower_yellow = np.array([20, 100, 100])
#     # upper_yellow = np.array([30, 255, 255])
    
#     lower_blue = np.array([100, 50, 50])
#     upper_blue = np.array([130, 255, 255])

#     mask = cv2.inRange(hsv, lower_blue, upper_blue)
     
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
     
#     for contour in contours:
#         x, y, w, h = cv2.boundingRect(contour)
#         width = frame.shape[1]
#         height = frame.shape[0]
#         are = w*h
#         if are > 10000 and x <= width/1 and x+w <= width/1:
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
#             print(width/2, height)
#             print(x, y, w, h)
    
#     return frame 
# cap = cv2.VideoCapture(2)

# while(True): 
#     ret, frame = cap.read() 
#     detected = detect_yellow(frame) 
#     cv2.imshow('Yellow Color Detection for R2', detected) 
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
 
# cap.release()
# cv2.destroyAllWindows()
















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