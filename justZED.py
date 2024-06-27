import cv2 

cap = cv2.VideoCapture(6)

while(True): 
    ret, frame = cap.read()  
    cv2.imshow('R2 Eyes', frame) 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
cap.release()
cv2.destroyAllWindows()