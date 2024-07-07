import cv2

def draw_center_line(frame):
    # Get frame dimensions
    height, width, _ = frame.shape
    
    # Calculate center point 
    
    # Draw a vertical line at the center
    # cv2.line(frame, (230, 0), (220, height), (0, 255, 0), thickness=2)
    offset = int((width // 2)/3)
    print(offset, offset*2)
    # cv2.line(frame, (0, 170), (width, 170), (0, 255, 0), thickness=2)
    # cv2.line(frame, (0, 200), (width, 200), (200, 255, 255), thickness=2)
    # cv2.line(frame, (0, 210), (width, 210), (255, 0, 255), thickness=2)
    cv2.line(frame, (width//4, 0), (width//4, height), (255, 0, 255), thickness=2)
    offs = 38
    margin = 7
    # cv2.line(frame, ((width//4)+offs-margin, 0), ((width//4)+offs-margin, height), (255, 255, 0), thickness=2)
    # cv2.line(frame, ((width//4)+offs, 0), ((width//4)+offs, height), (0, 255, 0), thickness=2)
    # cv2.line(frame, ((width//4)+offs+margin, 0), ((width//4)+offs+margin, height), (255, 255, 0), thickness=2)

    # cv2.line(frame, (0+offset, 0), (0+offset, height), (0, 255, 0), thickness=2)
    # cv2.line(frame, (0+offset+offset, 0), (0+offset+offset, height), (200, 255, 255), thickness=2) 

    return frame

def main():
    # Open default camera (usually webcam)
    cap = cv2.VideoCapture(2)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        # Draw vertical line at center
        frame_with_line = draw_center_line(frame)
        
        # Display the modified frame
        cv2.imshow('Frame with Center Line', frame_with_line)
        
        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
