import cv2

def draw_center_line(frame):
    # Get frame dimensions
    height, width, _ = frame.shape
    
    # Calculate center point
    center_x = (width // 4)+20
    
    # Draw a vertical line at the center
    cv2.line(frame, (center_x, 0), (center_x, height), (0, 255, 0), thickness=2)
    
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
