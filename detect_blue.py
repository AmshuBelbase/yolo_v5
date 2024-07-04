import cv2
import numpy as np

# Function to filter contours by area
def filter_contours(contours, min_area):
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            filtered_contours.append(contour)
    return filtered_contours

# Function to check if rectangle is at least half inside contours
def is_rectangle_half_inside(frame, blue_mask, rect, contours):
    # Convert rectangle corners to list of points
    rect_pts = np.array([[rect[0], rect[1]], 
                         [rect[0] + rect[2], rect[1]], 
                         [rect[0] + rect[2], rect[1] + rect[3]], 
                         [rect[0], rect[1] + rect[3]]], dtype=np.int32)
    
    rect_area = rect[2] * rect[3]  # Total area of the rectangle
    
    for contour in contours:
        intersection_area = 0
        
        for i in range(len(rect_pts)):
            pt1 = tuple(rect_pts[i])
            pt2 = tuple(rect_pts[(i + 1) % len(rect_pts)])
            
            # Calculate intersection area between rectangle edge and contour
            edge_mask = np.zeros_like(frame)
            cv2.line(edge_mask, pt1, pt2, (255, 255, 255), 2)
            
            edge_mask = cv2.cvtColor(edge_mask, cv2.COLOR_BGR2GRAY)
            edge_intersection = cv2.bitwise_and(edge_mask, blue_mask)
            intersection_area += np.sum(edge_intersection) / 255
        
        if intersection_area >= rect_area / 2:
            return True
    
    return False

# Function to get top-rightmost coordinate of a contour
def get_top_rightmost(contour):
    # Calculate bounding rectangle for the contour
    x, y, w, h = cv2.boundingRect(contour)
    # Calculate top-right corner coordinates
    top_right_x = x + w
    top_right_y = y
    return (top_right_x, top_right_y)

# Main function to process camera frames
def detect_blue_color():
    cap = cv2.VideoCapture(2)  # Open default camera
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define range of blue color in HSV
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([108, 255, 255])
        
        # Threshold the HSV image to get only blue colors
        blue_mask = cv2.inRange(hsv_frame, lower_blue, upper_blue)

        cv2.imshow('blue_mask', blue_mask)
        
        # Find contours
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        min_contour_area = 5000  # Adjust this threshold as needed
        filtered_contours = filter_contours(contours, min_contour_area)
        
        # Draw filtered contours on the original frame
        frame_with_filtered_contours = frame.copy()
        cv2.drawContours(frame_with_filtered_contours, filtered_contours, -1, (0, 255, 0), 2)
        
        # Example rectangle (x, y, width, height)
        rectangle = (300, 105, 10, 30)
        cv2.rectangle(frame_with_filtered_contours, (rectangle[0], rectangle[1]),
                          (rectangle[0] + rectangle[2], rectangle[1] + rectangle[3]), (255, 0, 255), 2)
        
        # Get top-rightmost coordinate among filtered contours
        if filtered_contours:
            top_rightmost_point = None
            for contour in filtered_contours:
                top_right = get_top_rightmost(contour)
                if top_rightmost_point is None or top_right[0] > top_rightmost_point[0]:
                    top_rightmost_point = top_right
            
            # Draw a circle at the top-rightmost point
            if top_rightmost_point is not None:
                print(top_rightmost_point)
                cv2.rectangle(frame_with_filtered_contours, (0, 0),
                          (top_rightmost_point[0], top_rightmost_point[1]), (255, 0, 255), 2)
                cv2.circle(frame_with_filtered_contours, top_rightmost_point, 5, (0, 0, 255), -1)
        
        # Display the frame with filtered contours, rectangle, and top-rightmost point
        cv2.imshow('Blue Contours (Filtered)', frame_with_filtered_contours)
        
        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()

# Run the main function
detect_blue_color()


# import cv2
# import numpy as np

# # Function to filter contours by area
# def filter_contours(contours, min_area):
#     filtered_contours = []
#     for contour in contours:
#         area = cv2.contourArea(contour)
#         if area > min_area:
#             filtered_contours.append(contour)
#     return filtered_contours

# # Function to check if rectangle is at least half inside contours
# def is_rectangle_half_inside(frame, blue_mask,rect, contours):
#     # Convert rectangle corners to list of points
#     rect_pts = np.array([[rect[0], rect[1]], 
#                          [rect[0] + rect[2], rect[1]], 
#                          [rect[0] + rect[2], rect[1] + rect[3]], 
#                          [rect[0], rect[1] + rect[3]]], dtype=np.int32)
    
#     rect_area = rect[2] * rect[3]  # Total area of the rectangle
    
#     for contour in contours:
#         intersection_area = 0
        
#         for i in range(len(rect_pts)):
#             pt1 = tuple(rect_pts[i])
#             pt2 = tuple(rect_pts[(i + 1) % len(rect_pts)])
            
#             # Calculate intersection area between rectangle edge and contour
#             edge_mask = np.zeros_like(frame)
#             cv2.line(edge_mask, pt1, pt2, (255, 255, 255), 2)
            
#             edge_mask = cv2.cvtColor(edge_mask, cv2.COLOR_BGR2GRAY)
#             edge_intersection = cv2.bitwise_and(edge_mask, blue_mask)
#             intersection_area += np.sum(edge_intersection) / 255
        
#         if intersection_area >= rect_area / 2:
#             return True
    
#     return False

# # Main function to process camera frames
# def detect_blue_color():
#     cap = cv2.VideoCapture(2)  # Open default camera
#     if not cap.isOpened():
#         print("Error: Could not open camera.")
#         return
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Could not read frame.")
#             break
        
#         hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
#         # Define range of blue color in HSV
#         lower_blue = np.array([90, 50, 50])
#         upper_blue = np.array([108, 255, 255])
        
#         # Threshold the HSV image to get only blue colors
#         blue_mask = cv2.inRange(hsv_frame, lower_blue, upper_blue)

#         cv2.imshow('blue_mask', blue_mask)
        
#         # Find contours
#         contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
#         # Filter contours by area
#         min_contour_area = 5000  # Adjust this threshold as needed
#         filtered_contours = filter_contours(contours, min_contour_area)
        
#         # Draw filtered contours on the original frame
#         frame_with_filtered_contours = frame.copy()
#         cv2.drawContours(frame_with_filtered_contours, filtered_contours, -1, (0, 255, 0), 2)
        
#         # Example rectangle (x, y, width, height)
#         rectangle = (300, 105, 10, 30)
#         cv2.rectangle(frame_with_filtered_contours, (rectangle[0], rectangle[1]),
#                           (rectangle[0] + rectangle[2], rectangle[1] + rectangle[3]), (255, 0, 255), 2)
#         # Check if rectangle is at least half inside the contours
#         if is_rectangle_half_inside(frame, blue_mask, rectangle, filtered_contours):
#             cv2.rectangle(frame_with_filtered_contours, (rectangle[0], rectangle[1]),
#                           (rectangle[0] + rectangle[2], rectangle[1] + rectangle[3]), (0, 0, 255), 2)
        
#         # Display the frame with filtered contours and rectangle
#         cv2.imshow('Blue Contours (Filtered)', frame_with_filtered_contours)
        
#         # Exit loop if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
    
#     # Release the camera and close all windows
#     cap.release()
#     cv2.destroyAllWindows()

# # Run the main function
# detect_blue_color()
