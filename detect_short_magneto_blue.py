# # YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
# """
# Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

# Usage - sources:
#     $ python detect.py --weights yolov5s.pt --source 0                               # webcam
#                                                      img.jpg                         # image
#                                                      vid.mp4                         # video
#                                                      screen                          # screenshot
#                                                      path/                           # directory
#                                                      list.txt                        # list of images
#                                                      list.streams                    # list of streams
#                                                      'path/*.jpg'                    # glob
#                                                      'https://youtu.be/LNwODJXcvt4'  # YouTube
#                                                      'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
    
#     $ python3 detect.py --weights runs/train/exp/weights/best.pt --source videos/video19.mp4 --data data/custom_data.yaml
#     $ python3 detect_short.py 

# amshu
# """

import argparse 
import math
import os
import platform
import time
import sys
import cv2
from pathlib import Path
import numpy as np
import torch
import serial 
import matplotlib.pyplot as plt  

gint_area = 0
delay_stat = False
neglect_ball_class = 0

bot_default_turn_speed = 27
bot_default_turn_speed_ball = 30

ball_silo = 0 # 0 : ball | 1 : silo
almost_aligned = False
cam_source = 2
serial_port = '/dev/ttyACM0'
baud_rate = 115200 
ser = serial.Serial(serial_port, baud_rate, timeout=1)

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
    weights=ROOT / "best.pt",  # model path or triton URL
    source=cam_source,  # file/dir/URL/glob/screen/0(webcam)
    data=ROOT / "custom_data.yaml",  # dataset.yaml path
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False,  # show results
    save_txt=False,  # save results to *.txt 
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features 
    project=ROOT / "runs/detect",  # save results to project/name
    name="exp",  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    vid_stride=1,  # video frame-rate stride
):
    global ball_silo, gint_area, delay_stat
    source = str(source)  
    webcam = source.isnumeric() or source.endswith(".streams") 

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        print(dataset)
        bs = len(dataset)  
    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:  
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim1
            if model.xml and im.shape[0] > 1:
                ims = torch.chunk(im, im.shape[0], 0) 

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            if model.xml and im.shape[0] > 1:
                pred = None
                for image in ims:
                    if pred is None:
                        pred = model(image, augment=augment, visualize=visualize).unsqueeze(0)
                    else:
                        pred = torch.cat((pred, model(image, augment=augment, visualize=visualize).unsqueeze(0)), dim=0)
                pred = [pred, None]
            else:
                pred = model(im, augment=augment, visualize=visualize)
        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det) 

        if ser.in_waiting > 0: 
            data_from_pico = ser.readline().strip().decode()
            print(f"Received from Pico: {data_from_pico}") 
            if int(data_from_pico) == 7:
                ball_silo = 1 
                gint_area = 0
                print("Received 7 | Search Silo")
            elif int(data_from_pico) == 1:
                ball_silo = 0
                print("Received 1 | Search Ball")
        else:
            print("..")

        # Process predictions 
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0 = path[i], im0s[i].copy()
                s += f"{i}: "
            else:
                p, im0 = path, im0s.copy()

            width = im0.shape[1]
            height = im0.shape[0]
            p = Path(p)  # to Path 
            s += "%gx%g " % im.shape[2:]  # print string 
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            prio_silo = 0

            if len(det):
                # print(det)
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results 
                short_det = det
                
                nearest_c = -1
                
                nearest = width/2
                final_top_left_x = width/4
                final_top_left_y = height
                final_bottom_right_x = width/4
                final_bottom_right_y = height
                final_xyxy = (final_top_left_x, final_top_left_y, final_bottom_right_x, final_bottom_right_y)
                counter = 0 

                prio_silo = 0 

                for *xyxy, conf, cls in reversed(short_det):
                    counter +=1
                    c = int(cls)  # integer class
                    # print(c)
                    label = names[c] if hide_conf else f"{names[c]}"
                    confidence = float(conf)
                    confidence_str = f"{confidence:.2f}"  
                    top_left_x = xyxy[0]
                    top_left_y = xyxy[1]
                    bottom_right_x = xyxy[2]
                    bottom_right_y = xyxy[3]
                    box_width = bottom_right_x - top_left_x
                    box_height = bottom_right_y - top_left_y 
                    if(top_left_x < width/2 and bottom_right_x < width/2):
                        if view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                            annotator.box_label(xyxy, label, color=colors(c, True))
                        if c == neglect_ball_class:
                            c = 2
                        if (c == 0 or c == 2 or c == 1) and ball_silo == 0:
                            # Define the original range
                            in_width_min = 0
                            in_width_max = width / 2

                            # Define the new range
                            new_min = -(width / 4)
                            new_max = (width / 4)


                            # Value to map
                            value = (top_left_x + bottom_right_x)//2

                            # Map the value to the new range
                            mapped_value = ((value - in_width_min) / (in_width_max - in_width_min)) * (new_max - new_min) + new_min
                            actual_top_left_y = height - top_left_y

                            h_dist = (float(abs(mapped_value)) ** 2 + float(actual_top_left_y ** 2)) ** 0.5

                            if(h_dist < nearest):
                                is_inside = False  
                                offset = 15   
                                if c == 2:  
                                    counter_in = 0
                                    for *xyxy_in, conf_in, cls_in in reversed(short_det):
                                        if cls_in == neglect_ball_class:
                                            cls_in = 2
                                        if counter != counter_in and cls_in != 2 and not is_inside and xyxy_in[0] < width/2 and xyxy_in[2] < width/2: 
                                            vertices_rect_in = [
                                                (xyxy_in[0], xyxy_in[1]),
                                                (xyxy_in[0], xyxy_in[3]),
                                                (xyxy_in[2], xyxy_in[1]),
                                                (xyxy_in[2], xyxy_in[3])
                                            ]
                                            annotator.box_label(xyxy_in, "checck", color=(100, 100, 0)) 
                                            for vertex in vertices_rect_in:
                                                x, y = vertex
                                                if (xyxy[0] - offset) <= x <= (xyxy[2] + offset) and (xyxy[1] - offset) <= y <= (xyxy[3]+ offset):
                                                    is_inside = True 
                                        counter_in += 1
                                if is_inside or c!=2: 
                                    inside_blue = False
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
                                    def detect_blue(frame, ball_xyxy, inside_blue_f): 
                                        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                                        # lower_yellow = np.array([20, 100, 100])
                                        # upper_yellow = np.array([30, 255, 255])
                                        
                                        lower_blue = np.array([100, 50, 50])
                                        upper_blue = np.array([130, 255, 255])

                                        mask = cv2.inRange(hsv, lower_blue, upper_blue)
                                        
                                        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                        width =  frame.shape[1] 
                                        height = frame.shape[0]  
                                        for contour in contours:
                                            x, y, w, h = cv2.boundingRect(contour)
                                            are = w*h
                                            xyxy_yellow = [x, y, x+w, y+h]
                                            # msg_d = "Area Unchecked"+str(are)
                                            # annotator.box_label(xyxy_yellow, msg_d, color=(0,0,0))
                                                
                                            if are > 14000 and x <= width/1 and x+w <= width/1: #80000 
                                                xyxy_yellow = [x, y, x+w, y+h]
                                                msg_d = "Blue Area"+str(are)
                                                annotator.box_label(xyxy_yellow, msg_d, color=(130, 255, 255)) 
                                                # Example usage:
                                                rect1 = (0, 0, x+w, y+h)  # (x, y, width, height)
                                                rect2 = (ball_xyxy[0], ball_xyxy[1], ball_xyxy[2]-ball_xyxy[0], ball_xyxy[3]-ball_xyxy[1])  # (x, y, width, height)
                                                area_rect2 = rect2[2]*rect2[3]
                                                int_area = intersection_area(rect1, rect2)
                                                if int_area > ((90*area_rect2)/100):
                                                    inside_blue_f = True
                                                    print("Inside")
                                                    delay_stat = True
                                                else:
                                                    print("Outside")
                                                    delay_stat = False
                                                print("Area of intersection:", intersection_area(rect1, rect2))                                         

                                        return inside_blue_f  
                                     
                                    frame = im0s[i].copy()
                                    inside_blue = detect_blue(frame, xyxy, inside_blue)
                                    if not inside_blue:
                                        nearest = h_dist 
                                        nearest_c = (top_left_y*10)+c
                                        # if nearest_c != 2:
                                        #     nearest_c = top_left_y
                                        final_top_left_x = xyxy[0]
                                        final_top_left_y = xyxy[1]
                                        final_bottom_right_x = xyxy[2]
                                        final_bottom_right_y = xyxy[3]
                                        final_xyxy = (final_top_left_x - offset, final_top_left_y - offset, final_bottom_right_x + offset, final_bottom_right_y + offset)
                                
                                # if is_inside or c!=2:
                                #     nearest = h_dist 
                                #     nearest_c = c
                                #     if nearest_c != 2:
                                #         nearest_c = top_left_y
                                #     final_top_left_x = xyxy[0]
                                #     final_top_left_y = xyxy[1]
                                #     final_bottom_right_x = xyxy[2]
                                #     final_bottom_right_y = xyxy[3]
                                #     final_xyxy = (final_top_left_x - offset, final_top_left_y - offset, final_bottom_right_x + offset, final_bottom_right_y + offset)

                        elif c != 0 and c != 1 and c!= 2 and ball_silo == 1:
                            defence_mode = 1 # 1 : defence | 0 : attack | 2 : opponent weak
                            if defence_mode == 1:
                                arr_prio = [30, 30, 30, 3, 4, 5, 2, 1, 1, 1, 30,30,30,30,30,30,30,30]
                            elif defence_mode == 0:
                                arr_prio = [30, 30, 30, 3, 4, 5, 1, 1, 1, 2, 30,30,30,30,30,30,30,30]
                            elif defence_mode == 2:
                                arr_prio = [30, 30, 30, 4, 3, 5, 1, 1, 1, 2, 30,30,30,30,30,30,30,30] 

                            # if arr_prio[c] != 30 and (arr_prio[c] < arr_prio[prio_silo] or (arr_prio[c] == arr_prio[prio_silo] and abs(xyxy[0] - int(width/4)) < abs(final_top_left_x - int(width/4)))):     
                            #     prio_silo = c
                            #     final_top_left_x = xyxy[0]
                            #     final_top_left_y = xyxy[1]
                            #     final_bottom_right_x = xyxy[2]
                            #     final_bottom_right_y = xyxy[3]
                            #     final_xyxy = (final_top_left_x, final_top_left_y, final_bottom_right_x, final_bottom_right_y)  
                            
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
                            

                            margin = 0

                            silo_center =  xyxy[0] + ((xyxy[2]-xyxy[0])/2) - (int(width/4)) # 73

                            bot_center = (int(width/4)-3, 0, int(width/4)-3, height)
                            annotator.box_label(bot_center, "Center Line L", color=colors(c, True))

                            bot_center = (int(width/4)+3, 0, int(width/4)+3, height)
                            annotator.box_label(bot_center, "Center Line R", color=colors(c, True))

                            rect2 = (xyxy[0]+((xyxy[2]-xyxy[0])/2), xyxy[1], xyxy[2]-((xyxy[2]-xyxy[0])/2), xyxy[3])
                            annotator.box_label(rect2, str(xyxy[0] + ((xyxy[2]-xyxy[0])/2)- (int(width/4)+margin)), color=(0, 100, 0))
                            
                            offset = int((width // 2)/3) 
                            rect1 = (offset, 0, offset, height)  # (x, y, width, height)
                            rect2 = (xyxy[0], xyxy[1], xyxy[2]-xyxy[0], xyxy[3]-xyxy[1])  # (x, y, width, height) 
                            int_area = intersection_area(rect1, rect2)
                            # print("Intersection Area: ", int_area, " Greatest Intersection Area: ", gint_area)
                            rect1 = (offset, 0, offset+offset, height)
                            annotator.box_label(rect1, "SILO RANGE", color=(0, 0, 0)) 

                            # print("height: ", (xyxy[3] - xyxy[1]))
                            # print("width: ", (xyxy[2] - xyxy[0]))
                            # print("sILO cENTER: ", silo_center)

                            # just_check = (int(width/4)-160, 0, int(width/4)+160, height)
                            # annotator.box_label(just_check, "CHECK RANGE", color=(255, 0, 255))

                            if abs(silo_center) < 175: 
                                print(" !!!!!!!!!!!!!!!!!!!!!!! ----------- ALMOST ALIGNED ----------- !!!!!!!!!!!!!!!!!!!!!!! ")
                                gint_area = 0 

                            if arr_prio[c] != 30 and int_area > 0 and int_area > gint_area and ((xyxy[3] - xyxy[1]) > 200 or (xyxy[2] - xyxy[0])>95):
                                rect2 = (xyxy[0], xyxy[1], xyxy[2], xyxy[3])
                                annotator.box_label(rect2, "SILO", color=(255, 255, 255))
                                gint_area = int_area
                                prio_silo = c
                                final_top_left_x = xyxy[0]
                                final_top_left_y = xyxy[1]
                                final_bottom_right_x = xyxy[2]
                                final_bottom_right_y = xyxy[3]
                                final_xyxy = (final_top_left_x, final_top_left_y, final_bottom_right_x, final_bottom_right_y)
                            elif gint_area == 0:
                                if arr_prio[c] != 30 and (((xyxy[3] - xyxy[1]) > 200 or (xyxy[2] - xyxy[0])>95) or (arr_prio[c] < arr_prio[prio_silo] or (arr_prio[c] == arr_prio[prio_silo] and abs(xyxy[0] - int(width/4)) < abs(final_top_left_x - int(width/4))))):     
                                    prio_silo = c
                                    final_top_left_x = xyxy[0]
                                    final_top_left_y = xyxy[1]
                                    final_bottom_right_x = xyxy[2]
                                    final_bottom_right_y = xyxy[3]
                                    final_xyxy = (final_top_left_x, final_top_left_y, final_bottom_right_x, final_bottom_right_y) 
                            elif arr_prio[c] != 30:
                                prio_silo = 16
                                print("FOllowing Movement")
                            print("searching Silos")

                print("Nearest:", nearest)
                if view_img:  # Add bbox to image
                    c = int(cls)  # integer class
                    label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                    annotator.box_label(final_xyxy, "nearest", color=(255, 255, 0)) 
 
                if ball_silo == 0 and nearest != width/2:
                    dist_ball = 0
                    box_width = final_bottom_right_x - final_top_left_x
                    box_height = final_bottom_right_y - final_top_left_y 
                    if(box_width>box_height):
                        dist_ball = box_width
                    else:
                        dist_ball = box_height


                    dist_ball = dist_ball.cpu().numpy()  # Move to CPU and convert to NumPy array

                    width_height_max = [16, 18, 20, 21, 25, 28, 33, 40, 48, 54, 58, 64, 69, 74, 85]
                    ball_distance = [310, 280, 250, 230, 200, 180, 150, 125, 100, 90, 80, 70, 60, 50, 40]

                    dist_ball = np.interp(dist_ball, width_height_max, ball_distance)
                    dist_ball = int(dist_ball)
                        
                    # Define the input and output range
                    i_min = 40
                    i_max = 310
                    o_min = 50
                    o_max = 20
                    scale_factor = 100
                    if(dist_ball > 310 or dist_ball < 40):
                        scale_factor = 140
                    else:
                        scale_factor = (dist_ball-i_min) * (o_max-o_min) / (i_max - i_min) + o_min

                    if scale_factor <= 10:
                        scale_factor *= 7
                    elif scale_factor <= 20:
                        scale_factor *= 5
                    elif scale_factor <= 30:
                        scale_factor *= 3
                    elif scale_factor <= 40:
                        scale_factor *= 2
                    elif scale_factor <= 50:
                        scale_factor *= 1.5
                    # scale_factor *= 1.5

                    # LOGGER.info(f"Width: {width}, Height: {height}")  
                     

                    linear_x = (final_top_left_x - int(width/4))/scale_factor #nominal 40
                    linear_y = (final_top_left_y - height)/scale_factor 

                    # LOGGER.info(f'distance - {dist_ball} scale factor - {scale_factor} linear_x - {linear_x} linear_y - {linear_y}')

                    # LOGGER.info(f"X: {linear_x}, Y: {linear_y}, Z: {0}")  
                    # matrix_4x3 = np.array([[15.75, 0, -5.66909078166105],
                    #                     [0, 15.75, 5.66909078166105],
                    #                     [-15.75, 0, 5.66909078166105],
                    #                     [0, -15.75,-5.66909078166105]]) 
                    matrix_4x3 = np.array([[15.75, 0, -5.66909078166105],
                                        [0, 15.75, -5.66909078166105],
                                        [-15.75, 0, -5.66909078166105],
                                        [0, -15.75,-5.66909078166105]]) 
            
                    # matrix_3x1 = np.array([[linear_x],
                    #                     [linear_y],
                    #                     [angular_z]])
                    az = math.atan2(-linear_y, -linear_x)
                    # Move the tensors to CPU and convert to NumPy arrays 
                    linear_x_cpu = linear_x.cpu().numpy()
                    linear_y_cpu = linear_y.cpu().numpy()   

                    # LOGGER.info(f"X: {linear_x_cpu}, Y: {linear_y_cpu}, Z: {0}") 
                    # Create the matrix_3x1 using the CPU tensors
                    matrix_3x1 = np.array([linear_x_cpu, linear_y_cpu, az])  
                    result_matrix = np.dot(matrix_4x3, matrix_3x1)        
                        
                        

                    # Define floats to send
                    fr = result_matrix[0]
                    fl = result_matrix[1]
                    bl = result_matrix[2]
                    br = result_matrix[3]
                    
                    fr /=1.5
                    fl /=1.5
                    br /=1.5
                    bl /=1.5     

                    if (abs(fr) <= 20 or abs(bl) <= 20) and not (-11 <= fr <= -5):
                        fr *= 2.9 
                        fl *= 1 
                        br *= 1 
                        if final_top_left_y > 160:
                            fl *= 2.9 
                            br *= 2.9 
                        bl *= 2.9 
                    elif (abs(fr) <= 30 or abs(bl) <= 30) and not (-11 <= fr <= -5):
                        fr *= 1.7
                        fl *= 1 
                        br *= 1
                        if final_top_left_y > 160:
                            fl *= 1.7
                            br *= 1.7
                        bl *= 1.7 
                    elif abs(fr) <= 40 and abs(bl) <= 40 and not (-11 <= fr <= -5):
                        fr *= 1.3
                        fl *= 1 
                        br *= 1 
                        if final_top_left_y > 160:
                            fl *= 1.3
                            br *= 1.3
                        bl *= 1.3 

                    # Convert to bytes
                    data = (str(int(fr)) + '|' + 
                            str(int(fl)) + '|' +
                            str(int(bl)) + '|' +
                            str(int(br)) + '|' +
                            str(int(nearest_c))) + "#"
                        
                    # Send data
                    # time.sleep(0.05)
                    ser.write(data.encode())  
                    LOGGER.info(f"Sent 1: {data}")
                    # LOGGER.info(f"Front Right: {result_matrix[0]}, Front Left: {result_matrix[1]}, Back Left: {result_matrix[2]}, Back Right: {result_matrix[3]}")

                    # flag = 1 
                elif ball_silo == 1 and prio_silo != 0 and prio_silo !=16: 
                    box_width = final_bottom_right_x - final_top_left_x
                    box_height = final_bottom_right_y - final_top_left_y 
                    print(box_width, box_height)
                    scale_factor = 40   

                    # if box_height > 100 or box_width>50: 
                    #     scale_factor = 70                       

                    linear_x = (final_top_left_x + (box_width/2) - 25 - int(width/4))/scale_factor #nominal 40
                    linear_y = (final_top_left_y + (box_height/2) - height)/scale_factor 
                    
                    matrix_4x3 = np.array([[15.75, 0, -5.66909078166105],
                                        [0, 15.75, -5.66909078166105],
                                        [-15.75, 0, -5.66909078166105],
                                        [0, -15.75,-5.66909078166105]])  
                    
                    near_far = -4 #  -1 : No Detection | -3 : Near | -4 : Far | -5 : Aligned
                    silo_center =  final_top_left_x + (box_width/2) - (int(width/4))
                    if box_height > 180 or box_width>90:  # 150 75
                        near_far = -3 
                        print("Silo Center : ", silo_center) 
                    
                    # az = math.atan2(-linear_y, -linear_x)
                    az = 0
                    # Move the tensors to CPU and convert to NumPy arrays 
                    linear_x_cpu = linear_x.cpu().numpy()
                    linear_y_cpu = linear_y.cpu().numpy()   
 
                    # Create the matrix_3x1 using the CPU tensors
                    matrix_3x1 = np.array([linear_x_cpu, linear_y_cpu, az])  
                    result_matrix = np.dot(matrix_4x3, matrix_3x1)        
                        
                    fr = result_matrix[0]
                    fl = result_matrix[1]
                    bl = result_matrix[2]
                    br = result_matrix[3] 

                    rect2 = (10, 10, 40, 40)
                    annotator.box_label(rect2, str(silo_center), color=(0, 0, 0))
                    print("Silo Center : ", silo_center)
                    
                    if -3 <= silo_center <= 3:   # -3 3
                        print(" --------------- Aligned ---------------------")
                        print("Silo Center : ", silo_center)
                        fr = 0
                        bl = 0
                        near_far = -5
                    
                    # if box_height > 250 or box_width>150:  # 150 75 
                    #     fr = 0
                    #     fl = 0
                    #     bl = 0
                    #     br = 0
                    
                    fr /=1
                    fl /=1
                    br /=1
                    bl /=1                      

                    # Convert to bytes
                    data = (str(int(fr)) + '|' + 
                            str(int(fl)) + '|' +
                            str(int(bl)) + '|' +
                            str(int(br)) + '|' +
                            str(int(near_far))) + "#"
                        
                    # Send data
                    # time.sleep(0.05)
                    ser.write(data.encode())  
                    LOGGER.info(f"Sent 2: {data}") 
                
                elif ball_silo == 1  and prio_silo !=16:
                    fr = +0
                    fl = 0
                    bl = -0
                    br = 0 

                    near_far = -1 #  -1 : No Detection | -3 : Near | -4 : Far                   

                    # Convert to bytes
                    data = (str(int(fr)) + '|' + 
                            str(int(fl)) + '|' +
                            str(int(bl)) + '|' +
                            str(int(br)) + '|' +
                            str(int(near_far))) + "#"
                        
                    # Send data
                    # time.sleep(0.05)
                    ser.write(data.encode())  
                    LOGGER.info(f"Sent 6: {data}")
                      
                elif ball_silo == 0:
                    fr = bot_default_turn_speed_ball
                    fl = -bot_default_turn_speed_ball
                    bl = bot_default_turn_speed_ball
                    br = -bot_default_turn_speed_ball                   

                    # Convert to bytes
                    data = (str(int(fr)) + '|' + 
                            str(int(fl)) + '|' +
                            str(int(bl)) + '|' +
                            str(int(br)) + '|' +
                            str(int(nearest_c))) + "#"
                        
                    # Send data
                    # time.sleep(0.05)
                    ser.write(data.encode())  
                    LOGGER.info(f"Sent 5: {data}")
                    if delay_stat:
                        time.sleep(2)
                        delay_stat = False
            
            elif ball_silo == 1:
                fr = +0
                fl = 0
                bl = -0
                br = 0 

                near_far = -1 #  -1 : No Detection | -3 : Near | -4 : Far                   

                # Convert to bytes
                data = (str(int(fr)) + '|' + 
                        str(int(fl)) + '|' +
                        str(int(bl)) + '|' +
                        str(int(br)) + '|' +
                        str(int(near_far))) + "#"
                    
                # Send data
                # time.sleep(0.05)
                ser.write(data.encode())  
                LOGGER.info(f"Sent 7: {data}")


            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == "Linux" and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond
                # time.sleep(2)

        # Print time (inference-only)
        if len(det):
            pass              
        elif ball_silo == 0:
            fr = bot_default_turn_speed_ball
            fl = -bot_default_turn_speed_ball
            bl = bot_default_turn_speed_ball
            br = -bot_default_turn_speed_ball 

            near_far = -1 # -1 : No Detection | -3 : Near | -4 : Far                   

            # Convert to bytes
            data = (str(int(fr)) + '|' + 
                    str(int(fl)) + '|' +
                    str(int(bl)) + '|' +
                    str(int(br)) + '|' +
                    str(int(near_far))) + "#"
                
            # Send data
            # time.sleep(0.05)
            ser.write(data.encode())  
            LOGGER.info(f"Sent 8: {data}")
            if delay_stat:
                time.sleep(2)
                delay_stat = False
        elif ball_silo == 1:
            fr = +0
            fl = 0
            bl = -0
            br = 0 

            near_far = -1 #  -1 : No Detection | -3 : Near | -4 : Far                   

            # Convert to bytes
            data = (str(int(fr)) + '|' + 
                    str(int(fl)) + '|' +
                    str(int(bl)) + '|' +
                    str(int(br)) + '|' +
                    str(int(near_far))) + "#"
                
            # Send data
            # time.sleep(0.05)
            ser.write(data.encode())  
            LOGGER.info(f"Sent 9: {data}")

        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
    LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t) 


def parse_opt():
    """Parses command-line arguments for YOLOv5 detection, setting inference options and model configurations."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "best.pt", help="model path or triton URL")
    parser.add_argument("--source", type=str, default=cam_source, help="file/dir/URL/glob/screen/0(webcam)")
    parser.add_argument("--data", type=str, default=ROOT / "custom_data.yaml", help="(optional) dataset.yaml path")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument("--project", default=ROOT / "runs/detect", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")
    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
    parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    """Executes YOLOv5 model inference with given options, checking requirements before running the model."""
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    run(**vars(opt))


# if __name__ == "__main__":
opt = parse_opt()
main(opt)