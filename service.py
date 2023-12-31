import os
import math
import torch
import numpy as np
import cv2
import sys
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
# sys.path.append("commons/ai_models/yolov5")
import cvzone

# from utils.general import non_max_suppression
# from nebullvm.inference_learners.base import LearnerMetadata

weights = "head_model.pt"
device = "cpu"
conf_thres = 0.6  # confidence threshold
iou_thres = 0.5  # NMS IOU threshold
max_det = 1000  # maximum detections per image
classes = 1  # filter by class: --class 0, or --class 0 2 3
agnostic_nms = False  # class-agnostic NMS
width, height = 640, 480

# path_a = os.path.join("Services/people_entry_exit/resources/weights/head_lite")  # optimize yolo model from nebula
# model = LearnerMetadata.read(path_a).load_model(path_a)


# count_check_dict_in = {}
# count_check_dict_out = {}
in_count = 0
out_count = 0
posList = []
# model.fp16 = False
# model.device = device


def getAngle(a, b, c):
    """calculates angle between three points

    Args:
        a (int): point 1
        b (int): point 2
        c (int): point 3

    Returns:
        int: angle
    """
    ang = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
    total = ang + 360 if ang < 0 else ang
    return total



def counting_in_out_person_test(track_bbs_ids, frame, intrusion_in,scalex,scaley,count_check_dict_in):
    """counts entry and exit person count

    Args:
        track_bbs_ids (np.ndarray): bounding box with tracking id_
        frame (np.ndarray): image
        out_count (int): total out count
        in_count (int): total in count

    Returns:
        updated information out_count, in_count, frame
    """

    give_event = False
    bbox_output = []
    image_selection_bbox = []

    for current_sort_ids in track_bbs_ids:
        x1, y1, x2, y2, id_ = current_sort_ids
        # bbox_output = [{'top_left':(x1, y1), 'top_right':(x2, y1), 'bottom_left':(x1, y2), 'bottom_right':(x2, y2),'class':""}]

        x1n = (x2-x1)/2
        if id_ not in count_check_dict_in: 
            count_check_dict_in[id_] = False

        # intrusion_in
        # intrusion_in = []
        # fx = (x1+x1n, y2)
        # if len(intrusion_in):
        if "image_selection_params" in intrusion_in:


            for i in intrusion_in["image_selection_params"]:
                # print(i)

                a1,a2,a3,a4 = (int(i[0]['x']*scalex),int(i[0]['y']*scaley)), (int(i[1]['x']*scalex),int(i[1]['y']*scaley)), (int(i[2]['x']*scalex),int(i[2]['y']*scaley)), (int(i[3]['x']*scalex),int(i[3]['y']*scaley))
                # a1,a2,a3,a4 = (0,0),(2000,0),(2000,2000),(0,2000)
                image_selection_bbox.append([a1, a2, a3, a4])
                
                # pts = np.array([a1,a2,a3,a4]).reshape((-1, 1, 2))
                # cv2.polylines(frame, [pts],True, (3,186,252), 3)

                point = Point(x1+x1n, y2)
                polygon = Polygon([a1,a2,a3,a4])
                # print((getAngle(a1, fx, a2),getAngle(a2, fx, a3),getAngle(a3, fx, a4),getAngle(a4, fx, a1) ))
                if (polygon.contains(point)) and (count_check_dict_in[id_] == False):
                    bbox_output.append({'top_left':(x1, y1), 'top_right':(x2, y1), 'bottom_left':(x1, y2), 'bottom_right':(x2, y2),'class':""})
                    # cvzone.cornerRect(frame, [int(x1),int(y1),int(x2-x1   ),int(y2-y1   )], 15, 3, 2, colorC=(51,70,245) , colorR=(255, 255, 255))
                    count_check_dict_in[id_] = True
                    give_event = True
        elif count_check_dict_in[id_] == False:
            
            count_check_dict_in[id_] = True
            give_event = True
            # cvzone.cornerRect(frame, [int(x1),int(y1),int(x2-x1   ),int(y2-y1   )], 15, 3, 2, colorC=(51,70,245) , colorR=(255, 255, 255))
            bbox_output.append({
                'top_left':(x1, y1), 'top_right':(x2, y1), 'bottom_left':(x1, y2), 'bottom_right':(x2, y2),'class':""})

       
        # cv2.putText(frame, str(int(id_)), (int(x1) - 5, int(y1) - 5), 0, 0.55, (0, 255, 255), 2)
        

        # cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2, 8)
        # cvzone.putTextRect(
        #     frame_org,
        #     clsfr_class_names[class_pred] ,
        #     (int(x1), int(y1- 15)),
        #     scale=2,colorR=(51,70,245)
        # )
    return frame,give_event,bbox_output,image_selection_bbox

def letterbox_image(image, size):
    '''
    Resize image with unchanged aspect ratio using padding
    '''

    # original image size
    ih, iw, ic = image.shape

    # given size
    h, w = size

    # scale and new size of the image
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)
    
    # placeholder letter box
    new_image = np.zeros((h, w, ic), dtype='uint8') + 128

    # top-left corner
    top, left = (h - nh)//2, (w - nw)//2

    # paste the scaled image in the placeholder anchoring at the top-left corner
    new_image[top:top+nh, left:left+nw, :] = cv2.resize(image, (nw, nh))
    
    return new_image