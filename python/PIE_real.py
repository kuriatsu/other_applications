#! /usr/bin/env python3
# -*- coding:utf-8 -*-

from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet

icon_path = '/home/kuriatsu/share/'

icon_dict = {
    'walker_cross_to_left':
        {
        'path':icon_path + 'walker_cross_to_left.png'
        },
    'walker_cross_to_right':
        {
        'path':icon_path + 'walker_cross_to_right.png'
        },
    'walker_checked':
        {
        'path':icon_path + 'walker_checked.png'
        },
    'stop':
        {
        'path':icon_path + 'stop.png'
        }
    }


netMain = None
metaMain = None
altNames = None

image_res = [1080, 1900]
darknet_res = None
current_obj_positions = []
window_name = 'demo'
touch_flag = False


def prepareIcon():

    for icon_info in icon_dict.values():
        img = cv2.imread(icon_info.get('path'))
        icon_info['roi'] = img.shape[:2]
        img2grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2grey, 10, 255, cv2.THRESH_BINARY)
        # ret, mask = cv2.threshold(img2grey, 200, 255, cv2.THRESH_BINARY_INV)
        icon_info['mask_inv'] = cv2.bitwise_not(mask)
        icon_info['icon_fg'] = cv2.bitwise_and(img, img, mask=mask)


def prepareEventHandler():
    """add mouse click callback to the window
    """
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, touchCallback)


def touchCallback( event, x, y, flags, param):
    """if mouce clicked, check position and judge weather the position is on the rectange or not
    """
    global touch_flag#, current_obj_positions

    for current_obj in current_obj_positions:
        # if the event handler is leftButtonDown
        if event == cv2.EVENT_LBUTTONDOWN and current_obj[0] < x < current_obj[2] and current_obj[1] < y < current_obj[3]:
            touch_flag = not touch_flag
            print('touched')


def renderInfo(detections, image):
    """add information to the image

    """
    global touch_flag
    current_obj_positions.clear()
    # global current_obj_positions

    for detection in detections:
        if detection[0].decode() != 'person':
            continue

        xtl, ytl, xbr, ybr = convertBack(float(detection[2][0]),
                                         float(detection[2][1]),
                                         float(detection[2][2]),
                                         float(detection[2][3])
                                         )

        xbr = min(int(xbr * image_res[1] / darknet_res[1]),image_res[1])
        xtl = max(int(xtl * image_res[1] / darknet_res[1]), 0)
        ybr = min(int(ybr * image_res[0] / darknet_res[0]), image_res[0])
        ytl = max(int(ytl * image_res[0] / darknet_res[0]), 0)

        if touch_flag:
            color = (0, 255, 0)
            icon_info = icon_dict['walker_checked']

        else:
            color = (0, 0, 255)
            if xbr < image_res[1] * 0.5:
                icon_info = icon_dict['walker_cross_to_right']
            else:
                icon_info = icon_dict['walker_cross_to_left']

        icon_offset_y = 30.0
        icon_offset_x = int((icon_info['roi'][1] - (xbr - xtl)) * 0.5)

        # position of the icon
        icon_ytl = int(ytl - icon_info['roi'][0] - icon_offset_y)
        icon_xtl = int(xtl - icon_offset_x)
        icon_ybr = int(ytl - icon_offset_y)
        icon_xbr = int(xtl + icon_info['roi'][1] - icon_offset_x)

        # put icon on image
        try:
            roi = image[icon_ytl:icon_ybr, icon_xtl:icon_xbr] # get roi from image
            image_bg = cv2.bitwise_and(roi, roi, mask=icon_info['mask_inv']) # remove color from area for icon by filter
            buf = cv2.add(icon_info['icon_fg'], image_bg) # put icon of roi
            image[icon_ytl:icon_ybr, icon_xtl:icon_xbr] = buf # replace image region to roi

        except:
            print('icon is out of range y:{}-{}, x:{}-{}'.format(icon_ytl, icon_ybr, icon_xtl, icon_xbr))

        if not touch_flag:

            # position of the icon
            icon_info = icon_dict['stop']
            icon_ytl = int((ybr + ytl) / 2 - icon_info['roi'][0] / 2)
            icon_xtl = int(image_res[1] / 2 - icon_info['roi'][1] / 2)
            icon_ybr = int((ybr + ytl) / 2 + icon_info['roi'][0] / 2)
            icon_xbr = int(image_res[1] / 2 + icon_info['roi'][1] / 2)

            wall = image[ytl:ybr, 200:image_res[1]-200,:]
            wall_color = np.zeros((ybr-ytl, image_res[1]-400, 3), dtype=np.uint8)
            wall_color[:] = (255, 255, 255)
            # put icon on image
            try:
                image[ytl:ybr, 200:image_res[1]-200] = cv2.addWeighted(wall, 0.5, wall_color, 0.5, 0)
                roi = image[icon_ytl:icon_ybr, icon_xtl:icon_xbr] # get roi from image
                image_bg = cv2.bitwise_and(roi, roi, mask=icon_info['mask_inv']) # remove color from area for icon by filter
                buf = cv2.add(icon_info['icon_fg'], image_bg) # put icon of roi
                image[icon_ytl:icon_ybr, icon_xtl:icon_xbr] = buf # replace image region to roi

            except:
                print('icon is out of range y:{}-{}, x:{}-{}'.format(icon_ytl, icon_ybr, icon_xtl, icon_xbr))


        # cv2.putText(
        #     image,
        #     # 'Cross?',
        #     # '{:.01f}'.format((focused_obj_info['xbr'] - focused_obj_info['xtl']) * (focused_obj_info['ybr'] - focused_obj_info['ytl'])),
        #     '{:.01f}%'.format(focused_obj_info['prob'] * 100),
        #     # '{:.01f}s'.format((focused_obj_info['critical_point'] - self.current_frame_num) / self.video_rate),
        #     (int(focused_obj_info['xtl']), int(focused_obj_info['ytl']) - 10),
        #     cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3, cv2.LINE_AA
        #     )

        current_obj_positions.append([xtl, ytl, xbr, ybr])
        image = cv2.rectangle(image, (xtl, ytl), (xbr, ybr), color, 6)

    return image


def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cvDrawBoxes(detections, img):
    for detection in detections:
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
        cv2.putText(img,
                    detection[0].decode() +
                    " [" + str(round(detection[1] * 100, 2)) + "]",
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2)
    return img


def YOLO():

    global metaMain, netMain, altNames
    configPath = "./cfg/yolov3.cfg"
    weightPath = "./yolov3-320.weights"
    metaPath = "./cfg/coco.data"
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass


    cap = cv2.VideoCapture(2)
    global image_res, darknet_res
    image_res = [image_res[0], int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)*(image_res[0] / cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))]

    prepareIcon()
    prepareEventHandler()

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(darknet.network_width(netMain),
                                    darknet.network_height(netMain),3)

    darknet_res = [darknet.network_height(netMain), darknet.network_width(netMain)]

    print("Starting the YOLO loop...")
    while True:
        prev_time = time.time()
        ret, frame_read = cap.read()
        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb,
                                   (darknet.network_width(netMain),
                                    darknet.network_height(netMain)),
                                   interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())

        detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)
        frame_read = cv2.resize(frame_read, (image_res[1], image_res[0]), interpolation=cv2.INTER_LINEAR)
        frame_read = renderInfo(detections, frame_read)
        # image = cvDrawBoxes(detections, frame_resized)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(1/(time.time()-prev_time))
        cv2.imshow(window_name, frame_read)
        cv2.waitKey(3)
    cap.release()

if __name__ == "__main__":
    YOLO()
