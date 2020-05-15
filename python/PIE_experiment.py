#! /usr/bin/python3
# -*- coding: utf-8 -*-
"""
welcome to PIE data visualize

"""

import cv2
import argparse
import numpy as np
import time
import xml.etree.ElementTree as ET


class PieDataVisualize(object):

    def __init__(self, args):
        self.window_name = 'frame'
        self.video = None
        self.video_rate = None
        self.attrib_tree = None
        self.pie_data = {}
        self.frame = None
        self.displayed_obj = [[0]]
        self.focused_obj_id = None
        self.icon = None
        self.icon_roi = None
        self.icon_fg = None


    def prepareIcon(self):
        img = cv2.imread('/home/kuriatsu/share/cross_small.png')
        self.icon_roi = img.shape[:2]
        img2grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2grey, 10, 255, cv2.THRESH_BINARY)
        # ret, mask = cv2.threshold(img2grey, 200, 255, cv2.THRESH_BINARY_INV)
        self.mask_inv = cv2.bitwise_not(mask)
        self.icon_fg = cv2.bitwise_and(img, img, mask=mask)


    def getVideo(self, video_file, rate_offset):
        try:
            self.video = cv2.VideoCapture(video_file)
            self.video_rate = int(self.video.get(cv2.CAP_PROP_FPS) + rate_offset)
        except:
            print('cannot open video')
            exit(0)


    def getAttrib(self, attrib_file):
        try:
            tree = ET.parse(attrib_file)
            self.attrib_tree = tree.getroot()
            del tree
        except:
            print('cannot open attrib file')
            exit(0)


    def getAnno(self, anno_file):
        try:
            tree = ET.parse(anno_file)
            root = tree.getroot()
        except:
            print('cannot open annotation file')
            exit(0)

        for track in root.findall('track'):

            for anno_itr in track.iter('box'):

                anno_info = {}

                # get id
                for attribute in anno_itr.findall('attribute'):
                    if attribute.attrib.get('name') == 'id':
                        anno_id = attribute.text

                # get basic information
                anno_info['label'] = track.attrib.get('label')
                anno_info['xbr'] = anno_itr.attrib.get('xbr')
                anno_info['xtl'] = anno_itr.attrib.get('xtl')
                anno_info['ybr'] = anno_itr.attrib.get('ybr')
                anno_info['ytl'] = anno_itr.attrib.get('ytl')

                # if object is pedestrian, get additional information
                if anno_info['label'] == 'pedestrian':
                    for attrib_itr in self.attrib_tree.iter('pedestrian'):
                        if attrib_itr.attrib.get('id') == anno_id:
                            anno_info['intention_prob'] = attrib_itr.attrib.get('intention_prob')
                            anno_info['critical_point'] = attrib_itr.attrib.get('critical_point')
                            anno_info['crossing_point'] = attrib_itr.attrib.get('crossing_point')
                            anno_info['exp_start_point'] = attrib_itr.attrib.get('exp_start_point')

                # add to pie_data dictionary
                if anno_itr.attrib.get('frame') not in self.pie_data:
                    self.pie_data[anno_itr.attrib.get('frame')] = {}

                self.pie_data[anno_itr.attrib.get('frame')][anno_id] = anno_info

        # delete objects to improve performance
        del root
        del tree
        del self.attrib_tree


    def prepareEventHandler(self):
        """add mouse click callback to the window
        """
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.touchCallback)


    def touchCallback(self, event, x, y, flags, param):
        """if mouce clicked, check position and judge weather the position is on the rectange or not
        """
        # if the event handler is leftButtonDown and current frame contains objects
        if event == cv2.EVENT_LBUTTONDOWN and self.frame in self.pie_data:

            for obj_id, obj_info in self.pie_data[self.frame].items():

                # if the clicked position is on the rectangle of one of the objects
                if int(float(obj_info['xtl'])) < x < int(float(obj_info['xbr'])) and int(float(obj_info['ytl'])) < y < int(float(obj_info['ybr'])):

                    # if the clicked position is on the focuced object
                    if self.displayed_obj[obj_id]['is_forcused']:

                        # update "is_forcused" in self.dislpayed_obj
                        self.updateFocusedObject(obj_id)
                        return


    def pushCallback(self):
        """callback of enter key push, target is focused object
        """
        for obj_id, obj_info in self.displayed_obj.items():
            if obj_info['is_forcused']:
                self.updateFocusedObject(obj_id)
                return


    def updateFocusedObject(self, checked_obj_id=None):
        """find focused object from self.displayed_obj
        """
        min_obj_id = None # initial variable for searching new imporant objects
        min_time = 20000 # initial variable for searching new imporant objects

        # is this method called by callback, checked_obj_id has target object id which is checked and should be unfocused
        if checked_obj_id is not None:
            self.displayed_obj[checked_obj_id]['is_checked'] = True
            self.displayed_obj[checked_obj_id]['is_forcused'] = False

        # find new focused object
        for obj_id, obj_info in self.displayed_obj.items():
            # search the new forcused obj
            if not obj_info['is_checked'] and not obj_info['is_passed'] and int(self.pie_data[self.frame][obj_id]['critical_point']) < min_time:
                min_obj_id = obj_id
                min_time = int(self.pie_data[self.frame][obj_id]['critical_point'])

        # if the new forcused obj is found, add the flag
        if min_obj_id is not None:
            self.displayed_obj[min_obj_id]['is_forcused'] = True


    def refleshDisplayObjects(self):
        """magage displaying object
        is_checked ; flag wether the subject check obj and input some action or not
        """
        display_obj = {} # new container
        is_forcused_obj_exist = False # flag

        for obj_id, obj_info in self.pie_data[self.frame].items():

            if obj_info['label'] != 'pedestrian': continue

            # if the obj was already displayed
            if obj_id in self.displayed_obj:

                display_obj[obj_id] = self.displayed_obj[obj_id]
                display_obj[obj_id]['is_passed'] = int(obj_info['critical_point']) < int(self.frame)

                # if the object was forcused
                if self.displayed_obj[obj_id]['is_forcused']:
                    is_forcused_obj_exist = True

            # if the obj is new
            else:
                display_obj[obj_id] = {'is_forcused':False, 'is_checked':False, 'is_passed':False}

        # reflesh displaying object container
        self.displayed_obj = display_obj

        # if the forcused object is not checked, don't search new forcused obj
        if not is_forcused_obj_exist:
            self.updateFocusedObject()


    def renderInfo(self, image):
        """add information to the image

        """
        # loop for each object in the frame from PIE dataset
        for obj_id, displayed_obj_info in self.displayed_obj.items():

            obj_info = self.pie_data[self.frame][obj_id]

            if displayed_obj_info['is_forcused']: # if forcused --- red
                color = (0, 0, 255)
                self.drawIcon(image, obj_info)

                cv2.putText(
                    image,
                    'Cross?',
                    (int(float(obj_info['xtl'])), int(float(obj_info['ytl'])) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
                    )

            elif displayed_obj_info['is_checked']: # if checked --- green
                color = (0, 255, 0)

            else: # else --- blue
                color = (255, 0, 0)

            # cv2.putText(
            #     image,
            #     '{:.01f}%'.format(float(obj_info['intention_prob']) * 100),
            #     (int(float(obj_info['xtl'])), int(float(obj_info['ytl'])) - 10),
            #     cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
            #     )

            image = cv2.rectangle(image,
                                  (int(float(obj_info['xtl'])), int(float(obj_info['ytl']))),
                                  (int(float(obj_info['xbr'])), int(float(obj_info['ybr']))),
                                  color,
                                  1)


    def drawIcon(self, image, obj_info):
        """draw icon to emphasize the target objects
        image : image
        obj_info : PIE dataset info of the object in the frame
        """

        icon_offset_y = 30.0
        icon_offset_x = int((self.icon_roi[1] - (float(obj_info['xbr']) - float(obj_info['xtl']))) * 0.5)

        # position of the icon
        icon_ytl = int(float(obj_info['ytl']) - self.icon_roi[0] - icon_offset_y)
        icon_xtl = int(float(obj_info['xtl']) - icon_offset_x)
        icon_ybr = int(float(obj_info['ytl']) - icon_offset_y)
        icon_xbr = int(float(obj_info['xtl']) + self.icon_roi[1] - icon_offset_x)

        # put icon on image
        try:
            roi = image[icon_ytl:icon_ybr, icon_xtl:icon_xbr] # get roi from image
            image_bg = cv2.bitwise_and(roi, roi, mask=self.mask_inv) # remove color from area for icon by filter
            buf = cv2.add(self.icon_fg, image_bg) # put icon of roi
            image[icon_ytl:icon_ybr, icon_xtl:icon_xbr] = buf # replace image region to roi

        except:
            print('icon is out of range y:{}-{}, x:{}-{}'.format(icon_ytl, icon_ybr, icon_xtl, icon_xbr))


    def loop(self):
        print('start_loop')

        sleep_time = self.video_rate
        frame = 0

        while(self.video.isOpened()):

            start = time.time()
            self.frame = str(frame)
            ret, image = self.video.read()

            if self.frame in self.pie_data:
                self.refleshDisplayObjects() # udpate self.displayed_obj
                self.renderInfo(image) # add info to the image

            cv2.imshow(self.window_name, image) # render

            #  calc sleep time to keep frame rate to be same with video rate
            sleep_time = max(int((1000 / (self.video_rate) - (time.time() - start))), 1)

            # sleep and wait quit key
            key = cv2.waitKey(sleep_time) & 0xFF
            if key is not 255 : print(key)
            if key == ord('q'):
                break
            if key == 13:
                self.pushCallback()

            frame += 1


    def __del__(self):
        print('delete instance...')
        self.video.release()
        cv2.destroyAllWindows()


def main():
    argparser = argparse.ArgumentParser( description = __doc__)
    argparser.add_argument(
        '--video', '-v',
        metavar='VIDEO',
        default='/media/ssd/PIE_data/PIE_clips/set01/video_0001.mp4')
    argparser.add_argument(
        '--anno',
        metavar='ANNO',
        default='/media/ssd/PIE_data/annotations/set01/video_0001_annt.xml')
    argparser.add_argument(
        '--attrib',
        metavar='ATTRIB',
        default='/media/ssd/PIE_data/annotations_attributes/set01/video_0001_attributes.xml')
    argparser.add_argument(
        '--rate_offset',
        metavar='OFFSET',
        default=11)

    args = argparser.parse_args()

    pie_data_visualize = PieDataVisualize(args)
    pie_data_visualize.getVideo(args.video, args.rate_offset)
    pie_data_visualize.getAttrib(args.attrib)
    pie_data_visualize.getAnno(args.anno)
    pie_data_visualize.prepareIcon()
    pie_data_visualize.prepareEventHandler()
    # print(pie_data_visualize.pie_data.get('1133'))
    pie_data_visualize.loop()

if __name__ == '__main__':
    main()
